"""Agent 主循环：决策 → 执行 → 观察 → 再判断，带护栏与自我修正。

决策器每轮输出一个结构化 JSON；解析失败 / 接口失败 / 未知工具时
自动回退到 policy.default_plan 的确定性策略，并显式标注 degraded。
"""
from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from ..adapters.errors import AdapterError
from ..logging_conf import get_logger
from ..pipeline import REFUSAL
from ..query_guard import GUARD_REFUSAL, guard_query
from ..schema import QueryRequest
from .policy import default_plan
from .state import AgentState, Decision, StopReason, ToolCall
from .tools import ToolRegistry

if TYPE_CHECKING:
    from ..pipeline import QAPipeline

log = get_logger("agent.loop")

DECISION_SYSTEM = """你是校规问答 Agent 的调度器。每一步根据当前状态选择一个工具，只输出一个 JSON 对象，不要输出任何其它文字。

策略：
1. 还没有候选 -> retrieve
2. 有候选但未精排 -> rerank
3. 命中多个发布年份或历年汇编 -> check_version
4. 证据不足 / 引用校验不通过 -> 用改写后的 query 再 retrieve（不要重复完全相同的参数）
5. 证据充分 -> pack_evidence -> answer
6. 确认无法回答 -> refuse
7. 答案已生成且通过引用校验 -> stop=true

JSON 格式：{"thought":"简短说明判断依据","action":"工具名","args":{...},"stop":false,"reason":"..."}
只输出 JSON，不要 Markdown 代码块。"""


def build_decision_user(state: AgentState, specs_text: str, max_steps: int) -> str:
    return (
        f"【用户问题】\n{state.effective_query}\n\n"
        f"【工具清单】\n{specs_text}\n\n"
        f"【当前状态】\n{state.state_summary()}，剩余 {max_steps - state.step} 步\n\n"
        f"【已执行步骤】\n{state.observation_summary()}\n\n"
        f"【版本冲突】\n{'；'.join(state.conflicts[:2]) or '无'}\n\n"
        "只输出一个决策 JSON。"
    )


class AgentLoop:
    def __init__(
        self,
        pipeline: "QAPipeline",
        *,
        decision_mode: str = "llm",
        max_steps: int = 6,
        ablations: tuple[str, ...] | list[str] = (),
    ):
        self.p = pipeline
        self.s = pipeline.s
        self.decision_mode = decision_mode
        self.max_steps = max(1, max_steps)
        self.ablations = set(ablations)
        self.registry = ToolRegistry(pipeline)

    @staticmethod
    def _parse_decision(raw: str) -> Decision:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("决策响应中没有 JSON 对象")
        payload: dict[str, Any] = json.loads(raw[start : end + 1])
        action = str(payload.get("action") or "").strip()
        stop = bool(payload.get("stop", False))
        if not action and not stop:
            raise ValueError("决策缺少 action 且未要求 stop")
        args = payload.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError("args 必须是对象")
        return Decision(
            thought=str(payload.get("thought") or "")[:400],
            action=action,
            args=args,
            stop=stop,
            reason=str(payload.get("reason") or "")[:200],
        )

    def _decide(self, state: AgentState) -> Decision:
        if state.decision_mode == "policy":
            return default_plan(state)
        try:
            state.api_calls += 1
            raw = self.p.llm.chat(
                system=DECISION_SYSTEM,
                user=build_decision_user(state, self.registry.specs_text(), self.max_steps),
                temperature=0.0,
                timeout=60,
            )
            decision = self._parse_decision(raw)
            if not decision.stop and decision.action not in self.registry.tools:
                raise ValueError(f"未知工具：{decision.action}")
            return decision
        except (AdapterError, ValueError, json.JSONDecodeError) as exc:
            state.policy_fallbacks += 1
            state.degraded = True
            state.degrade_reasons.append(f"LLM 决策不可用，回退规则策略：{exc}")
            log.warning("决策回退：%s", exc)
            return default_plan(state)

    @staticmethod
    def _converge_to_refusal(state: AgentState, reason: str) -> None:
        state.answer = REFUSAL
        state.refused = True
        state.stop_reason = StopReason.REFUSED
        state.degraded = True
        state.degrade_reasons.append(reason)

    @staticmethod
    def _guard_blocked_state(state: AgentState, guard) -> AgentState:
        """入口命中：循环一步都不启动，但轨迹里仍留一条可审计的 query_guard 记录。

        stop_reason 单独取 GUARD_BLOCKED，方便评测把「入口拦截」与
        「检索后证据不足拒答（REFUSED）」分开统计；degraded 保持 False。
        与 baseline 一致：不写会话记忆，避免域外话题污染多轮上下文。
        """
        t0 = time.perf_counter()
        state.answer = GUARD_REFUSAL
        state.refused = True
        state.stop_reason = StopReason.GUARD_BLOCKED
        state.guard_blocked = True
        state.guard = guard.to_dict()
        state.redline_refusals.append(guard.reason)
        state.verification = {
            "ok": True, "method": "skipped", "issues": [], "warnings": [],
            "guard_blocked": True,
        }
        state.tool_calls.append(
            ToolCall(
                step=0,
                action="query_guard",
                args={},
                thought="入口层确定性域外判定，不进入决策循环与任何模型调用",
                observation=f"blocked rules={','.join(guard.matched_rules) or '-'}",
                ok=True,
                ms=round((time.perf_counter() - t0) * 1000, 2),
            )
        )
        log.info("Agent 入口过滤：step=0 api_calls=0 stop=%s", state.stop_reason.value)
        return state

    def run(self, req: QueryRequest) -> AgentState:
        state = AgentState(
            raw_query=req.raw_query,
            session_id=req.session_id,
            decision_mode=self.decision_mode,
            ablations=set(self.ablations),
        )
        # v0.5 入口过滤：放在状态初始化之后、任何决策 / 工具 / 适配器调用之前。
        guard = guard_query(req.raw_query)
        if guard.blocked:
            return self._guard_blocked_state(state, guard)

        # 静态降级原因必须在起点就带上：与固定流水线一致，绝不留到出错时才补
        static_reasons = list(self.s.degrade_reasons())
        state.degrade_reasons.extend(static_reasons)
        state.degraded = bool(static_reasons)

        # 决策器本身依赖 LLM；LLM 缺失时不逐轮空转调用，直接切确定性策略并留痕
        if state.decision_mode == "llm" and not self.s.llm_ready:
            state.decision_mode = "policy"
            state.degraded = True
            state.degrade_reasons.append(
                "LLM 未配置：决策器不可用，强制切换为确定性策略（policy）"
            )
        # 注意：policy 决策模式与消融开关是调用方显式选择的「正常配置」，
        # 不是能力缺失 / 调用失败，因此不计入 degraded —— 否则评测里
        # agent/policy 组的 degraded_rate 会恒为 1.0，指标失去意义。
        # 两者分别由 state.decision_mode 与 state.ablations 字段追溯。

        seen: set[str] = set()
        consecutive_failures = 0

        while state.step < self.max_steps and state.stop_reason is None:
            decision = self._decide(state)

            if decision.stop:
                if state.answer:
                    state.stop_reason = StopReason.VERIFIED
                else:
                    self._converge_to_refusal(state, "决策器要求停止但未产出答案")
                break

            key = f"{decision.action}::{json.dumps(decision.args, sort_keys=True, ensure_ascii=False)}"
            if key in seen:
                state.degraded = True
                state.degrade_reasons.append(f"重复动作 {decision.action}，触发收敛护栏")
                decision = Decision(
                    thought="[护栏] 检测到重复动作，直接进入收敛路径",
                    action="answer" if state.evidences else "refuse",
                    args={} if state.evidences else {"reason": "重复动作，无法继续推进"},
                    reason="duplicate_guard",
                )
                key = (
                    f"{decision.action}::"
                    f"{json.dumps(decision.args, sort_keys=True, ensure_ascii=False)}"
                )
            seen.add(key)

            t0 = time.perf_counter()
            try:
                observation = self.registry.run(decision.action, state, decision.args)
                ok = True
                consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001 - 工具异常必须让决策器看见
                observation = f"工具异常：{type(exc).__name__}: {exc}"
                ok = False
                consecutive_failures += 1
                state.degraded = True
                state.degrade_reasons.append(f"工具 {decision.action} 执行异常：{exc}")
            state.tool_calls.append(
                ToolCall(
                    step=state.step + 1,
                    action=decision.action,
                    args=decision.args,
                    thought=decision.thought,
                    observation=observation[:600],
                    ok=ok,
                    ms=round((time.perf_counter() - t0) * 1000, 2),
                )
            )
            state.step += 1

            if consecutive_failures >= 2:
                state.stop_reason = StopReason.ERROR
                break

        if state.stop_reason is None:
            # 决策器退化过且始终没收敛，单独标记，便于评测里区分「答不出」和「决策器坏了」
            state.stop_reason = (
                StopReason.POLICY_FALLBACK
                if state.policy_fallbacks and self.decision_mode == "llm"
                else StopReason.MAX_STEPS
            )
        if not state.answer:
            state.answer = REFUSAL
            state.refused = True
            state.degraded = True
            state.degrade_reasons.append("循环结束仍未产出已验证答案，强制收敛为拒答")

        # 短期记忆：写用户原始输入与最终回复；不额外采集任何提示词数据
        self.p.memory.append(req.session_id, "user", req.raw_query)
        self.p.memory.append(
            req.session_id,
            "assistant",
            state.answer,
            meta={
                "refused": state.refused,
                "degraded": state.degraded,
                "mode": state.decision_mode,
                "stop_reason": state.stop_reason.value if state.stop_reason else None,
            },
        )
        log.info(
            "Agent 完成：mode=%s steps=%d api_calls=%d stop=%s refused=%s",
            state.decision_mode, state.step, state.api_calls,
            state.stop_reason.value if state.stop_reason else "-", state.refused,
        )
        return state
