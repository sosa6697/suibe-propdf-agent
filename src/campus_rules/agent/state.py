"""Agent 状态、决策与轨迹对象。

边界说明：raw_query 与改写后的查询只在内存中保留，供本轮决策使用。
trace 导出默认脱敏（不落盘用户原始提示词）——这是后续需求
「用户输入提示词数据采集」的预留边界，本期不实现任何采集逻辑。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..schema import Evidence, ScoredChunk, evidence_to_dict


class StopReason(str, Enum):
    VERIFIED = "verified"          # 已生成答案并通过引用校验
    REFUSED = "refused"            # 主动拒答（证据不足或工具判定）
    GUARD_BLOCKED = "guard_blocked"  # v0.5 入口过滤：域外问题，未进入循环
    MAX_STEPS = "max_steps"        # 达到步数上限，强制收敛
    POLICY_FALLBACK = "policy_fallback"
    ERROR = "error"                # 连续工具异常


@dataclass(slots=True)
class Decision:
    thought: str
    action: str
    args: dict[str, Any] = field(default_factory=dict)
    stop: bool = False
    reason: str = ""


@dataclass(slots=True)
class ToolCall:
    step: int
    action: str
    args: dict[str, Any]
    thought: str
    observation: str
    ok: bool
    ms: float

    def to_dict(self, *, redact_query: bool = True) -> dict[str, Any]:
        args = dict(self.args)
        if redact_query and "query" in args:
            raw = str(args["query"])
            args["query"] = {
                "redacted": True,
                "chars": len(raw),
                "sha8": hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8],
            }
        return {
            "step": self.step,
            "action": self.action,
            "args": args,
            "thought": self.thought,
            "observation": self.observation,
            "ok": self.ok,
            "ms": self.ms,
        }


@dataclass(slots=True)
class AgentState:
    raw_query: str                      # 内存中保留原始用户输入，不覆盖
    session_id: str
    decision_mode: str = "llm"          # llm | policy
    rewritten_queries: list[str] = field(default_factory=list)
    hits: dict[str, list[ScoredChunk]] = field(default_factory=dict)
    candidates: list[ScoredChunk] = field(default_factory=list)
    all_candidates: list[ScoredChunk] = field(default_factory=list)  # 精排前的完整候选池
    evidences: list[Evidence] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    step: int = 0
    draft: str = ""
    answer: str = ""
    # 生成后核查不通过时，带核查意见重生成的次数（确定性自我修正留痕）
    answer_retries: int = 0
    refused: bool = False
    degraded: bool = False
    degrade_reasons: list[str] = field(default_factory=list)
    # 红线拒答是预期行为，不算能力降级；单独留痕避免 degraded_rate 失真
    redline_refusals: list[str] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    relevance: dict[str, Any] = field(default_factory=dict)
    stop_reason: StopReason | None = None
    api_calls: int = 0                  # 本次问答发起的模型调用次数
    policy_fallbacks: int = 0           # LLM 决策失败回退确定性策略的次数
    ablations: set[str] = field(default_factory=set)
    # v0.5 入口过滤：命中域外信号时循环根本不启动，guard 只含规则名与固定文案
    guard_blocked: bool = False
    guard: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_query(self) -> str:
        """检索用的查询：多轮场景下可能是「上一轮＋本轮」的拼接词。"""
        return self.rewritten_queries[-1] if self.rewritten_queries else self.raw_query

    @property
    def answer_query(self) -> str:
        """生成与核查共同使用的问题：始终是用户本轮真正问的那一句。

        刻意不复用 effective_query —— 后者是为检索拼出来的，把它交给核查器会
        让「答案是否回答了问题」多出历史问题的范围，历史上造成 4 道多轮题误拒。
        """
        return self.raw_query

    def state_summary(self) -> str:
        return (
            f"候选 {len(self.candidates)} 块 / 证据 {len(self.evidences)} 条 / "
            f"版本冲突 {len(self.conflicts)} 条 / 已生成答案 {'是' if self.answer else '否'} / "
            f"已用 {self.step} 步"
        )

    def observation_summary(self, limit: int = 6) -> str:
        if not self.tool_calls:
            return "（尚无执行记录）"
        return "\n".join(
            f"  {t.step}. {t.action} -> {t.observation[:160]}" for t in self.tool_calls[-limit:]
        )

    def to_dict(self, *, include_raw_query: bool = False) -> dict[str, Any]:
        """序列化给 CLI / API / trace。默认不输出用户原始提示词全文。

        include_raw_query 是后续「提示词数据采集」需求的预留入口，
        当前所有调用方均不启用；启用与否由调用方显式决定，不做隐式落盘。
        """
        query_info: dict[str, Any] = {
            "chars": len(self.raw_query),
            "sha8": hashlib.sha256(self.raw_query.encode("utf-8")).hexdigest()[:8],
            "rewrite_count": len(self.rewritten_queries),
        }
        if include_raw_query:
            query_info["raw"] = self.raw_query
            query_info["rewrites"] = list(self.rewritten_queries)
        return {
            "answer": self.answer,
            "refused": self.refused,
            "session_id": self.session_id,
            "decision_mode": self.decision_mode,
            "degraded": self.degraded,
            "degrade_reasons": self.degrade_reasons,
            "redline_refusals": self.redline_refusals,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "api_calls": self.api_calls,
            "policy_fallbacks": self.policy_fallbacks,
            "answer_retries": self.answer_retries,
            "steps": self.step,
            "ablations": sorted(self.ablations),
            "guard_blocked": self.guard_blocked,
            "guard": self.guard,
            "query": query_info,
            "evidences": [evidence_to_dict(e) for e in self.evidences],
            "version_conflicts": self.conflicts,
            "verification": self.verification,
            "relevance": self.relevance,
            "trace": [t.to_dict(redact_query=not include_raw_query) for t in self.tool_calls],
        }
