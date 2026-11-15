"""工具注册表：把现有纯函数包装成 Agent 可调用的工具。

每个工具只做一件事、返回一条 observation 摘要，副作用统一写回 AgentState。
所有外部调用都会在 state.api_calls 里计数，便于控制成本与做消融统计。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from ..adapters.errors import AdapterError
from ..index.hybrid import (
    apply_recency_bias,
    detect_version_conflicts,
    expand_neighbors,
    rrf_fuse,
    rule_rerank,
)
from ..logging_conf import get_logger
from ..pipeline import REFUSAL
from ..prompts import ANSWER_SYSTEM, build_answer_user
from ..relevance import assess_relevance
from ..schema import Evidence
from ..verification import verify_answer
from .state import AgentState, StopReason

if TYPE_CHECKING:
    from ..pipeline import QAPipeline

log = get_logger("agent.tools")

# 只在出现指代词 / 省略式追问时才把上一轮问题并入检索词。
# 首轮评测发现无条件补全会稀释当前意图（M03 因此未命中）。
_NEEDS_CONTEXT = re.compile(r"^(那|那么|则|它|这个|那个)|呢[？?]?$|它|其|上述|前述")


def _degrade(state: AgentState, reason: str) -> None:
    state.degraded = True
    state.degrade_reasons.append(reason)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    args_schema: dict[str, str]
    fn: Callable[[AgentState, dict[str, Any]], str]


class ToolRegistry:
    def __init__(self, pipeline: "QAPipeline"):
        self.p = pipeline
        self.s = pipeline.s
        self.tools: dict[str, ToolSpec] = {}
        self._register()

    # ------------------------------------------------------------------ 注册表
    def _add(self, name: str, description: str, args_schema: dict[str, str], fn) -> None:
        self.tools[name] = ToolSpec(name, description, args_schema, fn)

    def specs(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "args": t.args_schema}
            for t in self.tools.values()
        ]

    def specs_text(self) -> str:
        lines = []
        for t in self.tools.values():
            args = ", ".join(f"{k}:{v}" for k, v in t.args_schema.items()) or "无参数"
            lines.append(f"- {t.name}({args}) : {t.description}")
        return "\n".join(lines)

    def run(self, name: str, state: AgentState, args: dict[str, Any]) -> str:
        spec = self.tools.get(name)
        if spec is None:
            raise KeyError(f"未知工具：{name}；可用：{list(self.tools)}")
        return spec.fn(state, args or {})

    def _register(self) -> None:
        self._add(
            "retrieve",
            "双路召回（向量 + BM25）→ RRF 融合 → 邻块扩展 → 规则预筛。"
            "证据不足时可换用改写后的 query 再调用一次。",
            {"query": "str 可选，改写后的查询", "use_vector": "bool 默认 true",
             "use_bm25": "bool 默认 true", "top_k": "int 默认配置值"},
            self._retrieve,
        )
        self._add(
            "rerank", "对候选做 API 精排；不可用时保留规则预筛结果",
            {"query": "str 可选"}, self._rerank,
        )
        self._add(
            "check_version", "检测命中文件里是否存在同一规章的新旧版本冲突",
            {}, self._check_version,
        )
        self._add(
            "pack_evidence", "把排名靠前的候选打包成带 [E编号] 的证据",
            {"top_n": "int 默认 FINAL_EVIDENCE"}, self._pack_evidence,
        )
        self._add(
            "answer", "依据证据生成答案并做生成后引用校验；不通过会返回原因供决策器修正",
            {}, self._answer,
        )
        self._add(
            "refuse", "证据不足时按红线拒答，终止循环",
            {"reason": "str 拒答原因"}, self._refuse,
        )

    # ------------------------------------------------------------------ 工具实现
    def _retrieve(self, state: AgentState, args: dict[str, Any]) -> str:
        explicit = args.get("query")
        query = str(explicit or state.effective_query)
        use_vector = bool(args.get("use_vector", True)) and "no_vector" not in state.ablations
        use_bm25 = bool(args.get("use_bm25", True))
        top_k = int(args.get("top_k", self.s.top_k_bm25))
        rank_lists = []
        notes = []

        # 多轮补全：只在明确出现指代词 / 省略式追问时，把上一轮用户问题并入检索词。
        # 只对首次检索生效，避免自我修正重检索时重复拼接；纯本地、零 API。
        if not explicit and state.step == 0 and _NEEDS_CONTEXT.search(query):
            prev = [
                t.content for t in self.p.memory.history(state.session_id)
                if t.role == "user"
            ]
            if prev:
                query = f"{prev[-1]} {query}"[:300]
                notes.append("多轮上下文补全（指代/省略信号）")

        if use_vector and self.s.enable_vector_recall:
            if self.s.embedding_ready:
                try:
                    state.api_calls += 1
                    vec = self.p.embedding.embed(query)
                    hits = self.p.store.search(vec, self.s.top_k_vector)
                    rank_lists.append(hits)
                    state.hits["vector"] = hits
                    notes.append(f"向量 {len(hits)}")
                except (AdapterError, ValueError) as exc:
                    _degrade(state, f"向量召回失败：{exc}")
                    notes.append("向量失败")
            else:
                notes.append("向量未配置")
        if use_bm25:
            hits = self.p.bm25.search(query, top_k, self.p.store.chunks)
            rank_lists.append(hits)
            state.hits["bm25"] = hits
            notes.append(f"BM25 {len(hits)}")

        fused = rrf_fuse(rank_lists, self.s.rrf_k)
        candidates = (
            fused if "no_expand" in state.ablations
            else expand_neighbors(self.p.store, fused, self.s.neighbor_window)
        )
        state.candidates = rule_rerank(query, candidates, max_per_doc=self.s.max_per_doc)
        # 保留精排前的完整候选池：版本冲突检测需要看池子全貌，而不是 top8
        state.all_candidates = list(state.candidates)
        if query != state.raw_query:
            state.rewritten_queries.append(query)
        return (
            f"召回[{'、'.join(notes)}] → 融合 {len(fused)} → 扩展 {len(candidates)} "
            f"→ 规则预筛 {len(state.candidates)} 块"
        )

    def _rerank(self, state: AgentState, args: dict[str, Any]) -> str:
        query = str(args.get("query") or state.effective_query)
        if not state.candidates:
            return "无可重排候选，请先 retrieve"
        cap = max(self.s.rerank_top_n * 5, 30)
        to_rerank = state.candidates[:cap]
        if self.s.rerank_ready and "no_rerank" not in state.ablations:
            try:
                state.api_calls += 1
                ranked = self.p.rerank.rerank(query, to_rerank)
                state.candidates = apply_recency_bias(ranked)
                return (
                    f"API rerank：送入 {len(to_rerank)} 块 → 返回 {len(ranked)} 块；"
                    "已叠加汇编降权与时效纠偏"
                )
            except AdapterError as exc:
                _degrade(state, f"rerank API 失败，保留规则预筛结果：{exc}")
        return f"规则重排（未调用 API）：保留 {len(state.candidates)} 块"

    def _check_version(self, state: AgentState, args: dict[str, Any]) -> str:
        pool = state.all_candidates or state.candidates
        state.conflicts = detect_version_conflicts(pool[:40])
        if not state.conflicts:
            return "未检测到版本冲突"
        return f"检测到 {len(state.conflicts)} 组版本冲突：" + "；".join(
            c[:90] for c in state.conflicts[:2]
        )

    def _pack_evidence(self, state: AgentState, args: dict[str, Any]) -> str:
        n = int(args.get("top_n", self.s.final_evidence))
        chosen = state.candidates[:n]
        state.evidences = [
            Evidence(
                label=f"E{i + 1}",
                chunk=c.chunk,
                score=c.final_score,
                source="rerank_api" if c.rerank_score is not None else "rule_rerank",
            )
            for i, c in enumerate(chosen)
        ]
        if not state.evidences:
            return "证据打包失败：候选为空"
        return f"打包 {len(state.evidences)} 条证据：" + "、".join(
            f"{e.label} {e.chunk.citation[-24:]}" for e in state.evidences
        )

    def _answer(self, state: AgentState, args: dict[str, Any]) -> str:
        if not state.evidences:
            return "生成失败：尚无证据，请先 pack_evidence 或 refuse"
        if not self.s.llm_ready:
            return "生成失败：LLM 未配置，仅能展示检索证据（见 degrade_reasons）"

        # 红线「证据不足即拒答」在工具层强制执行，不交给决策器裁决
        rel = assess_relevance(
            state.effective_query,
            state.evidences,
            idf=self.p.bm25.idf_map,
            min_score=self.s.min_relevance_score,
            semantic_score=max((e.score for e in state.evidences), default=0.0),
            min_semantic=self.s.min_semantic_score,
        )
        state.relevance = rel.to_dict()
        if not rel.ok:
            # 红线正常生效，不算能力降级；单独留痕，避免 degraded_rate 随拒答率虚高
            state.redline_refusals.append(rel.reason)
            state.answer = REFUSAL
            state.refused = True
            state.stop_reason = StopReason.REFUSED
            return f"红线拒答：{rel.reason}"

        # 生成与核查必须看同一句问题：本轮用户原话。
        # 早期把 state.effective_query（多轮检索用的「上一轮＋本轮」拼接词）传给核查器，
        # 导致核查器要求答案覆盖历史问题，M01/M02/M04/M05 被误判「回答不完整」。
        question = state.answer_query
        history = self.p.memory.history_text(state.session_id)
        user_prompt = build_answer_user(question, state.evidences, state.conflicts, history)
        max_attempts = self.s.max_generation_retry + 1
        draft = ""

        for attempt in range(max_attempts):
            try:
                state.api_calls += 1
                draft = self.p.llm.chat(system=ANSWER_SYSTEM, user=user_prompt)
            except AdapterError as exc:
                _degrade(state, f"LLM 生成失败：{exc}")
                return f"生成失败：{exc}"
            state.draft = draft

            if "no_verify" in state.ablations:
                state.verification = {
                    "ok": True, "issues": [], "warnings": [],
                    "method": "ablated", "citations": [], "refusal": False,
                }
                break
            try:
                state.api_calls += 1
                state.verification = verify_answer(
                    draft, state.evidences, self.p.llm if self.s.llm_ready else None,
                    query=question,
                )
            except Exception as exc:  # noqa: BLE001 - 校验异常也要让决策器看到
                state.verification = {
                    "ok": False, "issues": [f"校验异常：{exc}"], "warnings": [],
                    "method": "error", "citations": [], "refusal": False,
                }
            if state.verification["ok"]:
                break
            if attempt < max_attempts - 1:
                # 确定性自我修正：不重新检索、不占用额外步数，只把核查意见回填生成提示重答一次。
                # 旧实现会丢弃草稿、让决策器再走一轮，直到 6 步耗尽被强制收敛为拒答。
                state.answer_retries += 1
                user_prompt += (
                    "\n\n【上次答案未通过核查，请只依据上面的证据修正后重答】\n"
                    + "；".join(state.verification["issues"])
                )
        else:
            # 连续核查不通过：按准确性红线拒答。预期红线行为，不算能力降级。
            # 未通过核查的草稿只留在 state.draft，绝不写进 state.answer / trace 导出。
            reason = "生成后引用校验连续不通过：" + "；".join(
                state.verification.get("issues", [])[:2]
            )
            state.redline_refusals.append(reason)
            state.answer = REFUSAL
            state.refused = True
            state.stop_reason = StopReason.REFUSED
            return f"{reason}；按红线拒答"

        state.answer = draft
        state.refused = False
        state.stop_reason = StopReason.VERIFIED
        suffix = f"（含 {state.answer_retries} 次带核查意见重生成）" if state.answer_retries else ""
        return f"生成并通过引用校验（{state.verification['method']}）：{len(draft)} 字{suffix}"

    def _refuse(self, state: AgentState, args: dict[str, Any]) -> str:
        reason = str(args.get("reason") or "证据不足")
        state.redline_refusals.append(f"决策器裁决：{reason}")
        state.answer = REFUSAL
        state.refused = True
        state.stop_reason = StopReason.REFUSED
        return f"按红线拒答：{reason}"
