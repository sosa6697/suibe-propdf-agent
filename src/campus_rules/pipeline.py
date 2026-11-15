"""主编排：显式流水线，每一步可观测、可降级、可追溯。

完整链路：
  多轮记忆 → 查询改写 → [向量召回 + BM25 召回] → RRF 融合 → 邻块扩展
  → API rerank（未配置则规则重排）→ 阈值拒答 → 证据打包 → LLM 生成
  → 生成后引用校验（不通过重试一次，仍不通过则拒答）→ 写入短期记忆

原则：任何组件缺失都必须在 degraded / degrade_reasons / trace 中显式留痕，
绝不静默跳过，也绝不把「仅检索到证据」伪装成「已生成答案」。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .adapters.embedding import EmbeddingClient
from .adapters.errors import AdapterError
from .adapters.llm import LLMClient
from .adapters.rerank import RerankClient
from .config import Settings
from .index.bm25_store import BM25Store
from .index.hybrid import (
    apply_recency_bias,
    detect_version_conflicts,
    expand_neighbors,
    rrf_fuse,
    rule_rerank,
)
from .index.vector_store import VectorStore
from .logging_conf import get_logger, setup_logging
from .memory import SQLiteMemory
from .prompts import (
    ANSWER_SYSTEM,
    REWRITE_SYSTEM,
    build_answer_user,
    build_rewrite_user,
)
from .query_guard import GUARD_REFUSAL, guard_query
from .relevance import assess_relevance
from .schema import AnswerResult, Evidence, QueryRequest, ScoredChunk
from .verification import verify_answer

log = get_logger("pipeline")

REFUSAL = "现有校规文件中未找到该问题的明确依据，建议咨询学校相关职能部门。"


@dataclass(slots=True)
class RetrievalResult:
    candidates: list[ScoredChunk]
    conflicts: list[str]
    reasons: list[str]
    trace: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "degraded": bool(self.reasons),
            "degrade_reasons": self.reasons,
            "version_conflicts": self.conflicts,
            "candidates": [
                {
                    "rank": i + 1,
                    "chunk_id": c.chunk.chunk_id,
                    "citation": c.chunk.citation,
                    "final_score": round(c.final_score, 6),
                    "rerank_score": c.rerank_score,
                    "vector_score": round(c.vector_score, 6),
                    "bm25_score": round(c.bm25_score, 6),
                    "sources": sorted(c.sources),
                    "expanded": c.expanded,
                    "preview": c.chunk.content.strip()[:160],
                }
                for i, c in enumerate(self.candidates)
            ],
            "trace": self.trace,
        }


class QAPipeline:
    def __init__(self, settings: Settings, *, ablations: tuple[str, ...] | list[str] = ()):
        """ablations 仅供评测做消融对照；默认空集合时行为与 v0.1 完全一致。

        支持：no_vector / no_expand / no_rerank / no_verify。
        """
        self.s = settings
        self.ablations = set(ablations)
        setup_logging(settings.log_level)
        log.info("初始化流水线：%s", settings.safe_summary())

        self.store = VectorStore(settings.data_dir).load()
        self.bm25 = BM25Store(settings.var_dir).load(self.store.chunks)
        self.embedding = EmbeddingClient(settings)
        self.rerank = RerankClient(settings)
        self.llm = LLMClient(settings)
        self.memory = SQLiteMemory(settings.var_dir / "memory.db", settings.memory_turns)

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _step(trace: list[dict], name: str, status: str, detail: str, t0: float) -> None:
        trace.append(
            {
                "step": name,
                "status": status,
                "detail": detail,
                "ms": round((time.perf_counter() - t0) * 1000, 2),
            }
        )
        log.debug("%s | %s | %s", name, status, detail)

    def health(self) -> dict:
        return {
            "status": "ok",
            "chunks": len(self.store),
            "dim": self.store.dim,
            **self.s.safe_summary(),
            "embedding_ready": self.s.embedding_ready,
            "llm_ready": self.s.llm_ready,
            "rerank_ready": self.s.rerank_ready,
            "vector_recall_enabled": self.s.enable_vector_recall,
            "bm25_top_k": self.s.top_k_bm25,
            "degrade_reasons": self.s.degrade_reasons(),
        }

    # ------------------------------------------------------------- 查询改写
    def _rewrite(self, req: QueryRequest, reasons: list[str]) -> str:
        turns = self.memory.history(req.session_id)
        if not turns:
            return self.memory.history_text(req.session_id)

        history_text = self.memory.history_text(req.session_id)
        if self.s.llm_ready:
            try:
                rewritten = self.llm.chat(
                    system=REWRITE_SYSTEM,
                    user=build_rewrite_user(history_text, req.raw_query),
                    timeout=60,
                )
                req.rewritten_query = rewritten.strip().splitlines()[0][:300]
                return history_text
            except AdapterError as exc:
                reasons.append(f"查询改写失败，回退到规则补全：{exc}")
        # 规则降级：把上一轮用户问题并入当前问题，保留全部原始关键词
        prev_user = [t.content for t in turns if t.role == "user"]
        if prev_user:
            req.rewritten_query = f"{prev_user[-1]} {req.raw_query}"[:300]
        return history_text

    # --------------------------------------------------------------- 检索
    def retrieve(self, query: str, reasons: list[str] | None = None) -> RetrievalResult:
        # 独立调用时只带检索相关的降级原因；ask() 会传入包含 LLM 状态的完整列表
        reasons = (
            list(reasons)
            if reasons is not None
            else [r for r in self.s.degrade_reasons() if not r.startswith("LLM")]
        )
        trace: list[dict] = []
        rank_lists: list[list[ScoredChunk]] = []

        t = time.perf_counter()
        if "no_vector" in self.ablations:
            self._step(
                trace, "vector_recall", "disabled",
                "消融实验 no_vector：关闭向量召回", t,
            )
        elif not self.s.enable_vector_recall:
            self._step(
                trace, "vector_recall", "disabled",
                "ENABLE_VECTOR_RECALL=false，按配置关闭向量召回", t,
            )
        elif self.s.embedding_ready:
            try:
                vec = self.embedding.embed(query)
                hits = self.store.search(vec, self.s.top_k_vector)
                rank_lists.append(hits)
                self._step(trace, "vector_recall", "ok", f"命中 {len(hits)} 块", t)
            except (AdapterError, ValueError) as exc:
                reasons.append(f"embedding 调用失败，向量召回关闭：{exc}")
                self._step(trace, "vector_recall", "degraded", str(exc), t)
        else:
            self._step(trace, "vector_recall", "skipped", "未配置 embedding，仅走 BM25", t)

        t = time.perf_counter()
        bm25_hits = self.bm25.search(query, self.s.top_k_bm25, self.store.chunks)
        rank_lists.append(bm25_hits)
        self._step(trace, "bm25_recall", "ok", f"命中 {len(bm25_hits)} 块", t)

        t = time.perf_counter()
        fused = rrf_fuse(rank_lists, k=self.s.rrf_k)
        self._step(trace, "rrf_fusion", "ok", f"融合后 {len(fused)} 个候选", t)

        t = time.perf_counter()
        if "no_expand" in self.ablations:
            candidates = fused
            self._step(trace, "neighbor_expansion", "disabled", "消融实验 no_expand：跳过邻块扩展", t)
        else:
            candidates = expand_neighbors(self.store, fused, self.s.neighbor_window)
            self._step(
                trace,
                "neighbor_expansion",
                "ok",
                f"邻块扩展后 {len(candidates)} 个候选（新增 {len(candidates) - len(fused)}）",
                t,
            )

        t = time.perf_counter()
        # 先规则预筛：邻块扩展后可能有 200+ 候选，直接全送 API 会超请求体/条数上限
        pre_ranked = rule_rerank(query, candidates, max_per_doc=self.s.max_per_doc)
        self._step(trace, "rerank_prefilter", "ok", f"规则预筛后 {len(pre_ranked)} 个候选", t)

        t = time.perf_counter()
        if self.s.rerank_ready and "no_rerank" not in self.ablations:
            cap = max(self.s.rerank_top_n * 5, 30)
            to_rerank = pre_ranked[:cap]
            try:
                ranked = self.rerank.rerank(query, to_rerank)
                ranked = apply_recency_bias(ranked)
                self._step(
                    trace, "rerank", "ok",
                    f"API rerank：送入 {len(to_rerank)} 块，返回 {len(ranked)} 块；"
                    "已叠加汇编降权与时效纠偏", t,
                )
            except AdapterError as exc:
                reasons.append(f"rerank API 调用失败，降级为规则重排：{exc}")
                ranked = pre_ranked
                self._step(trace, "rerank", "degraded", str(exc), t)
        else:
            # 静态 degrade_reasons 已声明「rerank 未配置」，此处只留 trace，不重复记因
            ranked = pre_ranked
            detail = "消融实验 no_rerank：仅规则重排" if "no_rerank" in self.ablations else "规则重排"
            self._step(trace, "rerank", "degraded", detail, t)

        # 版本检测要看「候选池」而不是精排后的 top8，否则同名多版本很难同时出现
        conflicts = detect_version_conflicts(pre_ranked[:40])
        return RetrievalResult(ranked, conflicts, reasons, trace)

    # --------------------------------------------------------- 入口过滤
    def _guard_blocked_result(self, req: QueryRequest, guard) -> AnswerResult:
        """入口命中的短路结果：0 次模型调用、0 条证据、0 次降级、不写会话记忆。

        刻意不调用 self.memory.append：域外问题不应污染多轮上下文，也守住
        「本期不采集 / 不落盘用户提示词」的边界。trace 里只写规则名，不回显 query。
        """
        trace: list[dict] = []
        t = time.perf_counter()
        self._step(
            trace, "query_guard", "blocked",
            f"rules={','.join(guard.matched_rules) or '-'} | {guard.reason}", t,
        )
        return AnswerResult(
            answer=GUARD_REFUSAL,
            refused=True,
            evidences=[],
            session_id=req.session_id,
            degraded=False,  # 未使用任何能力，谈不上降级；也不是红线误伤
            degrade_reasons=[],
            trace=trace,
            verification={
                "ok": True, "method": "skipped", "issues": [], "warnings": [],
                "guard_blocked": True,
            },
            version_conflicts=[],
            relevance={},
            redline_refusals=[guard.reason],
            guard_blocked=True,
            guard=guard.to_dict(),
        )

    # --------------------------------------------------------------- 问答
    def ask(self, req: QueryRequest) -> AnswerResult:
        # v0.5 入口过滤：必须早于查询改写、向量召回、rerank 与任何 LLM 调用。
        # 命中即返回固定拒答，用户的域外提示词不会进入任何模型。
        guard = guard_query(req.raw_query)
        if guard.blocked:
            return self._guard_blocked_result(req, guard)

        reasons = list(self.s.degrade_reasons())
        redline_refusals: list[str] = []
        trace: list[dict] = []

        t = time.perf_counter()
        history_text = self._rewrite(req, reasons)
        self._step(
            trace,
            "query_rewrite",
            "ok" if req.rewritten_query else "skipped",
            f"effective_query={req.effective_query[:120]}",
            t,
        )

        ret = self.retrieve(req.effective_query, reasons)
        trace.extend(ret.trace)
        reasons = ret.reasons

        t = time.perf_counter()
        candidates = ret.candidates
        best = candidates[0].final_score if candidates else 0.0
        gate_failed = not candidates or (
            self.s.min_evidence_score > 0 and best < self.s.min_evidence_score
        )
        if gate_failed:
            redline_refusals.append(
                "无候选" if not candidates
                else f"最高证据分 {best:.4f} 低于阈值 {self.s.min_evidence_score}"
            )
        self._step(
            trace,
            "threshold_gate",
            "refused" if gate_failed else "ok",
            f"best_score={best:.6f} threshold={self.s.min_evidence_score}",
            t,
        )

        evidences = [
            Evidence(
                label=f"E{i + 1}",
                chunk=c.chunk,
                score=c.final_score,
                source="rerank_api" if c.rerank_score is not None else "rule_rerank",
            )
            for i, c in enumerate(candidates[: self.s.final_evidence])
        ]

        t = time.perf_counter()
        relevance = assess_relevance(
            req.effective_query,
            evidences,
            idf=self.bm25.idf_map,
            min_score=self.s.min_relevance_score,
            semantic_score=max((e.score for e in evidences), default=0.0),
            min_semantic=self.s.min_semantic_score,
        )
        relevance_failed = not relevance.ok
        self._step(
            trace, "relevance_gate", "refused" if relevance_failed else "ok",
            f"score={relevance.score:.4f} threshold={relevance.threshold} | {relevance.reason}", t,
        )
        if relevance_failed:
            # 红线正常生效：单独留痕，不计入 degraded（否则 degraded_rate 会随拒答率虚高）
            redline_refusals.append(relevance.reason)
        gate_failed = gate_failed or relevance_failed

        answer = REFUSAL
        refused = True
        verification: dict = {"ok": True, "method": "skipped", "issues": [], "warnings": []}

        if gate_failed:
            log.info("证据不足，触发拒答")
        elif not self.s.llm_ready:
            answer = self._evidence_only_answer(evidences, ret.conflicts)
            self._step(trace, "generate", "degraded", "仅返回证据，未生成答案", time.perf_counter())
        else:
            # 生成与核查统一看本轮用户原话：req.effective_query 是检索用的改写词，
            # 把它交给核查器会让「答案是否回答了问题」的判断范围偏离用户真正问的那一句。
            question = req.raw_query
            user_prompt = build_answer_user(
                question, evidences, ret.conflicts, history_text
            )
            for attempt in range(self.s.max_generation_retry + 1):
                t = time.perf_counter()
                try:
                    draft = self.llm.chat(system=ANSWER_SYSTEM, user=user_prompt)
                except AdapterError as exc:
                    reasons.append(f"LLM 生成失败：{exc}")
                    answer = f"【生成失败】{exc}。以下为检索证据：\n" + self._evidence_only_answer(
                        evidences, ret.conflicts
                    )
                    self._step(trace, "generate", "degraded", str(exc), t)
                    break

                self._step(trace, "generate", "ok", f"第 {attempt + 1} 次生成 {len(draft)} 字", t)
                t = time.perf_counter()
                if "no_verify" in self.ablations:
                    verification = {
                        "ok": True, "issues": [], "warnings": [],
                        "method": "ablated", "citations": [], "refusal": False,
                    }
                else:
                    verification = verify_answer(
                        draft, evidences, self.llm if self.s.llm_ready else None,
                        query=question,
                    )
                self._step(
                    trace,
                    "citation_verification",
                    "ok" if verification["ok"] else "retry_or_refuse",
                    f"method={verification['method']} issues={len(verification['issues'])}",
                    t,
                )
                if verification["ok"]:
                    answer = draft
                    refused = False
                    break
                if attempt < self.s.max_generation_retry:
                    user_prompt += (
                        "\n\n【上次答案未通过核查，请修正后重答】\n"
                        + "；".join(verification["issues"])
                    )
            else:
                reasons.append("生成后引用校验连续不通过，已降级为拒答")
                answer = REFUSAL
                refused = True

        # 短期记忆：写入用户原始输入与最终回复（不额外采集任何提示词数据）
        self.memory.append(req.session_id, "user", req.raw_query)
        self.memory.append(
            req.session_id,
            "assistant",
            answer,
            meta={
                "refused": refused,
                "degraded": bool(reasons),
                "redline_refusal": bool(redline_refusals),
            },
        )

        return AnswerResult(
            answer=answer,
            refused=refused,
            evidences=evidences,
            session_id=req.session_id,
            degraded=bool(reasons),
            degrade_reasons=reasons,
            trace=trace,
            verification=verification,
            version_conflicts=ret.conflicts,
            relevance=relevance.to_dict(),
            redline_refusals=redline_refusals,
        )

    @staticmethod
    def _evidence_only_answer(evidences: list[Evidence], conflicts: list[str]) -> str:
        lines = [
            "【未生成答案】以下仅为检索到的证据，不构成回答，请勿直接引用为结论"
            "（具体原因见 degrade_reasons）："
        ]
        for e in evidences:
            preview = " ".join(e.chunk.content.split())[:220]
            lines.append(f"{e.label} {e.chunk.citation}（score={e.score:.4f}）\n{preview}")
        if conflicts:
            lines.append("版本冲突提示：" + "；".join(conflicts))
        return "\n\n".join(lines)
