"""全链路共享的数据结构。

注意：QueryRequest.raw_query 始终保存用户原始输入，不做覆盖。
后续需求「采集用户输入的提示词数据」将以此为入口 —— 本期只保留字段与空接口，
不实现、不注入、不调用任何采集逻辑。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from .text_utils import normalize_for_prompt


@dataclass(frozen=True, slots=True)
class Chunk:
    """meta.jsonl 的一行，与 vectors.npy 第 i 行严格对齐。"""

    row: int
    chunk_id: str
    content: str
    doc_id: str
    file_id: str
    page_no: int
    doc_title: str
    source_type: str
    chunk_index: int
    doc_date: date | None
    is_compilation: bool

    @property
    def citation(self) -> str:
        return f"【{self.doc_title}｜第{self.page_no}页】"


@dataclass(slots=True)
class ScoredChunk:
    chunk: Chunk
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float | None = None
    final_score: float = 0.0
    sources: set[str] = field(default_factory=set)
    expanded: bool = False

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


@dataclass(frozen=True, slots=True)
class Evidence:
    """送进 LLM 的一条证据，带可校验的引用编号。"""

    label: str
    chunk: Chunk
    score: float
    source: str

    @property
    def text(self) -> str:
        return f"{self.label} {self.chunk.citation}\n{self.chunk.content.strip()}"


def evidence_to_dict(e: Evidence, *, excerpt_len: int = 260) -> dict[str, Any]:
    """证据 →「答案依据」卡片。/ask 与 /agent 共用，保证两处字段完全一致。

    excerpt 是**校规原文片段**（不是用户提示词），只做展示清洗：
    去掉 PDF 抽取的页码噪声、压掉多余空白，Chunk.content 原文保持不变。
    这样界面能把「答案凭什么这么说」直接展开给用户核对。
    """
    excerpt = normalize_for_prompt(e.chunk.content)
    if len(excerpt) > excerpt_len:
        excerpt = excerpt[:excerpt_len].rstrip() + "…"
    return {
        "label": e.label,
        "citation": e.chunk.citation,
        "chunk_id": e.chunk.chunk_id,
        "score": round(e.score, 6),
        "source": e.source,
        "doc_title": e.chunk.doc_title,
        "page_no": e.chunk.page_no,
        "is_compilation": e.chunk.is_compilation,
        "excerpt": excerpt,
    }


@dataclass(slots=True)
class QueryRequest:
    """一次问答的输入。raw_query 是用户原始提示词，永不改写覆盖。"""

    raw_query: str
    session_id: str = "default"
    rewritten_query: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    # TODO(后续需求): 用户输入的提示词数据采集。
    # 预留入口：raw_query + session_id + 时间戳即可构成一条原始提示词记录。
    # 本期禁止在此处添加采集字段、埋点或落盘调用。

    @property
    def effective_query(self) -> str:
        return self.rewritten_query or self.raw_query


@dataclass(slots=True)
class AnswerResult:
    answer: str
    refused: bool
    evidences: list[Evidence]
    session_id: str
    degraded: bool
    degrade_reasons: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    version_conflicts: list[str] = field(default_factory=list)
    relevance: dict[str, Any] = field(default_factory=dict)
    # 「证据不足即拒答」是红线正常生效，不是能力降级；单独留痕，不污染 degraded
    redline_refusals: list[str] = field(default_factory=list)
    # v0.5 入口过滤：命中域外问题时，/ask 直接返回固定拒答，0 次模型调用。
    # guard 只含规则名与固定文案，不含用户提示词原文。
    guard_blocked: bool = False
    guard: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "refused": self.refused,
            "session_id": self.session_id,
            "degraded": self.degraded,
            "degrade_reasons": self.degrade_reasons,
            "redline_refusals": self.redline_refusals,
            "guard_blocked": self.guard_blocked,
            "guard": self.guard,
            "evidences": [evidence_to_dict(e) for e in self.evidences],
            "verification": self.verification,
            "version_conflicts": self.version_conflicts,
            "relevance": self.relevance,
            "trace": self.trace,
        }


class PromptSink(Protocol):
    """后续需求占位：用户输入的提示词数据采集接口。

    v0.1 仅定义协议，不提供实现、不注入流水线、不调用。
    后续要启用时，再实现该协议并在编排层显式接线。
    """

    def capture(self, request: QueryRequest) -> None: ...
