"""问题—证据相关性门槛：生成前的硬防线，纯本地计算、零 API 调用。

背景（首轮评测实测）：语料外问题（食堂营业时间、保研率、地铁出口…）会被 LLM
拿不相关的校规证据硬答，而「引用校验」只查引用编号 / 数字 / 条款号，
查不出「答非所问」，因此 citation_pass_rate=1.00 但拒答类 5/5 全部漏拒。

v0.3 修正（6 组全量评测暴露的问题）：
早期实现是「词面 ∩ 语义」双否决（两者都要达标），其中语义项是
`max(E.score) >= 0.80` 的**绝对阈值**。全量复测证明这个设计有两处硬伤：

1. 语义分不可分：该拒的域外题 R04=0.7988，反而高于不该拒的 M03=0.7836 /
   V04=0.7791；换任何切割点都无法同时满足。
2. `E.score` 会随链路配置换尺度：关掉 API rerank 后规则重排分只有 0.50–0.74
   （A1 组 25 题全被误拒），Agent 自主改写 query 后掉到 0.56–0.76（A5 误拒 4 题）。
   一个拿特定服务分数标定的常数，不能跨链路复用。

现在改成「词面主判 + 语义补位」：

- **词面主判**：IDF 加权覆盖率。它与任何重排分数尺度无关，是唯一稳定的信号。
  实测域外题落在 0.05–0.27，正常题多数 ≥0.32，阈值 0.30 有清晰间隔。
  高频虚词（学校 / 学生）权重低；语料外词（食堂 / 供餐 / 地铁 / 哈佛）
  BM25 idf 为 0 或缺失，给最高权重，一旦证据里找不到，覆盖率明显偏低。
- **语义补位（不再是否决）**：只有当词面**不达标**时，才允许用高分把它救回来，
  专门照顾「那申诉期限呢？」这类省略式追问（词面天然低但语义确实相关）。
  且**仅在分数确实来自 API rerank 时可用** —— 规则重排的分数尺度与阈值标定
  不一致，此时语义项自动关闭，结果里显式标注 `semantic_source` 便于追溯。
- 第二道防线仍在已有的 LLM 引用核查调用里（「证据能否回答该问题」检查项），
  不增加任何调用次数。

`min_score<=0` 关闭整个门槛；`min_semantic<=0` 只关闭语义补位（便于消融）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .index.bm25_store import tokenize
from .schema import Evidence
from .text_utils import normalize_for_prompt

_DATE_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}_")

# 语义补位只认这个来源：0.80 是按 API rerank 分数标定的
_SEMANTIC_TRUSTED_SOURCE = "rerank_api"


@dataclass(frozen=True, slots=True)
class RelevanceResult:
    ok: bool
    score: float
    title_cov: float
    body_cov: float
    threshold: float
    reason: str
    semantic_score: float = 0.0
    semantic_threshold: float = 0.0
    semantic_available: bool = False
    semantic_source: str = "disabled"

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "score": round(self.score, 4),
            "title_cov": round(self.title_cov, 4),
            "body_cov": round(self.body_cov, 4),
            "threshold": self.threshold,
            "semantic_score": round(self.semantic_score, 6),
            "semantic_threshold": self.semantic_threshold,
            "semantic_available": self.semantic_available,
            "semantic_source": self.semantic_source,
            "reason": self.reason,
        }


def _weight(token: str, idf: dict[str, float], oov_weight: float) -> float:
    value = idf.get(token)
    # BM25Okapi 对语料外词返回 0 / 负数；这些恰恰是最该抓的信号，给最高权重
    return oov_weight if (value is None or value <= 0) else value


def _coverage(query_tokens: set[str], evidence_tokens: set[str],
              idf: dict[str, float], oov_weight: float) -> float:
    total = sum(_weight(t, idf, oov_weight) for t in query_tokens) or 1.0
    hit = sum(_weight(t, idf, oov_weight) for t in query_tokens if t in evidence_tokens)
    return hit / total


def _semantic_source(evidences: list[Evidence], min_semantic: float) -> tuple[bool, str]:
    """语义补位是否可用，以及为什么。

    绝对阈值 0.80 是按 API rerank 的分数尺度标定的；规则重排分数整体低一档，
    拿同一个常数去卡会误伤全部正常题（6 组全量评测里 A1 的真实教训）。
    """
    if min_semantic <= 0:
        return False, "disabled"
    sources = {e.source for e in evidences}
    if _SEMANTIC_TRUSTED_SOURCE not in sources:
        return False, "rule_rerank_scale_mismatch"
    return True, _SEMANTIC_TRUSTED_SOURCE


def assess_relevance(
    query: str,
    evidences: list[Evidence],
    *,
    idf: dict[str, float] | None = None,
    min_score: float = 0.30,
    semantic_score: float = 0.0,
    min_semantic: float = 0.80,
) -> RelevanceResult:
    """返回是否允许把证据送进 LLM 生成：词面主判 + 语义补位。

    - 词面达标即放行（与分数尺度无关的稳定信号）；
    - 词面不达标时，只有「来源可信 且 分数 >= min_semantic」才救回；
    - 两者都不达标才按红线拒答。

    语义项只在证据来自 API rerank 时启用；规则重排时自动关闭并在结果中标注，
    避免拿一个尺度标定的常数去卡另一个尺度的分数。
    """
    if min_score <= 0:
        return RelevanceResult(True, 1.0, 1.0, 1.0, 0.0, "门槛关闭")
    if not evidences:
        return RelevanceResult(False, 0.0, 0.0, 0.0, min_score, "无证据")
    semantic_available, semantic_source = _semantic_source(evidences, min_semantic)

    idf = idf or {}
    oov_weight = (max(idf.values()) if idf else 8.0) + 1.0
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return RelevanceResult(True, 1.0, 1.0, 1.0, min_score, "查询无可分词内容")

    title_cov = body_cov = 0.0
    for e in evidences:
        title = _DATE_PREFIX.sub("", e.chunk.doc_title)
        title_cov = max(title_cov, _coverage(q_tokens, set(tokenize(title)), idf, oov_weight))
        body_cov = max(
            body_cov,
            _coverage(q_tokens, set(tokenize(normalize_for_prompt(e.chunk.content))),
                      idf, oov_weight),
        )
    score = 0.6 * title_cov + 0.4 * body_cov
    lexical_ok = score >= min_score
    semantic_ok = semantic_available and semantic_score >= min_semantic
    ok = lexical_ok or semantic_ok

    if ok and lexical_ok:
        reason = f"词面相关性达标（{score:.3f}≥{min_score:.2f}）"
    elif ok:
        reason = (
            f"词面 {score:.3f}<{min_score:.2f}，但语义精排分 {semantic_score:.3f}"
            f"≥{min_semantic:.2f}（来源 {semantic_source}），按省略式追问补位放行"
        )
    elif semantic_available:
        reason = (
            f"问题—证据词面相关性 {score:.3f} 低于阈值 {min_score:.2f}，"
            f"语义精排分 {semantic_score:.3f} 也低于 {min_semantic:.2f}，判定为域外/证据不足"
        )
    else:
        reason = (
            f"问题—证据词面相关性 {score:.3f} 低于阈值 {min_score:.2f}，"
            f"且语义补位不可用（{semantic_source}），判定为域外/证据不足"
        )
    return RelevanceResult(
        ok, score, title_cov, body_cov, min_score, reason,
        semantic_score, min_semantic, semantic_available, semantic_source,
    )
