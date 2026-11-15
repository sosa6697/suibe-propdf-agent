"""混合召回与规则重排：RRF 融合 → 邻块扩展 → 规则打分 → 时效/版本冲突处理。

规则重排是 rerank API 未配置时的显式降级路径（会在输出中标注 degraded），
同时也是 API rerank 之前的候选整理，保证任何情况下链路不断。
"""
from __future__ import annotations

import re
from collections import defaultdict

from ..schema import Chunk, ScoredChunk
from ..text_utils import articles_in, normalize_for_prompt
from .bm25_store import tokenize
from .vector_store import VectorStore

_DATE_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}_")
_ENUM_QUERY = re.compile(r"哪些|几种|种类|包括|列举|有哪")
_ENUM_ITEM = re.compile(r"[（(][一二三四五六七八九十]+[）)]")


def rrf_fuse(rank_lists: list[list[ScoredChunk]], k: int = 60) -> list[ScoredChunk]:
    """倒数排名融合：不依赖分数尺度，适合向量余弦与 BM25 直接混合。"""
    merged: dict[int, ScoredChunk] = {}
    for hits in rank_lists:
        for rank, hit in enumerate(hits, start=1):
            row = hit.chunk.row
            cur = merged.get(row)
            if cur is None:
                cur = ScoredChunk(
                    chunk=hit.chunk,
                    vector_score=hit.vector_score,
                    bm25_score=hit.bm25_score,
                    sources=set(hit.sources),
                )
                merged[row] = cur
            else:
                cur.vector_score = max(cur.vector_score, hit.vector_score)
                cur.bm25_score = max(cur.bm25_score, hit.bm25_score)
                cur.sources.update(hit.sources)
            cur.rrf_score += 1.0 / (k + rank)
    return sorted(merged.values(), key=lambda x: -x.rrf_score)


def expand_neighbors(
    store: VectorStore, candidates: list[ScoredChunk], window: int = 1
) -> list[ScoredChunk]:
    """把命中块的左右邻块补进候选，修复「第 X 条」被 300 字切块截断的问题。"""
    if window <= 0:
        return candidates
    existing = {c.chunk.row: c for c in candidates}
    for parent in list(candidates):
        for nb in store.neighbors(parent.chunk, window):
            if nb.row in existing:
                continue
            hit = ScoredChunk(
                chunk=nb,
                rrf_score=parent.rrf_score * 0.5,
                vector_score=parent.vector_score * 0.5,
                sources=set(parent.sources) | {"neighbor"},
                expanded=True,
            )
            existing[nb.row] = hit
    return sorted(existing.values(), key=lambda x: -x.rrf_score)


def _title_stem(title: str) -> str:
    return _DATE_PREFIX.sub("", title).strip()


_VERSION_SUFFIX = re.compile(r"[（(]\s*20\d{2}\s*[）)]\s*$")
_SCHOOL_PREFIX = re.compile(r"^上海对外经贸大学")


def _version_key(title: str) -> str:
    """版本冲突分组键：去日期前缀、去末尾（2021）/（2023）年份、去校名前缀。

    首轮评测 conflict_detect_rate=0.00，主要就是因为同名规章的标题里
    带不同括号年份，被当成完全不同的主题。
    """
    stem = _VERSION_SUFFIX.sub("", _title_stem(title))
    return _SCHOOL_PREFIX.sub("", stem).strip("《》 ").strip()


def _recency_weight(chunk: Chunk, newest_year: int) -> float:
    if chunk.doc_date is None:
        return 0.5
    delta = newest_year - chunk.doc_date.year
    return max(0.25, 1.0 - 0.08 * max(0, delta))


def rule_rerank(
    query: str,
    candidates: list[ScoredChunk],
    *,
    max_per_doc: int = 2,
) -> list[ScoredChunk]:
    """规则重排：RRF + 标题/正文关键词 + 枚举条款 + 条款号 + 时效，汇编降权、同页限额。

    限额按 doc_id（页）而非 file_id（整份文件），否则一份 9 页的规章只能进 2 块，
    会把真正含条款的相邻块整块裁掉。
    """
    if not candidates:
        return []
    q_tokens = set(tokenize(query))
    q_articles = articles_in(query)
    want_enum = bool(_ENUM_QUERY.search(query))
    newest_year = max(
        (c.chunk.doc_date.year for c in candidates if c.chunk.doc_date), default=2026
    )
    max_rrf = max(c.rrf_score for c in candidates) or 1.0

    for c in candidates:
        text = normalize_for_prompt(c.chunk.content)
        content_cov = (
            len(q_tokens & set(tokenize(text))) / len(q_tokens) if q_tokens else 0.0
        )
        # 标题命中是校规检索的强信号：问题往往就是规章名称的关键词组合
        title_tokens = set(tokenize(_title_stem(c.chunk.doc_title)))
        title_cov = len(q_tokens & title_tokens) / len(q_tokens) if q_tokens else 0.0
        article_hit = 1.0 if (q_articles and q_articles & articles_in(text)) else 0.0
        # 「有哪些种类 / 包括哪些」这类问法，答案通常是带（一）（二）的枚举条款。
        # 必须同时命中标题，否则正文里引用了规章名的无关文件会被误加分。
        enum_hit = (
            1.0
            if want_enum and len(_ENUM_ITEM.findall(text)) >= 2 and title_cov >= 0.3
            else 0.0
        )
        recency = _recency_weight(c.chunk, newest_year)
        score = (
            0.38 * (c.rrf_score / max_rrf)
            + 0.20 * content_cov
            + 0.16 * title_cov
            + 0.10 * article_hit
            + 0.08 * recency
            + 0.08 * enum_hit
        )
        if c.chunk.is_compilation:
            score *= 0.75  # 历年《管理文件汇编》降权，避免盖过最新单行规定
        if c.expanded:
            score *= 0.85
        c.final_score = score

    ranked = sorted(candidates, key=lambda x: -x.final_score)
    per_page: dict[str, int] = defaultdict(int)
    out: list[ScoredChunk] = []
    for c in ranked:
        if per_page[c.chunk.doc_id] >= max_per_doc:
            continue
        per_page[c.chunk.doc_id] += 1
        out.append(c)
    return out


def apply_recency_bias(candidates: list[ScoredChunk]) -> list[ScoredChunk]:
    """API rerank 之后的时效 / 汇编纠偏。

    精排模型只看 query 与正文的语义相关度，不知道「这是 2017 年的汇编」，
    因此会用它自己的 relevance_score 覆盖规则预筛的降权结果 —— 实测中
    《2017年管理文件汇编》第76页曾以 0.982 压过最新单行规定（0.974）排到 #1，
    违反了「历年汇编降权、版本冲突不擅自裁决」的准确性红线。

    这里在精排之后把版本信号补回来：汇编 ×0.75，再叠加温和的时效因子。
    只在 API rerank 成功时调用；规则重排已做过同样降权，避免双重惩罚。
    """
    if not candidates:
        return candidates
    newest_year = max(
        (c.chunk.doc_date.year for c in candidates if c.chunk.doc_date), default=2026
    )
    for c in candidates:
        base = c.rerank_score if c.rerank_score is not None else c.final_score
        weight = 0.75 if c.chunk.is_compilation else 1.0
        weight *= 0.85 + 0.15 * _recency_weight(c.chunk, newest_year)
        c.final_score = base * weight
    return sorted(candidates, key=lambda x: -x.final_score)


def detect_version_conflicts(candidates: list[ScoredChunk]) -> list[str]:
    """同一规章主题命中多个不同发布日期的文件时，提示并列展示，不擅自裁决。"""
    groups: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for c in candidates:
        stem = _version_key(c.chunk.doc_title)
        date_str = c.chunk.doc_date.isoformat() if c.chunk.doc_date else "未知日期"
        groups[stem].add((date_str, c.chunk.doc_title))
    return [
        f"{stem} 命中 {len(versions)} 个版本："
        + "、".join(sorted(d for d, _ in versions))
        for stem, versions in groups.items()
        if len({d for d, _ in versions}) > 1
    ]
