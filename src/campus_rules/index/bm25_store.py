"""BM25 关键词索引：中文分词 + rank_bm25，索引缓存到 var/（可重建，已 gitignore）。

存在意义：建库 embedding 带「标题>页码」前缀而查询侧无法复现，
BM25 提供一条不依赖该前缀的独立召回通路，与向量召回做 RRF 融合。
"""
from __future__ import annotations

import hashlib
import pickle
import re
from pathlib import Path

from ..logging_conf import get_logger
from ..schema import Chunk, ScoredChunk

log = get_logger("index.bm25")

_PUNCT = re.compile(r"^[\s\W_]+$", re.UNICODE)


def tokenize(text: str) -> list[str]:
    import jieba

    return [t for t in jieba.lcut(text) if t.strip() and not _PUNCT.match(t)]


class BM25Store:
    def __init__(self, var_dir: Path):
        self.path = Path(var_dir) / "bm25.pkl"
        self._bm25 = None
        self._rows: list[int] = []

    @staticmethod
    def fingerprint(chunks: list[Chunk]) -> str:
        head = chunks[0].chunk_id if chunks else ""
        tail = chunks[-1].chunk_id if chunks else ""
        raw = f"{len(chunks)}::{head}::{tail}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def build(self, chunks: list[Chunk]) -> "BM25Store":
        from rank_bm25 import BM25Okapi

        log.info("构建 BM25 索引：%d 块（首次构建需中文分词，稍等）", len(chunks))
        corpus = [tokenize(c.content) for c in chunks]
        self._bm25 = BM25Okapi(corpus)
        self._rows = [c.row for c in chunks]
        payload = {
            "fingerprint": self.fingerprint(chunks),
            "bm25": self._bm25,
            "rows": self._rows,
        }
        with self.path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("BM25 索引已缓存：%s (%.1f MB)", self.path, self.path.stat().st_size / 1024 / 1024)
        return self

    def load(self, chunks: list[Chunk]) -> "BM25Store":
        if self.path.exists():
            try:
                with self.path.open("rb") as f:
                    payload = pickle.load(f)  # noqa: S301 - 本地自建缓存
                if payload.get("fingerprint") == self.fingerprint(chunks):
                    self._bm25 = payload["bm25"]
                    self._rows = payload["rows"]
                    log.info("BM25 索引命中缓存：%s", self.path)
                    return self
                log.warning("BM25 缓存指纹不匹配，重建索引")
            except Exception as exc:
                log.warning("BM25 缓存损坏（%s），重建索引", exc)
        return self.build(chunks)

    @property
    def idf_map(self) -> dict[str, float]:
        """语料 IDF，供「问题—证据相关性门槛」做稀有词加权（不额外调用任何 API）。"""
        return dict(self._bm25.idf) if self._bm25 is not None else {}

    def search(self, query: str, top_k: int, chunks: list[Chunk]) -> list[ScoredChunk]:
        if self._bm25 is None:
            raise RuntimeError("BM25 索引未构建，请先执行 build-index")
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        k = min(top_k, len(scores))
        if k <= 0:
            return []
        import numpy as np

        arr = np.asarray(scores, dtype=np.float32)
        idx = np.argpartition(-arr, k - 1)[:k]
        idx = idx[np.argsort(-arr[idx])]
        out = []
        for i in idx:
            if arr[i] <= 0:
                continue
            out.append(
                ScoredChunk(
                    chunk=chunks[int(self._rows[int(i)])],
                    bm25_score=float(arr[i]),
                    sources={"bm25"},
                )
            )
        return out
