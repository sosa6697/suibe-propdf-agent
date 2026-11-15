"""向量库读取：vectors.npy (mmap) + meta.jsonl，行号严格对齐。

关键约束：vectors.npy 第 i 行 <-> meta.jsonl 第 i 行 <-> chunks.jsonl 第 i 行。
加载时做硬校验，行数不一致立刻报错，绝不允许静默错位。
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import numpy as np

from ..logging_conf import get_logger
from ..schema import Chunk, ScoredChunk

log = get_logger("index.vector")

_DATE_PREFIX = re.compile(r"^(\d{4})-(\d{2})-(\d{2})_")


def _parse_doc_date(title: str) -> date | None:
    m = _DATE_PREFIX.match(title)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _parse_chunk_index(chunk_id: str) -> int:
    parts = chunk_id.rsplit("::", 2)
    if len(parts) >= 2 and parts[1].isdigit():
        return int(parts[1])
    return 0


class VectorStore:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.vectors: np.ndarray | None = None
        self.chunks: list[Chunk] = []
        self._by_id: dict[str, Chunk] = {}
        self._by_doc: dict[str, list[Chunk]] = {}

    @property
    def dim(self) -> int:
        if self.vectors is None:
            raise RuntimeError("向量库未加载")
        return int(self.vectors.shape[1])

    def __len__(self) -> int:
        return len(self.chunks)

    def load(self) -> "VectorStore":
        vec_path = self.data_dir / "vectors.npy"
        meta_path = self.data_dir / "meta.jsonl"
        for p in (vec_path, meta_path):
            if not p.exists():
                raise FileNotFoundError(f"缺少数据文件：{p}（检查 .env 的 DATA_DIR）")

        vectors = np.load(vec_path, mmap_mode="r")
        if vectors.ndim != 2:
            raise ValueError(f"vectors.npy 应为二维矩阵，实际 shape={vectors.shape}")

        chunks: list[Chunk] = []
        with meta_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                title = r.get("doc_title", "")
                chunks.append(
                    Chunk(
                        row=i,
                        chunk_id=r["chunk_id"],
                        content=r["content"],
                        doc_id=r["doc_id"],
                        file_id=r.get("file_id") or r["doc_id"],
                        page_no=int(r.get("page_no") or 0),
                        doc_title=title,
                        source_type=r.get("source_type", "pdf_page"),
                        chunk_index=_parse_chunk_index(r["chunk_id"]),
                        doc_date=_parse_doc_date(title),
                        is_compilation="管理文件汇编" in title,
                    )
                )

        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"行号错位：meta.jsonl {len(chunks)} 行，vectors.npy {vectors.shape[0]} 行，拒绝加载"
            )

        self.vectors = vectors
        self.chunks = chunks
        self._by_id = {c.chunk_id: c for c in chunks}
        self._by_doc = {}
        for c in chunks:
            self._by_doc.setdefault(c.doc_id, []).append(c)
        for group in self._by_doc.values():
            group.sort(key=lambda x: x.chunk_index)

        log.info("向量库加载完成：%d 块 / %d 维 / %d 个 doc_id", len(chunks), self.dim, len(self._by_doc))
        return self

    def search(self, query_vec: np.ndarray, top_k: int) -> list[ScoredChunk]:
        if self.vectors is None:
            raise RuntimeError("向量库未加载")
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"查询向量维度 {q.shape[0]} != 库维度 {self.dim}（embedding 模型是否与建库一致？）")
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm
        # 库内向量已归一化，点积即余弦相似度
        scores = np.asarray(self.vectors @ q, dtype=np.float32)
        k = min(top_k, scores.shape[0])
        if k <= 0:
            return []
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [
            ScoredChunk(
                chunk=self.chunks[int(i)],
                vector_score=float(scores[i]),
                sources={"vector"},
            )
            for i in idx
        ]

    def neighbors(self, chunk: Chunk, window: int = 1) -> list[Chunk]:
        """按 doc_id 内的 chunk_index 取相邻块，修复 300 字切块把条款拦腰截断的问题。"""
        if window <= 0:
            return []
        group = self._by_doc.get(chunk.doc_id, [])
        out: list[Chunk] = []
        for c in group:
            if c.chunk_id == chunk.chunk_id:
                continue
            if abs(c.chunk_index - chunk.chunk_index) <= window:
                out.append(c)
        return out
