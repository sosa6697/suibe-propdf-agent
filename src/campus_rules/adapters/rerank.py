"""Rerank 适配器：独立端点，url / apikey / model 由 .env 配置，后续自行填写。

支持两种常见请求风格（RERANK_STYLE）：
  cohere    : {"model","query","documents","top_n"}
  dashscope : {"model","input":{"query","documents"},"parameters":{"top_n"}}
响应里包含 results 列表（可嵌套在 output 下），每项带 index 与 relevance_score/score。

当前中转端点 /rerank 返回 404，故默认未配置；未配置时编排层显式降级为规则重排。
"""
from __future__ import annotations

from typing import Any

import httpx

from ..config import Settings
from ..logging_conf import get_logger
from ..schema import ScoredChunk
from ..text_utils import normalize_for_prompt
from .base import auth_headers, endpoint
from .errors import AdapterError

log = get_logger("adapter.rerank")


def _find_results(obj: Any) -> list[dict[str, Any]] | None:
    if isinstance(obj, dict):
        if isinstance(obj.get("results"), list):
            return obj["results"]
        for value in obj.values():
            found = _find_results(value)
            if found is not None:
                return found
    return None


class RerankClient:
    def __init__(self, settings: Settings):
        self.s = settings

    def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int | None = None) -> list[ScoredChunk]:
        if not self.s.rerank_ready:
            raise AdapterError("rerank", "缺少 RERANK_API_URL / API_KEY / MODEL")
        if not candidates:
            return []

        n = top_n or self.s.rerank_top_n
        documents = [normalize_for_prompt(c.chunk.content)[:1500] for c in candidates]
        if self.s.rerank_style == "dashscope":
            payload: dict[str, Any] = {
                "model": self.s.rerank_model,
                "input": {"query": query, "documents": documents},
                "parameters": {"return_documents": False, "top_n": min(n, len(documents))},
            }
        else:
            payload = {
                "model": self.s.rerank_model,
                "query": query,
                "documents": documents,
                "top_n": min(n, len(documents)),
            }

        url = endpoint(self.s.rerank_api_url or "", self.s.rerank_path)
        try:
            resp = httpx.post(
                url,
                headers=auth_headers(self.s.rerank_api_key or ""),
                json=payload,
                timeout=self.s.rerank_timeout,
            )
            if resp.status_code >= 400:
                raise AdapterError("rerank", f"HTTP {resp.status_code}: {resp.text[:160]}")
            results = _find_results(resp.json())
            if results is None:
                raise AdapterError("rerank", "响应中未找到 results 列表，检查 RERANK_STYLE 或返回格式")
        except AdapterError:
            raise
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise AdapterError("rerank", f"{type(exc).__name__}: {exc}") from exc

        for item in results:
            try:
                idx = int(item["index"])
                if not (0 <= idx < len(candidates)):
                    continue
                score = float(item.get("relevance_score", item.get("score", 0.0)))
            except (KeyError, TypeError, ValueError):
                continue
            hit = candidates[idx]
            hit.rerank_score = score
            hit.final_score = score
            hit.sources.add("rerank_api")

        ranked = sorted(candidates, key=lambda c: -(c.rerank_score if c.rerank_score is not None else -1e9))
        log.info("API rerank 完成：%d 候选 -> top %d", len(candidates), min(n, len(ranked)))
        return ranked[:n]
