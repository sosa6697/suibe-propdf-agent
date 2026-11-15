"""查询侧 Embedding：支持两种协议，按 EMBEDDING_STYLE 切换。

- openai    : OpenAI 兼容端点，POST {base}/embeddings，
              请求 {"model","input":[text]}，响应 data[0].embedding。
- dashscope : DashScope 原生端点（服务商文档 dashscope.TextEmbedding.call 的用法），
              base_http_api_url = ".../api/v1"，SDK 内部拼
              /services/embeddings/text-embedding/text-embedding，
              请求 {"model","input":{"texts":[text]}}，响应 output.embeddings[0].embedding。
              注意该模式下忽略 BASE_EMBEDDING_API_PATH，避免重复拼接。

已知风险：
1. 建库时 embedding_input = 「标题 > 第N页 + 正文」，查询侧无法复现该前缀；
2. 建库走的是 OpenAI 兼容 /embeddings，若查询改走 DashScope 原生，虽为同一模型，
   默认行为（如 text_type）可能有差异，语义空间一致性需实测验证。
"""
from __future__ import annotations

import httpx
import numpy as np

from ..config import Settings
from ..logging_conf import get_logger
from .base import auth_headers, endpoint
from .errors import AdapterError

log = get_logger("adapter.embedding")

DASHSCOPE_EMBEDDING_PATH = "/services/embeddings/text-embedding/text-embedding"


class EmbeddingClient:
    def __init__(self, settings: Settings):
        self.s = settings

    @property
    def use_dashscope(self) -> bool:
        return self.s.embedding_style == "dashscope"

    def dashscope_url(self) -> str:
        """base_http_api_url + DashScope 原生服务路径；若已是完整路径则不重复拼。"""
        base = (self.s.embedding_api_url or "").rstrip("/")
        return base if base.endswith(DASHSCOPE_EMBEDDING_PATH) else base + DASHSCOPE_EMBEDDING_PATH

    def embed(self, text: str) -> np.ndarray:
        if not self.s.embedding_ready:
            raise AdapterError("embedding", "缺少 BASE_EMBEDDING_API_KEY / URL / MODEL")
        try:
            vec = self._embed_dashscope(text) if self.use_dashscope else self._embed_openai(text)
        except AdapterError:
            raise
        except (httpx.HTTPError, KeyError, ValueError, TypeError, IndexError) as exc:
            raise AdapterError("embedding", f"{type(exc).__name__}: {exc}") from exc

        log.info("查询向量生成完成：style=%s dim=%d", self.s.embedding_style, vec.shape[0])
        return vec

    def _post(self, url: str, payload: dict) -> dict:
        resp = httpx.post(
            url,
            headers=auth_headers(self.s.embedding_api_key or ""),
            json=payload,
            timeout=30,
        )
        if resp.status_code >= 400:
            raise AdapterError("embedding", f"HTTP {resp.status_code} @ {url}: {resp.text[:200]}")
        return resp.json()

    def _embed_openai(self, text: str) -> np.ndarray:
        url = endpoint(self.s.embedding_api_url or "", self.s.embedding_path)
        body = self._post(url, {"model": self.s.embedding_model, "input": [text]})
        data = body.get("data") or []
        if not data:
            raise AdapterError("embedding", "OpenAI 兼容响应缺少 data 字段")
        item = sorted(data, key=lambda d: d.get("index", 0))[0]
        return np.asarray(item["embedding"], dtype=np.float32)

    def _embed_dashscope(self, text: str) -> np.ndarray:
        url = self.dashscope_url()
        body = self._post(url, {"model": self.s.embedding_model, "input": {"texts": [text]}})
        items = (body.get("output") or {}).get("embeddings") or []
        if not items:
            data = body.get("data") or []  # 少数网关仍返回 OpenAI 兼容结构
            if data:
                item = sorted(data, key=lambda d: d.get("index", 0))[0]
                return np.asarray(item["embedding"], dtype=np.float32)
            raise AdapterError("embedding", f"DashScope 响应缺少 output.embeddings：{str(body)[:200]}")
        item = sorted(items, key=lambda d: d.get("text_index", d.get("index", 0)))[0]
        return np.asarray(item["embedding"], dtype=np.float32)
