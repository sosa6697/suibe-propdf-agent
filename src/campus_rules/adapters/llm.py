"""LLM 适配器：答案生成、多轮查询改写、生成后引用校验、Agent 决策共用同一 chat 接口。

推理模型兼容（2026-11-15 实测）：deepseek-v4.1-flash 会把思考过程写进
`message.reasoning_content`；若 max_tokens 预算被思考链吃光，则 `content` 为空、
`finish_reason='length'`。此时不能用 reasoning_content 冒充答案（那是思考不是结论），
而是把预算翻倍重试一次；仍为空则抛出带诊断信息的错误，由编排层显式降级。
"""
from __future__ import annotations

import httpx

from ..config import Settings
from ..logging_conf import get_logger
from .base import auth_headers, endpoint
from .errors import AdapterError

log = get_logger("adapter.llm")

_MAX_BUDGET = 8192


class LLMClient:
    def __init__(self, settings: Settings):
        self.s = settings

    def chat(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: int = 60,
    ) -> str:
        if not self.s.llm_ready:
            raise AdapterError("llm", "缺少 BASE_LLM_API_KEY / URL / MODEL")
        url = endpoint(self.s.llm_api_url or "", self.s.llm_path)
        budget = max_tokens or self.s.llm_max_tokens
        payload = {
            "model": self.s.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        diag = ""
        msg_keys: list[str] = []
        choice_keys: list[str] = []

        for attempt in range(2):
            try:
                resp = httpx.post(
                    url,
                    headers=auth_headers(self.s.llm_api_key or ""),
                    json={**payload, "max_tokens": budget},
                    timeout=timeout,
                )
                if resp.status_code >= 400:
                    raise AdapterError("llm", f"HTTP {resp.status_code}: {resp.text[:160]}")
                choices = resp.json().get("choices") or []
                if not choices:
                    raise AdapterError("llm", "响应缺少 choices")
                choice = choices[0]
                message = choice.get("message") or {}
                msg_keys = sorted(message.keys()) if isinstance(message, dict) else []
                choice_keys = sorted(choice.keys()) if isinstance(choice, dict) else []
                content = message.get("content", "")
                if isinstance(content, str) and content.strip():
                    return content.strip()

                finish = str(choice.get("finish_reason") or "")
                reasoning = message.get("reasoning_content") or ""
                rlen = len(reasoning) if isinstance(reasoning, str) else 0
                diag = (
                    f"content 为空；finish_reason={finish!r}；"
                    f"reasoning_content 长度={rlen}；max_tokens={budget}；"
                    f"message 字段={msg_keys}；choices[0] 字段={choice_keys}"
                )
                if finish == "length" and attempt == 0 and budget < _MAX_BUDGET:
                    budget = min(budget * 2, _MAX_BUDGET)
                    log.warning("推理模型思考耗尽 token 预算，自动扩容重试：%d", budget)
                    continue
                break
            except AdapterError:
                raise
            except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
                raise AdapterError("llm", f"{type(exc).__name__}: {exc}") from exc

        raise AdapterError("llm", f"响应 {diag or 'content 为空'}（若是推理模型，请调大 LLM_MAX_TOKENS）")
