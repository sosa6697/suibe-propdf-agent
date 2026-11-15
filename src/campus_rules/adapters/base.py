"""OpenAI 兼容端点的小工具：URL 拼接与鉴权头。"""
from __future__ import annotations


def endpoint(base_url: str, suffix: str) -> str:
    """拼接端点。suffix 为空表示 base_url 已是完整端点（常见于服务商原生 API）。"""
    base = base_url.rstrip("/")
    if not suffix or base.endswith(suffix):
        return base
    return base + suffix


def auth_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
