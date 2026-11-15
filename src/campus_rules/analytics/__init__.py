"""数据中台：结构化事件 → 聚合指标 → 管理者看板。

设计边界（用户 2026-11-15 确认）：
- 独立 `var/analytics.db`，与短期记忆 `var/memory.db` 物理分离；
- **只存结构化元数据，不存用户提示词原文**（最小结构化事件方案）；
- 关键词口径 = 预定义校规主题词典命中率；
- 「被索引最多的数据」口径 = 最终采纳的 E1..En 证据；
- 全程纯本地，零模型 API 调用；不做登录 / 权限 / 多用户隔离（项目定位）。
"""
from .report import build_report, clean_title, render_text
from .seed import ensure_seeded, seed_from_history
from .store import KINDS, AnalyticsStore
from .topics import NONE_TOPIC, TOPICS, catalog, classify
from .view import PLACEHOLDER, render_dashboard_html

__all__ = [
    "AnalyticsStore", "KINDS", "TOPICS", "NONE_TOPIC", "classify", "catalog",
    "build_report", "render_text", "clean_title",
    "ensure_seeded", "seed_from_history",
    "render_dashboard_html", "PLACEHOLDER",
]
