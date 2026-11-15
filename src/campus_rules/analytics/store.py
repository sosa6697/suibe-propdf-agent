"""结构化运行事件日志（数据中台的写入层）。

与短期记忆 `var/memory.db` **物理分离**：后者是问答上下文，前者是运营统计，
两者职责、生命周期、保留策略都不同，绝不混库。

隐私边界（用户 2026-11-15 确认「最小结构化事件」方案）
----------------------------------------------------
事件表**没有 query / content / prompt 字段**，API 的 record() 也不接受这类参数，
从接口形状上就杜绝把用户提示词原文写进来。允许落盘的只有：

- 时间戳、事件类型、来源标记、会话 id（不含文本）；
- 主题词典命中的**标签名**（由 topics.classify 在内存里提取，标签来自内置词表）；
- 入口过滤命中的**规则名**；
- 校规文档的**公开元数据**（doc_title / page_no / score / source）；
- 整数计数（api_calls 等）。

用户原始输入只在内存里参与关键词提取，提取完即丢弃，不进入任何 SQL 参数。
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KINDS: tuple[str, ...] = (
    "query_received",     # 一次真实问答请求 → 关键词输入率 / 拦截率的分母
    "guard_blocked",      # 入口过滤命中         → 拦截率的分子
    "evidence_used",      # 一条 E1..En 采纳证据 → 被索引最多的数据
    "answer_completed",   # 一次回答结束         → 拒答率 / 模型调用数
)

_DDL = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    dedup_key   TEXT UNIQUE,
    ts          TEXT NOT NULL,
    kind        TEXT NOT NULL,
    source      TEXT NOT NULL,
    session_id  TEXT,
    topic       TEXT,
    topics_json TEXT,
    rule        TEXT,
    doc_title   TEXT,
    page_no     INTEGER,
    score       REAL,
    n           INTEGER NOT NULL DEFAULT 1,
    meta_json   TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_kind   ON events(kind);
CREATE INDEX IF NOT EXISTS idx_events_ts     ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_source ON events(source);
CREATE INDEX IF NOT EXISTS idx_events_doc    ON events(doc_title);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AnalyticsStore:
    """事件写入 + 轻量查询。默认 `:memory:`，由调用方显式指定落盘路径。

    内存模式保持一个长连接（sqlite 每次 connect(':memory:') 都是新库），
    因此自检可以用内存库验证完整链路而不污染真实 `var/analytics.db`。
    """

    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        self._memory = self.db_path == ":memory:"
        self._conn: sqlite3.Connection | None = None
        if not self._memory:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._memory:
            if self._conn is None:
                self._conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
            return self._conn
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_DDL)

    def record(
        self,
        kind: str,
        *,
        source: str = "live",
        session_id: str | None = None,
        topic: str | None = None,
        topics: list[str] | tuple[str, ...] = (),
        rule: str | None = None,
        doc_title: str | None = None,
        page_no: int | None = None,
        score: float | None = None,
        n: int = 1,
        meta: dict[str, Any] | None = None,
        ts: str | None = None,
        dedup_key: str | None = None,
    ) -> None:
        """写一条事件。**没有 query/content 参数**，从签名上禁止原文落盘。"""
        if kind not in KINDS:
            raise ValueError(f"未知事件类型 {kind}；可用 {list(KINDS)}")
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO events"
                "(dedup_key, ts, kind, source, session_id, topic, topics_json, rule,"
                " doc_title, page_no, score, n, meta_json) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    dedup_key, ts or _now(), kind, source, session_id,
                    topic, json.dumps(list(topics), ensure_ascii=False) if topics else None,
                    rule, doc_title, page_no, score, int(n),
                    json.dumps(meta, ensure_ascii=False) if meta else None,
                ),
            )

    def rows(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def scalar(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
            return row[0] if row else None

    def count(self, kind: str, source_prefix: str | None = None) -> int:
        if source_prefix:
            return int(self.scalar(
                "SELECT COUNT(*) FROM events WHERE kind=? AND source LIKE ?",
                (kind, source_prefix + "%"),
            ) or 0)
        return int(self.scalar("SELECT COUNT(*) FROM events WHERE kind=?", (kind,)) or 0)

    def all_text(self) -> str:
        """把全部文本列拼起来，供自检做「提示词原文零泄漏」断言。"""
        cols = ("source", "session_id", "topic", "topics_json", "rule", "doc_title", "meta_json")
        parts: list[str] = []
        for row in self.rows("SELECT " + ",".join(cols) + " FROM events"):
            parts.extend(str(v) for v in row.values() if v is not None)
        return "\n".join(parts)
