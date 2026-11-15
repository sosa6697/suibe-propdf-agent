"""短期记忆：多轮会话 + SQLite 持久化，按 session_id 隔离，进程重启可恢复。

表结构说明：turns 完整保存每轮的用户原始问题与助手回答（不截断、不覆盖）。
后续需求「采集用户输入的提示词数据」可基于此表扩展或另建独立表 —— 本期
只做问答记忆本身，不额外采集、不写埋点、不新增采集字段。
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .logging_conf import get_logger

log = get_logger("memory")


@dataclass(frozen=True, slots=True)
class Turn:
    role: str
    content: str
    created_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SQLiteMemory:
    def __init__(self, db_path: Path, max_turns: int = 6):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_turns = max(1, max_turns)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turns (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    meta_json   TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_turns_session
                    ON turns(session_id, id);
                """
            )

    def append(self, session_id: str, role: str, content: str, meta: dict | None = None) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions(session_id, created_at, updated_at) "
                "VALUES(?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at",
                (session_id, now, now),
            )
            conn.execute(
                "INSERT INTO turns(session_id, role, content, created_at, meta_json) VALUES(?, ?, ?, ?, ?)",
                (session_id, role, content, now, json.dumps(meta, ensure_ascii=False) if meta else None),
            )
        log.debug("记忆写入：session=%s role=%s len=%d", session_id, role, len(content))

    def history(self, session_id: str) -> list[Turn]:
        limit = self.max_turns * 2
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, created_at FROM turns "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [Turn(r["role"], r["content"], r["created_at"]) for r in reversed(rows)]

    def history_text(self, session_id: str) -> str:
        turns = self.history(session_id)
        if not turns:
            return "（无历史对话）"
        return "\n".join(f"{'用户' if t.role == 'user' else '助手'}：{t.content}" for t in turns)

    def clear(self, session_id: str) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            return cur.rowcount
