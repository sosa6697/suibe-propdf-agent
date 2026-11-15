"""历史真实记录回放：让看板第一次打开就有**真数据**可看，且不伪造。

数据来源全部是项目里已经落盘的真实产物：
- `eval/results/*.json`（19 个真实评测批次）+ `eval/golden.jsonl`（25 题金标题库）
- `var/*result*.json` / `var/demo_chat_runs.json`（真实联调 / 界面联调记录）

诚实标注（重要，写进 README 与看板）
------------------------------------
1. `query_received` 来自评测结果里真实的 case 记录；问题文本由 `case_id` 关联
   `eval/golden.jsonl` 得到，**只在内存里做主题匹配，绝不写入事件表**。
2. `guard_blocked` 是**用当前 v0.5 `query_guard` 对历史金标题库做的离线复算**，
   不是当时真实发生的拦截（v0.2–v0.4 还没有入口过滤）。source 标为
   `replay:guard_recompute`，meta 里带 `recomputed=true`。
3. `evidence_used` 有两类：评测结果里 `title_hit=True` 时由 `expected_titles`
   **推导**（`replay:title_hit`）；真实联调记录里则直接取回传的 E1..En
   证据元数据（`replay:run:*`）。两类来源都写进 `sources` 字段可追溯。
4. 19 个评测批次跑的是同一套 25 题金标题库，所以主题分布会重复放大；
   看板顶部显著标注「历史回放」与来源分布，不把它包装成线上流量。

幂等：每条事件带稳定 `dedup_key`，重复回放不会重复计数。
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT
from ..query_guard import guard_query
from .store import AnalyticsStore
from .topics import classify

_CITATION = re.compile(r"【(.+?)｜第(\d+)页】")


def _file_ts(path: Path, generated_at: str | None = None) -> str:
    if generated_at:
        return generated_at
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")


def _golden(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "eval" / "golden.jsonl"
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            case = json.loads(line)
            out[case["id"]] = case
    return out


def _evidence_meta(ev: dict[str, Any]) -> tuple[str | None, int | None]:
    """兼容两种证据格式：旧联调（citation/chunk_id）与新接口（doc_title/page_no）。"""
    title, page = ev.get("doc_title"), ev.get("page_no")
    if not title and ev.get("citation"):
        m = _CITATION.search(str(ev["citation"]))
        if m:
            title, page = m.group(1), int(m.group(2))
    if not title and ev.get("chunk_id"):
        title = str(ev["chunk_id"]).split("-p")[0]
    return (str(title) if title else None, int(page) if page is not None else None)


def seed_from_history(store: AnalyticsStore, *, root: Path | None = None) -> dict[str, Any]:
    """把已有真实产物回放为结构化事件。返回可读的回放统计（不含任何 query 原文）。"""
    root = Path(root or PROJECT_ROOT)
    golden = _golden(root)
    stats = {"result_files": 0, "cases": 0, "run_files": 0,
             "run_evidence": 0, "recomputed_guard": 0, "title_hit_derived": 0}

    # ---- A. 评测结果 -------------------------------------------------------
    for path in sorted((root / "eval" / "results").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            cases = payload.get("cases") or []
        except (OSError, json.JSONDecodeError):
            continue
        if not cases:
            continue
        stats["result_files"] += 1
        stem = path.stem
        ts = _file_ts(path, payload.get("generated_at"))
        mode = str((payload.get("options") or {}).get("mode") or "unknown")

        for case in cases:
            cid = str(case.get("id") or "")
            g = golden.get(cid, {})
            query = g.get("query", "")
            labels = classify(query) if query else ["未归类"]
            key = f"replay:{stem}:{cid}"
            expect_refusal = bool(g.get("expect_refusal"))
            refusal_correct = bool(case.get("refusal_correct"))
            refused = expect_refusal if refusal_correct else (not expect_refusal)

            store.record(
                "query_received", source=f"replay:eval:{stem}", ts=ts,
                session_id=f"eval::{cid}", topic=labels[0], topics=labels,
                dedup_key=key + ":query",
                meta={"batch": stem, "category": case.get("category"), "rid": key},
            )
            store.record(
                "answer_completed", source=f"replay:eval:{stem}", ts=ts,
                session_id=f"eval::{cid}", n=int(case.get("api_calls") or 0),
                dedup_key=key + ":answer",
                meta={"refused": refused, "mode": mode,
                      "stop_reason": case.get("stop_reason")},
            )

            # 当前规则对历史金标题库的离线复算，明确标注不是当时真实拦截
            guard = guard_query(query) if query else None
            if guard and guard.blocked:
                stats["recomputed_guard"] += 1
                store.record(
                    "guard_blocked", source="replay:guard_recompute", ts=ts,
                    session_id=f"eval::{cid}", rule=guard.matched_rules[0],
                    dedup_key=key + ":guard",
                    meta={"recomputed": True, "rules": list(guard.matched_rules),
                          "rid": key,
                          "note": "当前 query_guard 对历史金标题库的离线复算，非当时真实拦截"},
                )

            titles = g.get("expected_titles") or []
            if case.get("title_hit") and titles:
                stats["title_hit_derived"] += 1
                store.record(
                    "evidence_used", source="replay:title_hit", ts=ts,
                    session_id=f"eval::{cid}", doc_title=str(titles[0]),
                    score=case.get("best_evidence_score"), dedup_key=key + ":ev",
                    meta={"derived_from": "title_hit", "batch": stem},
                )
            stats["cases"] += 1

    # ---- B. 真实联调 / 界面联调记录（带 E1..En 证据元数据） ----------------
    candidates = sorted((root / "var").glob("*result*.json")) + [
        root / "var" / "demo_chat_runs.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        runs = payload if isinstance(payload, list) else [payload]
        ts = _file_ts(path, payload.get("recorded_at") if isinstance(payload, dict) else None)
        stem = path.stem
        seen_run = False
        for i, run in enumerate(runs):
            if not isinstance(run, dict):
                continue
            for j, ev in enumerate(run.get("evidences") or []):
                title, page = _evidence_meta(ev)
                if not title:
                    continue
                store.record(
                    "evidence_used", source=f"replay:run:{stem}", ts=ts,
                    doc_title=title, page_no=page, score=ev.get("score"),
                    dedup_key=f"replay:run:{stem}:{i}:{j}",
                    meta={"evidence_label": ev.get("label"),
                          "rerank_source": ev.get("source")},
                )
                stats["run_evidence"] += 1
                seen_run = True
        if seen_run:
            stats["run_files"] += 1

    return stats


def ensure_seeded(store: AnalyticsStore, *, root: Path | None = None,
                  force: bool = False) -> dict[str, Any]:
    """幂等入口：已有回放事件就跳过；force=True 可强制重放（仍靠 dedup_key 去重）。"""
    already = int(store.scalar(
        "SELECT COUNT(*) FROM events WHERE source LIKE 'replay:%'"
    ) or 0)
    if already and not force:
        return {"skipped": True, "existing_replay_events": already}
    stats = seed_from_history(store, root=root)
    return {"skipped": False, **stats}
