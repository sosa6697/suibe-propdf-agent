"""聚合层：把结构化事件算成管理者看板上的三类指标。

三个口径（用户 2026-11-15 确认）：
1. 提示词关键词输入率 = 某主题命中的 query_received 数 / query_received 总数
   （允许多主题命中，各主题占比之和可 > 100%）
2. 拦截率 = 被拦截的**请求数** / query_received 总数（入口过滤层，检索前）
   一次请求可能命中多条软规则 → 会写多条 guard_blocked 事件，用 meta.rid 归并回
   同一个请求，避免「规则命中次数」把比率算到 100% 以上；按规则拆分仍保留全部命中。
   另给「检索后拒答率」作补充，两者口径不同，看板分开显示。
3. 被索引最多的数据 = 按 doc_title 统计 evidence_used（最终采纳的 E1..En）次数

所有指标都从 `AnalyticsStore` 的只读查询算出，**不读任何用户提示词原文**。
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from .store import AnalyticsStore
from .topics import TOPICS, catalog

_SCHEMA = "campus-rules-agent/analytics-v1"
_DATE_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}_")
_SCHOOL_PREFIX = re.compile(r"^上海对外经贸大学")


def clean_title(title: str) -> str:
    return _SCHOOL_PREFIX.sub("", _DATE_PREFIX.sub("", title or "")).strip() or (title or "（未知文档）")


def _rate(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def build_report(store: AnalyticsStore, *, top_docs: int = 10) -> dict[str, Any]:
    total_queries = store.count("query_received")
    guard_rule_hits = store.count("guard_blocked")  # 规则命中次数（可 > 被拦请求数）
    evidence_events = store.count("evidence_used")

    # ---- 主题关键词输入率：只取内置标签，原文不在事件表里 ----
    topic_counts = {name: 0 for name in TOPICS}
    topic_counts["未归类"] = 0
    for row in store.rows(
        "SELECT topics_json FROM events WHERE kind='query_received'"
    ):
        labels = json.loads(row["topics_json"]) if row["topics_json"] else ["未归类"]
        for label in labels:
            topic_counts[label] = topic_counts.get(label, 0) + 1
    order = {name: i for i, name in enumerate(topic_counts)}
    topics = [
        {"topic": name, "count": cnt, "rate": _rate(cnt, total_queries)}
        for name, cnt in topic_counts.items()
    ]
    topics.sort(key=lambda x: (-x["count"], order[x["topic"]]))

    # ---- 拦截率：分子是「被拦请求数」，按规则拆分仍保留全部命中 ----
    guard_rows = store.rows(
        "SELECT rule, source, session_id, ts, meta_json FROM events "
        "WHERE kind='guard_blocked'"
    )
    rids: set[str] = set()
    rule_counts: dict[str, int] = {}
    day_rids: dict[str, set[str]] = {}
    for r in guard_rows:
        meta = json.loads(r["meta_json"]) if r["meta_json"] else {}
        # 兼容没有 rid 的早期回放事件：用来源 + 会话 + 时间戳归并
        rid = str(meta.get("rid") or f"{r['source']}|{r['session_id']}|{r['ts']}")
        rids.add(rid)
        name = r["rule"] or "-"
        rule_counts[name] = rule_counts.get(name, 0) + 1
        day_rids.setdefault((r["ts"] or "")[:10], set()).add(rid)
    blocked = len(rids)
    by_rule = [
        {"rule": k, "count": v}
        for k, v in sorted(rule_counts.items(), key=lambda x: (-x[1], x[0]))
    ]
    trend_rows = store.rows(
        "SELECT substr(ts,1,10) day, COUNT(*) queries FROM events "
        "WHERE kind='query_received' GROUP BY day ORDER BY day"
    )
    trend = [
        {"day": r["day"], "queries": int(r["queries"]),
         "blocked": len(day_rids.get(r["day"], ())),
         "rate": _rate(len(day_rids.get(r["day"], ())), int(r["queries"]))}
        for r in trend_rows
    ]

    # ---- 检索后拒答率（补充指标，与入口拦截分开） ----
    completed = store.rows(
        "SELECT meta_json, n FROM events WHERE kind='answer_completed'"
    )
    refused = 0
    api_calls = 0
    for r in completed:
        api_calls += int(r["n"] or 0)
        meta = json.loads(r["meta_json"]) if r["meta_json"] else {}
        if meta.get("refused"):
            refused += 1

    # ---- 被索引最多的数据：最终采纳证据 ----
    # 按「清洗后的标题」归并：同一份规章在源数据里可能带 / 不带日期前缀与校名前缀，
    # 直接按原始 doc_title 分组会把一份文档拆成多行，排行失真。
    buckets: dict[str, dict[str, Any]] = {}
    for r in store.rows(
        "SELECT doc_title, page_no, score, source FROM events "
        "WHERE kind='evidence_used' AND doc_title IS NOT NULL"
    ):
        key = clean_title(r["doc_title"] or "")
        b = buckets.setdefault(key, {
            "count": 0, "pages": set(), "scores": [], "sources": set(), "raw": set(),
        })
        b["count"] += 1
        if r["page_no"] is not None:
            b["pages"].add(int(r["page_no"]))
        if r["score"] is not None:
            b["scores"].append(float(r["score"]))
        if r["source"]:
            b["sources"].add(r["source"])
        b["raw"].add(r["doc_title"])
    docs = [
        {
            "doc_title": key,
            "display_title": key,
            "count": b["count"],
            "pages": len(b["pages"]),
            "avg_score": round(sum(b["scores"]) / len(b["scores"]), 4) if b["scores"] else None,
            "sources": sorted(b["sources"]),
            "raw_titles": sorted(b["raw"]),
            "rate": _rate(b["count"], evidence_events),
        }
        for key, b in buckets.items()
    ]
    docs.sort(key=lambda d: (-d["count"], d["display_title"]))
    docs = docs[:top_docs]
    distinct_docs = len(buckets)

    sources = [
        {"source": r["source"], "count": int(r["n"])}
        for r in store.rows(
            "SELECT source, COUNT(*) n FROM events GROUP BY source ORDER BY n DESC, source"
        )
    ]
    replay = sum(s["count"] for s in sources if s["source"].startswith("replay:"))
    live = sum(s["count"] for s in sources if s["source"] == "live")

    return {
        "schema": _SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overview": {
            "total_queries": total_queries,
            "guard_blocked": blocked,           # 被拦截的请求数（拦截率分子）
            "guard_rule_hits": guard_rule_hits,  # 规则命中次数（可 > 被拦请求数）
            "guard_rate": _rate(blocked, total_queries),
            "refused_after_retrieval": refused,
            "refusal_rate": _rate(refused, total_queries),
            "evidence_events": evidence_events,
            "distinct_docs": distinct_docs,
            "total_api_calls": api_calls,
            "event_rows": int(store.scalar("SELECT COUNT(*) FROM events") or 0),
            "replay_events": replay,
            "live_events": live,
        },
        "topics": {
            "dictionary": catalog(),
            "items": topics,
            "multi_label": True,
            "note": "一条输入可命中多个主题，占比之和可能超过 100%；分母为 query_received 总数",
        },
        "guard": {
            "rate": _rate(blocked, total_queries),
            "blocked": blocked,
            "rule_hits": guard_rule_hits,
            "by_rule": by_rule,
            "trend": trend,
            "note": "仅统计 query_guard 入口拦截（检索前，0 次模型调用）；分子为被拦请求数，"
                    "一次请求命中多条规则时按规则拆分之和会大于该数；检索后红线拒答单列",
        },
        "documents": {
            "items": docs,
            "top_n": top_docs,
            "note": "统计最终采纳的 E1..En 证据；doc_title 为校规公开元数据，不含用户提示词",
        },
        "sources": sources,
        "privacy": {
            "stores_raw_prompt": False,
            "fields": ["ts", "kind", "source", "session_id", "topic", "rule",
                       "doc_title", "page_no", "score", "n"],
            "note": "用户提示词原文只在内存里参与主题词典匹配，不进入 analytics.db",
        },
    }


def render_text(rep: dict[str, Any]) -> str:
    """CLI 用的人类可读报表；内容与看板 / JSON 端点同源。"""
    o = rep["overview"]
    pct = lambda v: "N/A" if v is None else f"{v * 100:.2f}%"

    lines = [
        "=" * 68,
        "校规问答 Agent · 数据中台",
        f"生成时间 {rep['generated_at']}  schema={rep['schema']}",
        "-" * 68,
        f"问答请求总数        {o['total_queries']}",
        f"入口拦截 / 拦截率   {o['guard_blocked']} / {pct(o['guard_rate'])}",
        f"检索后拒答 / 拒答率 {o['refused_after_retrieval']} / {pct(o['refusal_rate'])}",
        f"采纳证据 / 涉及文档 {o['evidence_events']} 条 / {o['distinct_docs']} 份",
        f"累计模型调用        {o['total_api_calls']} 次"
        "（主要来自历史回放；/ask live 不携带调用计数）",
        f"事件行数            {o['event_rows']}"
        f"（历史回放 {o['replay_events']} / 真实运行 {o['live_events']}）",
        "",
        "【提示词关键词输入率】主题词典命中 / 请求总数（可多主题命中）",
    ]
    for item in rep["topics"]["items"]:
        if item["count"]:
            lines.append(f"  {item['topic']:<10} {item['count']:>5}  {pct(item['rate'])}")

    lines += [
        "", "【拦截率】分子为被拦请求数；下面按规则拆分的是规则命中次数",
        f"  规则命中合计 {o['guard_rule_hits']}（一次请求可命中多条规则）",
    ]
    if rep["guard"]["by_rule"]:
        for row in rep["guard"]["by_rule"]:
            lines.append(f"  {row['rule']:<24} {row['count']:>5}")
    else:
        lines.append("  （暂无拦截事件）")

    lines += ["", f"【被索引最多的数据】Top {rep['documents']['top_n']}（最终采纳证据）"]
    for i, d in enumerate(rep["documents"]["items"], 1):
        score = "N/A" if d["avg_score"] is None else f"{d['avg_score']:.4f}"
        lines.append(
            f"  {i:>2}. {d['display_title'][:34]:<34} {d['count']:>4} 次 "
            f"{d['pages']:>2} 页 均分={score}"
        )

    lines += [
        "",
        "数据源分布：" + "、".join(
            f"{s['source']}={s['count']}" for s in rep["sources"][:6]
        ),
        "隐私：analytics.db 不存用户提示词原文；看板含历史回放数据，非线上实时流量。",
        "=" * 68,
    ]
    return "\n".join(lines)
