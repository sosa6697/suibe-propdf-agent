"""Agent 轨迹导出：JSON / 纯文本时间线 / 独立 HTML 报告。

导出默认脱敏：不写出用户原始提示词全文，只保留长度与 sha8 前缀。
include_raw_query 是后续「提示词数据采集」需求的显式开关，
本期所有 CLI / API / 评测调用方都不启用，也不做隐式落盘。
"""
from __future__ import annotations

import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .state import AgentState

_SCHEMA = "campus-rules-agent/trace-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def trace_dict(state: AgentState, *, include_raw_query: bool = False) -> dict[str, Any]:
    """结构化轨迹，供 API / 评测 / JSON 文件共用，保证三处字段一致。"""
    payload = state.to_dict(include_raw_query=include_raw_query)
    payload["schema"] = _SCHEMA
    payload["generated_at"] = _now()
    payload["redacted"] = not include_raw_query
    return payload


def render_text(state: AgentState) -> str:
    """CLI 用的人类可读时间线：thought → action → observation。"""
    lines = [
        f"Agent 轨迹 | mode={state.decision_mode} | "
        f"stop={state.stop_reason.value if state.stop_reason else '-'} | "
        f"steps={state.step} | api_calls={state.api_calls}",
        f"查询指纹：chars={len(state.raw_query)} "
        f"sha8={hashlib.sha256(state.raw_query.encode('utf-8')).hexdigest()[:8]} "
        f"（默认脱敏，不显示原文）",
        "-" * 68,
    ]
    for t in state.tool_calls:
        flag = "OK " if t.ok else "ERR"
        lines.append(f"[{t.step}] {flag} {t.action} ({t.ms} ms)")
        if t.thought:
            lines.append(f"     thought : {t.thought}")
        if t.args:
            lines.append(f"     args    : {json.dumps(t.args, ensure_ascii=False)[:200]}")
        lines.append(f"     observe : {t.observation}")
    lines.append("-" * 68)
    lines.append(f"答案：{state.answer}")
    if state.evidences:
        lines.append("证据：" + "、".join(
            f"{e.label} {e.chunk.citation}" for e in state.evidences
        ))
    if state.conflicts:
        lines.append("版本冲突：" + "；".join(state.conflicts))
    if state.verification:
        lines.append(
            f"引用校验：ok={state.verification.get('ok')} "
            f"method={state.verification.get('method')} "
            f"issues={len(state.verification.get('issues', []))}"
        )
    if state.degrade_reasons:
        lines.append("降级原因：" + "；".join(state.degrade_reasons))
    return "\n".join(lines)


def save_json(state: AgentState, path: str | Path, *, include_raw_query: bool = False) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(trace_dict(state, include_raw_query=include_raw_query),
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return out


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _css() -> str:
    return """
:root{--bg:#f7f8fa;--panel:#fff;--line:#e3e6ec;--fg:#1b1f27;--muted:#6b7280;
--accent:#2563eb;--ok:#0f7b4f;--err:#b42318;--warn:#a15c00;--chip:#eef2ff}
@media (prefers-color-scheme:dark){:root{--bg:#111318;--panel:#1a1e26;--line:#2c3240;
--fg:#e8ebf2;--muted:#9aa3b2;--accent:#7aa2ff;--ok:#5ad39b;--err:#ff8b80;
--warn:#e8b562;--chip:#232a3a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding:32px 20px 64px}h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--muted);margin-bottom:24px}.panel{background:var(--panel);
border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin-bottom:18px}
h2{font-size:15px;margin:0 0 12px;color:var(--accent)}
.meta{display:flex;flex-wrap:wrap;gap:8px}.chip{background:var(--chip);border-radius:6px;
padding:3px 9px;font-size:12px}.ok{color:var(--ok)}.err{color:var(--err)}.warn{color:var(--warn)}
.step{border-left:3px solid var(--line);padding:0 0 16px 16px;margin-left:6px;position:relative}
.step:last-child{padding-bottom:0}.step::before{content:"";position:absolute;left:-6px;top:6px;
width:9px;height:9px;border-radius:50%;background:var(--accent)}
.step.fail::before{background:var(--err)}.lbl{color:var(--muted);font-size:12px}
pre{white-space:pre-wrap;word-break:break-word;margin:4px 0 0;font:inherit}
.answer{white-space:pre-wrap}ul{margin:6px 0 0;padding-left:20px}
"""


def render_html(state: AgentState) -> str:
    """独立 HTML 报告，可用浏览器直接打开；所有动态内容均做转义。"""
    stop = state.stop_reason.value if state.stop_reason else "-"
    meta = [
        ("模式", state.decision_mode), ("终止原因", stop),
        ("步数", state.step), ("模型调用", state.api_calls),
        ("决策回退", state.policy_fallbacks),
        ("拒答", "是" if state.refused else "否"),
        ("降级", "是" if state.degraded else "否"),
        ("消融", ",".join(sorted(state.ablations)) or "无"),
        ("查询指纹", f"chars={len(state.raw_query)}（原文已脱敏）"),
        ("生成时间", _now()),
    ]
    meta_html = "".join(
        f'<span class="chip">{_esc(k)}: {_esc(v)}</span>' for k, v in meta
    )

    steps = []
    for t in state.tool_calls:
        args = json.dumps(t.args, ensure_ascii=False) if t.args else "{}"
        cls = "step" if t.ok else "step fail"
        steps.append(
            f'<div class="{cls}"><div><b>#{t.step} {_esc(t.action)}</b> '
            f'<span class="lbl">{t.ms} ms · {"成功" if t.ok else "失败"}</span></div>'
            f'<div class="lbl">thought</div><pre>{_esc(t.thought or "-")}</pre>'
            f'<div class="lbl">args</div><pre>{_esc(args)}</pre>'
            f'<div class="lbl">observation</div><pre>{_esc(t.observation)}</pre></div>'
        )

    ev_html = "".join(
        f"<li>{_esc(e.label)} {_esc(e.chunk.citation)} "
        f"score={e.score:.4f} [{_esc(e.source)}]</li>" for e in state.evidences
    ) or "<li>（无）</li>"
    conflicts = "".join(f"<li>{_esc(c)}</li>" for c in state.conflicts) or "<li>（无）</li>"
    reasons = "".join(f"<li>{_esc(r)}</li>" for r in state.degrade_reasons) or "<li>（无）</li>"
    issues = state.verification.get("issues", []) if state.verification else []
    verify_html = (
        f"ok={_esc(state.verification.get('ok'))} · "
        f"method={_esc(state.verification.get('method'))} · "
        f"issues={len(issues)}"
        if state.verification else "（未执行）"
    )

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent 轨迹 · {_esc(stop)}</title><style>{_css()}</style></head><body>
<div class="wrap">
  <h1>校规问答 Agent · 决策轨迹</h1>
  <div class="sub">schema={_esc(_SCHEMA)} · 导出内容默认脱敏，不含用户原始提示词全文</div>
  <div class="panel"><h2>运行摘要</h2><div class="meta">{meta_html}</div></div>
  <div class="panel"><h2>最终答案</h2><div class="answer">{_esc(state.answer or '（无）')}</div></div>
  <div class="panel"><h2>决策时间线：thought → tool → observation</h2>{''.join(steps) or '<div class="lbl">（无步骤）</div>'}</div>
  <div class="panel"><h2>证据与校验</h2><ul>{ev_html}</ul>
    <div class="lbl" style="margin-top:10px">引用校验：{verify_html}</div></div>
  <div class="panel"><h2>版本冲突</h2><ul>{conflicts}</ul></div>
  <div class="panel"><h2>降级原因</h2><ul>{reasons}</ul></div>
</div></body></html>
"""


def save_html(state: AgentState, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(state), encoding="utf-8")
    return out
