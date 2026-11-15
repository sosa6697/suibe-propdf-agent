"""命令行入口：config / check / build-index / retrieve / ask / agent / eval /
analytics / history / clear / serve。

ask 走 v0.1 固定流水线；agent 走手写 Agent Loop。后者可导出脱敏决策轨迹。
analytics 只读结构化事件库，输出文本 / JSON 报表或导出独立 HTML 快照，零模型调用。
"""
from __future__ import annotations

import argparse
import json
import sys

from .agent import AgentLoop
from .agent.trace import render_text, save_html, save_json, trace_dict
from .config import PROJECT_ROOT, Settings
from .index.bm25_store import BM25Store
from .index.vector_store import VectorStore
from .pipeline import QAPipeline
from .schema import QueryRequest


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def _health_ok(base: str, timeout: float = 1.0) -> bool:
    """端口上是否有服务在监听（只看 /health，不判断版本）。"""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{base}/health", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def _service_ready(base: str, timeout: float = 1.0) -> bool:
    """探测 /analytics/summary：既是健康检查，也确认是**当前版本**的服务。

    只读端点，不触发任何模型调用。旧版本的服务（v0.3 等，没有数据中台端点）会被
    判为「未就绪」，避免一键启动把浏览器带到一个 404 页面。
    """
    import urllib.request

    try:
        with urllib.request.urlopen(f"{base}/analytics/summary", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001 - 连不上或版本不对都算「未就绪」
        return False


def _open_browser_when_ready(base: str, page: str, timeout: float = 90.0) -> None:
    """后台线程轮询 /health，服务就绪后自动开浏览器；不阻塞 uvicorn 主线程。

    轮询的是本地只读端点，全程 0 次模型 API 调用。
    """
    import threading
    import time
    import webbrowser

    def _worker() -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _service_ready(base, timeout=1):
                url = f"{base}/{page}"
                try:
                    webbrowser.open(url)
                    print(f"已自动打开浏览器：{url}")
                except Exception as exc:  # noqa: BLE001
                    print(f"[warn] 无法自动打开浏览器，请手动访问 {url}（{exc}）")
                return
            time.sleep(0.5)
        print(f"[warn] 服务未在 {timeout:.0f}s 内就绪，请手动访问 {base}/{page}")

    threading.Thread(target=_worker, name="open-browser", daemon=True).start()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="campus-rules",
        description="上海对外经贸大学校规问答 Agent（准确性优先的可观测 RAG 流水线）",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("config", help="打印能力摘要（不含任何密钥）")
    sub.add_parser("check", help="链路自检：数据对齐、索引、能力状态、一次试检索")

    b = sub.add_parser("build-index", help="构建/重建 BM25 索引缓存")
    b.add_argument("--force", action="store_true", help="忽略已有缓存，强制重建")

    r = sub.add_parser("retrieve", help="只跑检索链路，不生成答案")
    r.add_argument("query")
    r.add_argument("--top-k", type=int, default=10)
    r.add_argument("--json", action="store_true")

    a = sub.add_parser("ask", help="v0.1 固定流水线问答（对照组）")
    a.add_argument("query")
    a.add_argument("--session", default="default")
    a.add_argument("--json", action="store_true")

    ag = sub.add_parser("agent", help="手写 Agent Loop 问答（决策轨迹可导出）")
    ag.add_argument("query")
    ag.add_argument("--session", default="default")
    ag.add_argument("--mode", choices=("llm", "policy"), default="llm",
                    help="llm=模型自主决策；policy=确定性固定流程（消融对照）")
    ag.add_argument("--max-steps", type=int, default=6)
    ag.add_argument("--ablate", default="",
                    help="逗号分隔：no_vector,no_expand,no_rerank,no_verify")
    ag.add_argument("--json", action="store_true", help="输出结构化脱敏轨迹 JSON")
    ag.add_argument("--trace-out", default="", help="导出脱敏 JSON 轨迹文件")
    ag.add_argument("--trace-html", default="", help="导出独立 HTML 轨迹报告")

    ev = sub.add_parser("eval", help="评测 / 消融（默认 dry-run，零 API 调用）")
    ev.add_argument("--mode", choices=("baseline", "agent"), default="baseline")
    ev.add_argument("--decision", choices=("llm", "policy"), default="policy")
    ev.add_argument("--ablate", default="")
    ev.add_argument("--max-steps", type=int, default=6)
    ev.add_argument("--categories", default="")
    ev.add_argument("--ids", default="", help="逗号分隔的 case_id，只跑指定题目（诊断用）")
    ev.add_argument("--limit", type=int, default=0)
    ev.add_argument("--run", action="store_true", help="真实执行（会发起模型 API 调用）")
    ev.add_argument("--confirm-api-calls", action="store_true",
                    help="确认调用清单已获批准；与 --run 同时给出才执行")
    ev.add_argument("--out", default="")
    ev.add_argument("--golden", default="")

    an = sub.add_parser("analytics", help="数据中台报表（零模型调用）")
    an.add_argument("--json", action="store_true", help="输出聚合报表 JSON")
    an.add_argument("--html", default="", help="导出带数据的独立 HTML 看板快照")
    an.add_argument("--reseed", action="store_true", help="强制重放历史真实记录（幂等）")
    an.add_argument("--top-docs", type=int, default=10, help="文档排行条数")

    h = sub.add_parser("history", help="查看某会话的短期记忆")
    h.add_argument("--session", default="default")

    c = sub.add_parser("clear", help="清空某会话的短期记忆")
    c.add_argument("--session", default="default")

    s = sub.add_parser("serve", help="启动 FastAPI 服务")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--open", action="store_true",
                    help="服务就绪后自动打开浏览器（默认打开数据看板 /dashboard）")
    s.add_argument("--open-page", default="dashboard",
                   choices=("dashboard", "chat", "demo", "docs"),
                   help="--open 打开的页面，默认 dashboard")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.load()

    if args.cmd == "config":
        print(_dump({"config": settings.safe_summary(), "degrade_reasons": settings.degrade_reasons()}))
        return 0

    if args.cmd == "build-index":
        store = VectorStore(settings.data_dir).load()
        bm25 = BM25Store(settings.var_dir)
        if args.force or not bm25.path.exists():
            bm25.build(store.chunks)
        else:
            bm25.load(store.chunks)
        print(f"BM25 索引就绪：{bm25.path} | {len(store)} 块")
        return 0

    if args.cmd == "eval":
        # 延迟导入：eval 是项目根目录下的独立包，不随 src 安装
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from pathlib import Path

        from eval.harness import EvalOptions
        from eval.harness import run as run_eval

        options = EvalOptions(
            mode=args.mode,
            decision=args.decision,
            ablations=tuple(x.strip() for x in args.ablate.split(",") if x.strip()),
            max_steps=args.max_steps,
            categories=tuple(x.strip() for x in args.categories.split(",") if x.strip()),
            ids=tuple(x.strip() for x in args.ids.split(",") if x.strip()),
            limit=args.limit,
            run=args.run,
            confirm_api_calls=args.confirm_api_calls,
            out=Path(args.out) if args.out else None,
            golden=Path(args.golden) if args.golden else PROJECT_ROOT / "eval" / "golden.jsonl",
        )
        try:
            return run_eval(options)
        except (ValueError, FileNotFoundError) as exc:
            print(f"评测配置错误：{exc}", file=sys.stderr)
            return 2

    if args.cmd == "analytics":
        # 不需要加载向量库 / BM25，也绝不触发任何模型调用
        from pathlib import Path

        from .analytics import (
            AnalyticsStore, build_report, ensure_seeded,
            render_dashboard_html, render_text,
        )

        store = AnalyticsStore(settings.var_dir / "analytics.db")
        seed_info = ensure_seeded(store, root=PROJECT_ROOT, force=args.reseed)
        report = build_report(store, top_docs=args.top_docs)
        report["seed_info"] = seed_info
        print(_dump(report) if args.json else render_text(report))
        if args.html:
            out = Path(args.html)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(render_dashboard_html(report), encoding="utf-8")
            print(f"独立 HTML 看板已导出：{out.resolve()}")
        return 0

    pipeline = QAPipeline(settings)

    if args.cmd == "check":
        report = pipeline.health()
        probe = pipeline.retrieve("学籍管理规定")
        report["probe"] = {
            "candidates": len(probe.candidates),
            "degraded": bool(probe.reasons),
            "degrade_reasons": probe.reasons,
            "steps": [t["step"] + ":" + t["status"] for t in probe.trace],
        }
        print(_dump(report))
        return 0

    if args.cmd == "retrieve":
        result = pipeline.retrieve(args.query)
        payload = result.to_dict()
        payload["candidates"] = payload["candidates"][: args.top_k]
        if args.json:
            print(_dump(payload))
        else:
            print(f"检索到 {len(result.candidates)} 个候选，展示前 {len(payload['candidates'])} 个")
            if result.reasons:
                print("⚠ degraded：" + "；".join(result.reasons))
            for c in payload["candidates"]:
                print(f"\n#{c['rank']} {c['citation']} score={c['final_score']} src={','.join(c['sources'])}")
                print(f"   {c['preview']}")
        return 0

    if args.cmd == "ask":
        result = pipeline.ask(QueryRequest(raw_query=args.query, session_id=args.session))
        if args.json:
            print(_dump(result.to_dict()))
        else:
            print("\n" + result.answer)
            print("\n" + "-" * 60)
            if result.degraded:
                print("⚠ degraded：" + "；".join(result.degrade_reasons))
            print(f"拒答={result.refused} | 引用校验={result.verification.get('ok')} ({result.verification.get('method')})")
            for e in result.evidences:
                print(f"  {e.label} {e.chunk.citation} score={e.score:.4f} [{e.source}]")
            if result.trace:
                print("链路：" + " → ".join(f"{t['step']}:{t['status']}" for t in result.trace))
        return 0

    if args.cmd == "agent":
        ablations = tuple(x.strip() for x in args.ablate.split(",") if x.strip())
        loop = AgentLoop(
            pipeline,
            decision_mode=args.mode,
            max_steps=args.max_steps,
            ablations=ablations,
        )
        state = loop.run(QueryRequest(raw_query=args.query, session_id=args.session))
        print(_dump(trace_dict(state)) if args.json else render_text(state))
        if args.trace_out:
            path = save_json(state, args.trace_out)
            print(f"JSON 轨迹已导出：{path}")
        if args.trace_html:
            path = save_html(state, args.trace_html)
            print(f"HTML 轨迹已导出：{path}")
        return 0

    if args.cmd == "history":
        turns = pipeline.memory.history(args.session)
        print(_dump([{"role": t.role, "content": t.content, "at": t.created_at} for t in turns]))
        return 0

    if args.cmd == "clear":
        n = pipeline.memory.clear(args.session)
        print(f"已清空会话 {args.session} 的 {n} 条记忆")
        return 0

    if args.cmd == "serve":
        import uvicorn
        import webbrowser

        from .api import create_app

        base = f"http://{args.host}:{args.port}"
        target = f"{base}/{args.open_page}"
        print(f"启动服务：{base}  (文档 /docs)")
        print("可用页面：/dashboard 数据中台 | /chat 问答界面 | /demo 决策轨迹 | /docs 接口文档")
        print("关闭本窗口或按 Ctrl+C 停止服务；启动服务与打开看板均为 0 次模型调用")

        # 幂等：重复双击快捷方式时，当前版本服务已在跑就只把浏览器带到目标页面。
        if args.open:
            if _service_ready(base):
                print(f"服务已在运行：{base}（本次不重复启动）")
                webbrowser.open(target)
                print(f"已打开浏览器：{target}")
                return 0
            if _health_ok(base):
                print(f"[warn] 端口 {args.port} 被旧版本服务占用（有 /health、无 /analytics/summary）。")
                print("       它多半是早期会话遗留的 serve 进程，看板会 404。请先关闭那个窗口")
                print(f"       或按 Ctrl+C 停止它；也可换端口：serve --open --port {args.port + 1}")
                return 1
            _open_browser_when_ready(base, args.open_page)
        uvicorn.run(create_app(pipeline), host=args.host, port=args.port)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
