"""FastAPI 服务层：把同一条流水线暴露为 HTTP 接口，不另起一套逻辑。

端点：
  GET  /health          能力状态与降级原因
  POST /ask             v0.1 固定流水线问答（对照组）
  POST /agent           手写 Agent Loop 问答，返回脱敏决策轨迹
  GET  /agent/tools     Agent 可用工具清单（演示页用）
  GET  /guard/rules     入口过滤规则清单（零模型调用）
  POST /guard/check     只跑入口判定（零模型调用，不落盘、不回显提示词）
  GET  /analytics/summary  数据中台聚合指标 JSON
  GET  /dashboard      管理者看板（单文件 HTML，零 CDN）
  POST /retrieve        只跑检索链路
  GET  /demo            Agent 决策轨迹演示页（面试讲解用）
  GET  /chat            问答机器人界面（每条回答强制展示答案依据）
  GET/DELETE /sessions/{id}  多轮短期记忆

注意：/ask 与 /agent 在配置了外部能力时会真实消耗模型 API 调用。
按项目约定，发起前须先列出接口/次数/用途并获得明确批准。
/guard/*、/analytics/* 与 /retrieve 不调用任何模型。

数据中台口径：只在 /ask 与 /agent 记录**结构化事件**（主题标签 / 规则名 /
文档元数据 / 计数），写入独立 var/analytics.db，绝不存用户提示词原文。
只读端点、CLI、评测、离线自检都不写事件，因此运营统计不会被测试流量污染。
"""
from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .agent import AgentLoop
from .agent.tools import ToolRegistry
from .agent.trace import trace_dict
from .analytics import (
    AnalyticsStore,
    build_report,
    classify,
    ensure_seeded,
    render_dashboard_html,
)
from .config import PROJECT_ROOT
from .logging_conf import get_logger
from .pipeline import QAPipeline
from .query_guard import guard_query, signal_catalog
from .schema import QueryRequest

log = get_logger("api")

ABLATIONS = ("no_vector", "no_expand", "no_rerank", "no_verify")


class AskIn(BaseModel):
    query: str = Field(..., min_length=1, description="用户原始问题")
    session_id: str = Field(default="default", description="会话 id，用于多轮记忆")


class AgentIn(AskIn):
    mode: str = Field(default="llm", pattern="^(llm|policy)$")
    max_steps: int = Field(default=6, ge=1, le=12)
    ablations: list[str] = Field(default_factory=list)


class GuardIn(BaseModel):
    query: str = Field(..., min_length=1, description="只用于本地规则匹配，不落盘、不回显")


class RetrieveIn(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)


def _track(analytics: AnalyticsStore, query: str, session_id: str,
           payload: dict, *, mode: str) -> None:
    """把一次真实问答写成结构化事件。**query 只用于内存里做主题匹配，绝不入 SQL**。

    任何写入异常都只记 warning，不能让统计故障影响问答本身。

    `rid` 把同一次请求产生的多条 `guard_blocked`（一次可命中多条软规则）关联起来，
    报表据此按「被拦请求数」算拦截率，避免规则命中次数把比率推到 100% 以上。
    它只是随机标识，不含任何用户输入文本。
    """
    try:
        rid = uuid.uuid4().hex
        labels = classify(query)
        analytics.record(
            "query_received", session_id=session_id, topic=labels[0], topics=labels,
            meta={"mode": mode, "rid": rid},
        )
        if payload.get("guard_blocked"):
            rules = (payload.get("guard") or {}).get("matched_rules") or ["-"]
            for rule in rules:
                analytics.record(
                    "guard_blocked", session_id=session_id, rule=rule,
                    meta={"rid": rid},
                )
        for ev in payload.get("evidences") or []:
            analytics.record(
                "evidence_used", session_id=session_id,
                doc_title=ev.get("doc_title"), page_no=ev.get("page_no"),
                score=ev.get("score"),
                meta={"rerank_source": ev.get("source"), "label": ev.get("label")},
            )
        # /agent 的脱敏 trace 带 api_calls；/ask 的 AnswerResult 不携带调用计数，
        # 因此这里显式标记 known=false，绝不把「取不到」伪装成「消耗为 0」。
        analytics.record(
            "answer_completed", session_id=session_id,
            n=int(payload.get("api_calls") or 0),
            meta={"refused": bool(payload.get("refused")), "mode": mode,
                  "api_calls_known": "api_calls" in payload, "rid": rid},
        )
    except Exception as exc:  # noqa: BLE001 - 统计写入失败绝不能拖垮问答
        log.warning("数据中台事件写入失败（不影响问答）：%s", exc)


def create_app(pipeline: QAPipeline, *, analytics: AnalyticsStore | None = None) -> FastAPI:
    # 默认落盘到独立 var/analytics.db；调用方（如离线自检）可显式传 :memory: 库。
    own_store = analytics is None
    if analytics is None:
        analytics = AnalyticsStore(PROJECT_ROOT / "var" / "analytics.db")
    if own_store:
        try:
            ensure_seeded(analytics, root=PROJECT_ROOT)
        except Exception as exc:  # noqa: BLE001 - 历史文件缺失不应阻止服务启动
            log.warning("历史回放跳过：%s", exc)

    app = FastAPI(
        title="校规问答 Agent",
        version="0.6.0",
        description="手写 Agent Loop + 准确性优先的可观测 RAG；含结构化事件数据中台。",
    )
    registry = ToolRegistry(pipeline)

    @app.get("/health")
    def health():
        return {**pipeline.health(), "agent_tools": list(registry.tools)}

    @app.post("/ask")
    def ask(body: AskIn):
        # 提示词**原文**采集仍是后续需求：QueryRequest.raw_query 原样保留但不落盘。
        # 这里只把内存里算出的主题标签 / 规则名 / 文档元数据写进 analytics.db。
        result = pipeline.ask(QueryRequest(raw_query=body.query, session_id=body.session_id))
        payload = result.to_dict()
        _track(analytics, body.query, body.session_id, payload, mode="baseline")
        return payload

    @app.post("/agent")
    def agent(body: AgentIn):
        unknown = set(body.ablations) - set(ABLATIONS)
        if unknown:
            raise HTTPException(status_code=422, detail=f"未知消融开关：{sorted(unknown)}")
        loop = AgentLoop(
            pipeline,
            decision_mode=body.mode,
            max_steps=body.max_steps,
            ablations=tuple(body.ablations),
        )
        state = loop.run(QueryRequest(raw_query=body.query, session_id=body.session_id))
        payload = trace_dict(state)  # 默认脱敏，不含用户原始提示词全文
        _track(analytics, body.query, body.session_id, payload, mode="agent")
        return payload

    @app.get("/agent/tools")
    def agent_tools():
        return registry.specs()

    @app.get("/guard/rules")
    def guard_rules():
        """入口过滤规则清单：不读数据、不调模型，用于讲解判定依据。"""
        return {"strategy": "conservative_pass", "rules": signal_catalog()}

    @app.post("/guard/check")
    def guard_check(body: GuardIn):
        """只跑入口判定。响应只含规则名与固定文案，绝不回显或落盘用户输入。"""
        return guard_query(body.query).to_dict()

    @app.get("/analytics/summary")
    def analytics_summary():
        """数据中台聚合指标：主题输入率 / 拦截率 / 被索引最多的数据。零模型调用。"""
        return build_report(analytics)

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard():
        """管理者看板：服务端把当前报表注入单文件模板，不依赖任何前端框架。"""
        return HTMLResponse(render_dashboard_html(build_report(analytics)))

    def serve_page(filename: str) -> HTMLResponse:
        page = PROJECT_ROOT / "web" / filename
        if not page.exists():
            return HTMLResponse(f"<h1>web/{filename} 缺失</h1>", status_code=404)
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/demo", response_class=HTMLResponse)
    def demo():
        return serve_page("index.html")

    @app.get("/chat", response_class=HTMLResponse)
    def chat():
        """面向使用者的问答界面；默认离线回放真实历史记录，不主动发起模型调用。"""
        return serve_page("chat.html")

    @app.post("/retrieve")
    def retrieve(body: RetrieveIn):
        result = pipeline.retrieve(body.query)
        payload = result.to_dict()
        payload["candidates"] = payload["candidates"][: body.top_k]
        return payload

    @app.get("/sessions/{session_id}")
    def history(session_id: str):
        return [
            {"role": t.role, "content": t.content, "at": t.created_at}
            for t in pipeline.memory.history(session_id)
        ]

    @app.delete("/sessions/{session_id}")
    def clear(session_id: str):
        return {"deleted": pipeline.memory.clear(session_id)}

    return app
