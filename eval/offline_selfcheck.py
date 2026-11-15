"""零 API 调用离线自检：一次跑完所有新增能力的验证。

外部能力全部置空（不是「假设不调用」，而是把配置改成不可用），
并在适配器上挂计数器，结束时断言计数必须为 0。

运行：python -m eval.offline_selfcheck
"""
from __future__ import annotations

import compileall
import contextlib
import io
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from campus_rules.agent import AgentLoop, AgentState, Decision, StopReason, ToolRegistry  # noqa: E402
from campus_rules.agent import trace as trace_mod  # noqa: E402
from campus_rules.agent.policy import BASELINE_PLAN, default_plan  # noqa: E402
from campus_rules.analytics import (  # noqa: E402
    AnalyticsStore, PLACEHOLDER, build_report, classify, ensure_seeded,
    render_dashboard_html, render_text, seed_from_history,
)
from campus_rules.config import Settings  # noqa: E402
from campus_rules.pipeline import QAPipeline  # noqa: E402
from campus_rules.query_guard import (  # noqa: E402
    GUARD_REFUSAL, guard_query, signal_catalog,
)
from campus_rules.schema import QueryRequest  # noqa: E402
from eval import harness  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def offline_settings() -> Settings:
    """把三项外部能力改成不可用，从配置层杜绝任何模型请求。"""
    return replace(
        Settings.load(),
        llm_api_key=None, llm_api_url=None, llm_model=None,
        embedding_api_key=None, embedding_api_url=None, embedding_model=None,
        rerank_api_url=None, rerank_api_key=None, rerank_model=None,
        enable_vector_recall=False,
    )


def count_adapters(p: QAPipeline) -> dict[str, int]:
    counter = {"n": 0}
    for obj, name in ((p.embedding, "embed"), (p.rerank, "rerank"), (p.llm, "chat")):
        original = getattr(obj, name)

        def make(orig):
            def inner(*a, **kw):
                counter["n"] += 1
                return orig(*a, **kw)
            return inner

        setattr(obj, name, make(original))
    return counter


def main() -> int:
    print("=" * 72)
    print("campus-rules-agent · 零 API 调用离线自检")
    print("=" * 72)

    # 1. 编译
    print("\n[1] 语法编译")
    ok_src = compileall.compile_dir(str(SRC), quiet=2, force=True)
    ok_eval = compileall.compile_dir(str(ROOT / "eval"), quiet=2, force=True)
    check("compileall src/ + eval/", bool(ok_src and ok_eval))

    # 2. 配置
    print("\n[2] 配置与能力置空")
    s = offline_settings()
    check("三项外部能力均不可用",
          not s.embedding_ready and not s.llm_ready and not s.rerank_ready)
    check("静态降级原因完整", len(s.degrade_reasons()) == 3, str(s.degrade_reasons()))

    p = QAPipeline(s)
    counter = count_adapters(p)
    check("向量库行号对齐", len(p.store) == 8751, f"chunks={len(p.store)} dim={p.store.dim}")

    # 3. 工具注册表
    print("\n[3] 工具注册表")
    reg = ToolRegistry(p)
    expected = {"retrieve", "rerank", "check_version", "pack_evidence", "answer", "refuse"}
    check("6 个工具全部注册", set(reg.tools) == expected, ",".join(reg.tools))
    check("工具清单可注入决策器", all(t["name"] in reg.specs_text() for t in reg.specs()))

    # 4. policy 决策序列
    print("\n[4] 确定性策略与消融对照")
    st = AgentState(raw_query="自检问题", session_id="selfcheck")
    seq = []
    for i in range(len(BASELINE_PLAN) + 1):
        st.step = i
        seq.append(default_plan(st).action)
    check("无证据时 answer 自动改 refuse",
          seq[:4] == ["retrieve", "rerank", "check_version", "pack_evidence"]
          and seq[4] == "refuse", "→".join(seq))
    st2 = AgentState(raw_query="x", session_id="selfcheck")
    st2.step = 4
    # 用一条最小 Evidence 验证「有证据 → answer」分支
    from campus_rules.schema import Chunk, Evidence  # noqa: E402
    from datetime import date  # noqa: E402
    c = Chunk(0, "c0", "内容", "d0", "f0", 1, "标题", "pdf_page", 0, date(2024, 1, 1), False)
    st2.evidences = [Evidence("E1", c, 1.0, "rule_rerank")]
    check("有证据时按计划执行 answer", default_plan(st2).action == "answer")

    # 5. 决策 JSON 解析
    print("\n[5] 决策 JSON 解析与容错")
    d = AgentLoop._parse_decision('```json\n{"thought":"t","action":"retrieve","args":{"query":"q"}}\n```')
    check("容错 Markdown 包裹", d.action == "retrieve" and d.args["query"] == "q")
    try:
        AgentLoop._parse_decision("没有任何 JSON")
        check("非法响应必须报错", False)
    except ValueError:
        check("非法响应必须报错", True)

    # 6. 状态脱敏
    print("\n[6] AgentState 脱敏序列化")
    s3 = AgentState(raw_query="这是不应出现在导出结果里的原始提示词", session_id="selfcheck")
    red = s3.to_dict()
    full = s3.to_dict(include_raw_query=True)
    check("默认不输出提示词原文", "raw" not in red["query"] and red["query"]["sha8"])
    check("显式开关才输出原文", full["query"]["raw"] == s3.raw_query)
    check("包含新增计数字段", "policy_fallbacks" in red and "ablations" in red)

    # 7. trace 导出
    print("\n[7] trace 导出")
    td = trace_mod.trace_dict(s3)
    txt = trace_mod.render_text(s3)
    htm = trace_mod.render_html(s3)
    check("JSON schema 正确", td["schema"] == "campus-rules-agent/trace-v1" and td["redacted"])
    check("文本/HTML 渲染可用", "Agent 轨迹" in txt and "<html" in htm)
    out_dir = s.var_dir / "traces"
    jf = trace_mod.save_json(s3, out_dir / "selfcheck.json")
    hf = trace_mod.save_html(s3, out_dir / "selfcheck.html")
    check("JSON / HTML 文件已写出", jf.exists() and hf.exists())
    check("导出文件不含提示词原文",
          s3.raw_query not in jf.read_text(encoding="utf-8")
          and s3.raw_query not in hf.read_text(encoding="utf-8"))

    # 8. 完整离线 Agent 循环
    print("\n[8] 完整离线 Agent 循环")
    loop = AgentLoop(p, decision_mode="policy", max_steps=6,
                     ablations=("no_vector", "no_rerank", "no_verify"))
    state = loop.run(QueryRequest(raw_query="学生违纪处分有哪几种？", session_id="selfcheck-run"))
    check("循环正常收敛", state.stop_reason is StopReason.REFUSED, str(state.stop_reason))
    check("轨迹步数 = 6", state.step == 6 and len(state.tool_calls) == 6)
    check("外部能力不可用时不生成答案", state.api_calls == 0 and state.refused)
    check("降级原因含配置缺失，但正常配置不污染降级",
          any("LLM 未配置" in r for r in state.degrade_reasons)
          and not any("消融实验开启" in r for r in state.degrade_reasons)
          and "no_rerank" in state.ablations and state.degraded)
    check("证据已打包且带溯源", len(state.evidences) == 5
          and all("第" in e.chunk.citation for e in state.evidences))

    # 9. 重复动作护栏（脚本化决策器，不触碰外部能力）
    print("\n[9] 重复动作护栏")
    class ScriptedLoop(AgentLoop):
        def _decide(self, state):
            return Decision(thought="重复调用同一工具", action="check_version", args={})

    guarded = ScriptedLoop(p, decision_mode="llm", max_steps=4).run(
        QueryRequest(raw_query="护栏自检", session_id="selfcheck-guard")
    )
    check("重复动作触发收敛", any("重复动作" in r for r in guarded.degrade_reasons),
          f"steps={guarded.step}")
    check("护栏后主动拒答", guarded.refused and guarded.stop_reason is StopReason.REFUSED)

    # 10. baseline 消融开关 + API rerank 时效纠偏
    print("\n[10] 固定流水线消融 / 时效纠偏")
    p_ab = QAPipeline(s, ablations=("no_vector", "no_expand", "no_rerank", "no_verify"))
    counter_ab = count_adapters(p_ab)
    ret = p_ab.retrieve("学籍管理规定")
    statuses = {t["step"]: t["status"] for t in ret.trace}
    check("四项消融均体现在 trace",
          statuses.get("vector_recall") == "disabled"
          and statuses.get("neighbor_expansion") == "disabled"
          and statuses.get("rerank") == "degraded", str(statuses))
    check("消融后仍有候选且零调用",
          len(ret.candidates) > 0 and counter_ab["n"] == 0,
          f"candidates={len(ret.candidates)} calls={counter_ab['n']}")

    from datetime import date  # noqa: E402
    from campus_rules.index.hybrid import apply_recency_bias  # noqa: E402
    from campus_rules.schema import Chunk, ScoredChunk  # noqa: E402

    comp = Chunk(1, "c1", "汇编正文", "d1", "f1", 76,
                 "2018-04-02_上海对外经贸大学2017年管理文件汇编", "pdf_page", 0,
                 date(2018, 4, 2), True)
    single = Chunk(2, "c2", "单行规定正文", "d2", "f2", 1,
                   "2018-11-11_上海对外经贸大学学生违纪处分管理规定", "pdf_page", 0,
                   date(2018, 11, 11), False)
    a = ScoredChunk(chunk=comp, rerank_score=0.9820, final_score=0.9820)
    b = ScoredChunk(chunk=single, rerank_score=0.9743, final_score=0.9743)
    fixed = apply_recency_bias([a, b])
    check("API rerank 后汇编不再压过单行规定",
          fixed[0].chunk is single,
          f"#1={fixed[0].chunk.doc_title[-12:]} score={fixed[0].final_score:.4f}")

    # 10b. 拒答门槛 v0.3（词面主判 + 语义补位 + 分数尺度感知）
    from campus_rules.agent.tools import _NEEDS_CONTEXT  # noqa: E402
    from campus_rules.index.hybrid import detect_version_conflicts  # noqa: E402
    from campus_rules.prompts import build_verify_user  # noqa: E402
    from campus_rules.relevance import assess_relevance  # noqa: E402
    from campus_rules.schema import Evidence  # noqa: E402

    OUT_Q = "学校第一食堂一楼窗口每天几点停止供餐？"
    NORMAL_Q = "学生违纪处分有哪几种？"
    TRAP_Q = "哈佛大学的校长奖学金在我校如何申请？"
    SHORT_Q = "那学生对处分不服的申诉期限是多久？"

    def evidences_for(q, source="rule_rerank"):
        ret = p.retrieve(q)
        return [Evidence(f"E{i + 1}", c.chunk, c.final_score, source)
                for i, c in enumerate(ret.candidates[:5])]

    def gate(q, *, source="rule_rerank", sem=0.0, min_lex=0.30, min_sem=0.80):
        return assess_relevance(
            q, evidences_for(q, source), idf=p.bm25.idf_map,
            min_score=min_lex, semantic_score=sem, min_semantic=min_sem,
        )

    bad = gate(OUT_Q, sem=0.133)
    good = gate(NORMAL_Q, sem=0.10)
    check("域外问题被词面主判拦下", not bad.ok, f"lex={bad.score:.3f}")
    check("词面达标即放行，语义分不再否决", good.ok, f"lex={good.score:.3f} sem=0.10")
    trap = gate(TRAP_Q, source="rerank_api", sem=0.7988)
    check("R04 陷阱题 lex<0.30 且语义<0.80 仍拒", not trap.ok, f"lex={trap.score:.3f}")
    rescued = gate(SHORT_Q, source="rerank_api", sem=0.92)
    check("省略式追问由 API rerank 语义补位救回",
          rescued.ok and rescued.semantic_available, f"lex={rescued.score:.3f}")
    mismatch = gate(SHORT_Q, source="rule_rerank", sem=0.92)
    check("规则重排分数不触发语义补位（尺度错配保护）",
          not mismatch.ok and not mismatch.semantic_available
          and mismatch.semantic_source == "rule_rerank_scale_mismatch",
          mismatch.semantic_source)
    check("语义补位可单独关闭做消融",
          not gate(SHORT_Q, source="rerank_api", sem=0.92, min_sem=0).ok)
    check("门槛可整体关闭做消融", assess_relevance("任意", [], min_score=0).ok)
    check("结果字典带 semantic_source 便于追溯",
          "semantic_source" in rescued.to_dict() and "semantic_available" in rescued.to_dict())
    check("核查提示词已带上用户问题",
          "【用户问题】" in build_verify_user("答案", [], "测试问题"))

    v1 = Chunk(3, "v1", "正文", "dv1", "fv1", 1,
               "2022-10-29_上海对外经贸大学本科毕业论文管理条例（2021）",
               "pdf_page", 0, date(2022, 10, 29), False)
    v2 = Chunk(4, "v2", "正文", "dv2", "fv2", 1,
               "2023-10-23_上海对外经贸大学本科毕业论文管理条例（2023）",
               "pdf_page", 0, date(2023, 10, 23), False)
    check("括号年份归一化后可检出版本冲突",
          len(detect_version_conflicts([ScoredChunk(chunk=v1), ScoredChunk(chunk=v2)])) == 1)
    check("多轮补全只对指代/省略式追问生效",
          bool(_NEEDS_CONTEXT.search("那申诉期限是多久？"))
          and not _NEEDS_CONTEXT.search("住宿公约里对学生作息有什么要求？"))
    picked = harness._select_cases(harness.load_cases(),
                                   harness.EvalOptions(ids=("F02", "V01")))
    check("--ids 筛选生效", [c["id"] for c in picked] == ["F02", "V01"])

    # 10c. 指标口径：红线拒答不污染 degraded；citation_ok 区分「通过」与「未测」
    # v0.5 起 OUT_Q / TRAP_Q 已在入口被 query_guard 拦截，改用一道「不命中入口
    # 规则、但检索后仍证据不足」的域外题，保留 relevance_gate 红线的完整回归覆盖。
    print("\n[10c] 指标口径回归")
    RED_Q = "量子纠缠现象最早是谁提出的？"
    res = p.ask(QueryRequest(raw_query=RED_Q, session_id="selfcheck-redline"))
    check("红线拒答单独留痕 redline_refusals",
          bool(res.redline_refusals) and res.refused, str(res.redline_refusals)[:70])
    check("红线原因不写进 degrade_reasons",
          not any("相关性" in r for r in res.degrade_reasons), str(res.degrade_reasons)[:80])
    check("AnswerResult.to_dict 暴露 redline_refusals",
          "redline_refusals" in res.to_dict())

    st_red = AgentLoop(
        p, decision_mode="policy", max_steps=6,
        ablations=("no_vector", "no_rerank", "no_verify"),
    ).run(QueryRequest(raw_query=RED_Q, session_id="selfcheck-redline-agent"))
    check("Agent 路径红线拒答同样单独留痕",
          bool(st_red.redline_refusals)
          and not any("相关性" in r for r in st_red.degrade_reasons))
    check("AgentState.to_dict 暴露 redline_refusals",
          "redline_refusals" in st_red.to_dict())

    check("未执行核查 => citation_ok=None",
          harness._citation_ok({"ok": True, "method": "ablated"}) is None
          and harness._citation_ok({"ok": True, "method": "skipped"}) is None
          and harness._citation_ok(None) is None)
    check("真正跑了核查才判定通过/不通过",
          harness._citation_ok({"ok": True, "method": "rules+llm"}) is True
          and harness._citation_ok({"ok": False, "method": "rules+llm"}) is False)

    def fake_record(cid, cite):
        return {"id": cid, "category": "fact", "scored_for_title": True,
                "title_hit": True, "refusal_correct": True, "citation_ok": cite,
                "conflict_detected": False, "steps": 1, "api_calls": 0,
                "degraded": False}

    agg_none = harness._aggregate([fake_record("x1", None), fake_record("x2", None)])
    check("全组未核查时 citation_pass_rate=None（不再假 1.00）",
          agg_none["totals"]["citation_pass_rate"] is None
          and agg_none["totals"]["citation_scored_cases"] == 0)
    agg_mix = harness._aggregate(
        [fake_record("x1", True), fake_record("x2", False), fake_record("x3", None)]
    )
    check("混合组只按可评题算分母",
          agg_mix["totals"]["citation_pass_rate"] == 0.5
          and agg_mix["totals"]["citation_scored_cases"] == 2)

    # 10d. v0.4：核查口径统一 / 年份误伤 / 确定性自我修正 / 答案级金标
    print("\n[10d] 核查口径 / 年份数字 / 自我修正 / 答案级金标")
    import inspect  # noqa: E402
    from types import SimpleNamespace  # noqa: E402

    from campus_rules.agent.tools import ToolRegistry as _ToolRegistry  # noqa: E402
    from campus_rules.pipeline import REFUSAL  # noqa: E402
    from campus_rules.prompts import ANSWER_SYSTEM, VERIFY_SYSTEM  # noqa: E402
    from campus_rules.verification import verify_answer  # noqa: E402

    q_state = AgentState(raw_query="本轮真正的问题", session_id="selfcheck-q")
    q_state.rewritten_queries.append("上一轮问题 本轮真正的问题")
    check("生成/核查问题范围固定为本轮原话，不被检索拼接词污染",
          q_state.answer_query == "本轮真正的问题"
          and q_state.effective_query != q_state.answer_query)
    check("核查器被明确禁止要求覆盖历史问题 / 做证据外间接推断",
          "只答本轮问题" in VERIFY_SYSTEM and "禁止间接推断" in VERIFY_SYSTEM)
    src_answer = inspect.getsource(_ToolRegistry._answer)
    src_ask = inspect.getsource(QAPipeline.ask)
    check("Agent 与 baseline 的核查 query 都不再用检索改写词",
          "question = state.answer_query" in src_answer
          and "query=question" in src_answer
          and "question = req.raw_query" in src_ask
          and "query=question" in src_ask)

    ev_chunk = Chunk(9, "cv4", "处分分为警告、严重警告、记过、留校察看、开除学籍。",
                     "dv4", "fv4", 1, "学生违纪处分管理规定", "pdf_page", 0,
                     date(2024, 1, 1), False)
    ev_v4 = [Evidence("E1", ev_chunk, 1.0, "rule_rerank")]
    year_check = verify_answer("依据 2019 年文件，处分分为警告 [E1]。", ev_v4)
    check("年份不再被当作臆造数值（C05 回归）",
          year_check["ok"]
          and not any("数字" in i or "数值" in i for i in year_check["issues"]),
          str(year_check["issues"]))
    fake_num = verify_answer("处分种类共有 9999 种 [E1]。", ev_v4)
    check("真正的臆造数值仍被拦下",
          not fake_num["ok"] and any("数值" in i for i in fake_num["issues"]),
          str(fake_num["issues"]))

    class StubLLM:
        """只返回本地字符串，绝不发起任何网络请求。"""
        def __init__(self, verify_results):
            self.verify_results = list(verify_results)
            self.answer_prompts: list[str] = []
            self.verify_prompts: list[str] = []

        def chat(self, *, system, user, **kw):
            if system == ANSWER_SYSTEM:
                self.answer_prompts.append(user)
                return ("处分分为警告 [E1]" if len(self.answer_prompts) == 1
                        else "处分分为警告、严重警告、记过 [E1]")
            self.verify_prompts.append(user)
            payload = (self.verify_results.pop(0) if self.verify_results
                       else {"supported": True, "issues": []})
            return json.dumps(payload, ensure_ascii=False)

    def run_answer_with_stub(stub):
        s2 = replace(s, llm_api_key="stub", llm_api_url="stub", llm_model="stub",
                     min_relevance_score=0.0)
        fake_p = SimpleNamespace(
            s=s2, llm=stub,
            memory=SimpleNamespace(history_text=lambda sid: ""),
            bm25=SimpleNamespace(idf_map={}),
        )
        st = AgentState(raw_query="学生违纪处分有哪几种？", session_id="selfcheck-retry")
        st.evidences = list(ev_v4)
        obs = _ToolRegistry(fake_p).run("answer", st, {})
        return st, obs

    stub_ok = StubLLM([{"supported": False, "issues": ["答案遗漏了处分种类"]},
                       {"supported": True, "issues": []}])
    st_retry, _ = run_answer_with_stub(stub_ok)
    check("核查不通过时带意见重生成一次并采纳修正答案",
          st_retry.answer_retries == 1
          and st_retry.stop_reason is StopReason.VERIFIED
          and "严重警告" in st_retry.answer
          and st_retry.verification["ok"] is True,
          f"retries={st_retry.answer_retries} stop={st_retry.stop_reason}")
    check("第二次生成提示里带核查意见（确定性自我修正）",
          len(stub_ok.answer_prompts) == 2
          and "上次答案未通过核查" in stub_ok.answer_prompts[-1])
    check("核查器收到的是本轮用户原话（与生成同一句）",
          all("学生违纪处分有哪几种？" in u for u in stub_ok.verify_prompts))
    check("自我修正的调用计数可追溯且不触发外部请求",
          st_retry.api_calls == 4 and counter["n"] == 0)
    check("AgentState.to_dict 暴露 answer_retries",
          "answer_retries" in st_retry.to_dict())

    stub_bad = StubLLM([{"supported": False, "issues": ["证据不足"]},
                        {"supported": False, "issues": ["证据不足"]}])
    st_bad, _ = run_answer_with_stub(stub_bad)
    check("连续核查不通过按红线拒答，不伪装成能力降级",
          st_bad.stop_reason is StopReason.REFUSED and st_bad.refused
          and st_bad.answer == REFUSAL
          and bool(st_bad.redline_refusals) and not st_bad.degraded)

    fact_cases = harness.load_cases()
    check("20 题带答案级 key_facts，5 道拒答题为空",
          sum(1 for c in fact_cases if c["key_facts"]) == 20
          and sum(1 for c in fact_cases if not c["key_facts"]) == 5
          and all(not c["key_facts"] for c in fact_cases if c["expect_refusal"]))
    check("每条 key_facts 都有 facts_source 可追溯",
          all(c["facts_source"] for c in fact_cases if c["key_facts"]))
    check("事实锚点命中判定容忍空格 / 全角差异",
          harness._fact_hits("认定等级分为一般困难和特别困难 [E1]。",
                             ["一般困难", "特别困难"]) == [True, True]
          and harness._fact_hits("10 日内向申诉委员会提出书面申诉", ["10日内"]) == [True]
          and harness._fact_hits("每学年 12000 元", ["12000"]) == [True])
    check("未命中的锚点记 False 而不是丢题",
          harness._fact_hits("只写了警告 [E1]", ["警告", "开除学籍"]) == [True, False])

    def fact_record(cid, fr):
        r = fake_record(cid, True)
        r["fact_recall"] = fr
        return r

    agg_fact = harness._aggregate(
        [fact_record("f1", 1.0), fact_record("f2", 0.0), fact_record("f3", None)]
    )
    check("fact_recall_rate 只按可评题算均值（None 不进分母）",
          agg_fact["totals"]["fact_recall_rate"] == 0.5
          and agg_fact["totals"]["fact_scored_cases"] == 2)

    rec_v4 = harness._run_case(p, None, {"n": 0}, fact_cases[0], harness.EvalOptions())
    check("结果记录不落盘答案全文 / 锚点原文 / query",
          all(k not in rec_v4 for k in ("answer", "key_facts", "query"))
          and "fact_hits" in rec_v4 and "fact_recall" in rec_v4)

    # 10e. v0.5 入口提示词过滤：纯本地、保守放行、零调用、不落盘
    print("\n[10e] 入口过滤 query_guard")
    catalog = signal_catalog()
    check("规则清单可解释：1 条硬信号 + 5 条软信号",
          len(catalog) == 6 and sum(1 for s in catalog if s["hard"]) == 1,
          ",".join(s["name"] for s in catalog))

    g_cases = harness.load_cases()
    g_blocked = [c["id"] for c in g_cases if guard_query(c["query"]).blocked]
    g_expected = [c["id"] for c in g_cases if c["expect_refusal"]]
    check("25 题金标：5 挡 20 放，且拦截的恰是 5 道拒答题",
          len(g_cases) == 25 and g_blocked == g_expected and len(g_blocked) == 5,
          f"blocked={g_blocked}")
    check("硬信号（外校机构）不可被校规强信号覆盖",
          guard_query("哈佛大学的学生申诉处理办法里的校长奖学金在我校如何申请？").blocked
          and not guard_query("哈佛大学的学生申诉处理办法里的校长奖学金在我校如何申请？").domain_override)
    ov = guard_query("图书馆借阅管理办法规定的开放时间是什么？")
    check("软信号被校规强信号覆盖，避免误伤合法规章提问",
          not ov.blocked and ov.domain_override and "venue_hours" in ov.matched_rules,
          str(ov.matched_rules))
    check("未命中任何域外信号时保守放行（fail-open）",
          not guard_query("学生申诉处理办法里的申诉期限是多久？").blocked
          and not guard_query("").blocked)
    check("判定结果不含用户提示词原文，可安全进入响应 / trace / 结果文件",
          OUT_Q not in json.dumps(guard_query(OUT_Q).to_dict(), ensure_ascii=False)
          and guard_query(OUT_Q).to_dict()["api_calls"] == 0)

    before_guard = counter["n"]
    g_ask = p.ask(QueryRequest(raw_query=OUT_Q, session_id="selfcheck-guard-ask"))
    g_dict = g_ask.to_dict()
    check("baseline 入口拦截：固定拒答 / 0 证据 / 0 降级 / 0 调用",
          g_ask.guard_blocked and g_ask.refused
          and g_ask.answer == GUARD_REFUSAL
          and g_ask.evidences == [] and not g_ask.degraded
          and g_ask.degrade_reasons == []
          and counter["n"] == before_guard)
    check("baseline trace 只留 query_guard 一步且带规则名",
          len(g_ask.trace) == 1 and g_ask.trace[0]["step"] == "query_guard"
          and g_ask.trace[0]["status"] == "blocked"
          and OUT_Q not in json.dumps(g_ask.trace, ensure_ascii=False),
          str(g_ask.trace[0]["detail"])[:60])
    check("AnswerResult 暴露 guard_blocked / guard / redline_refusals",
          g_dict["guard_blocked"] is True
          and g_dict["guard"]["blocked"] is True
          and bool(g_ask.redline_refusals))
    check("baseline 入口拦截不写入会话记忆（不落盘域外提示词）",
          len(p.memory.history("selfcheck-guard-ask")) == 0)

    g_state = AgentLoop(p, decision_mode="llm", max_steps=6).run(
        QueryRequest(raw_query=OUT_Q, session_id="selfcheck-guard-agent")
    )
    check("Agent 入口拦截：循环不启动，step=0 / api_calls=0 / 专用终止原因",
          g_state.guard_blocked and g_state.refused
          and g_state.stop_reason is StopReason.GUARD_BLOCKED
          and g_state.step == 0 and g_state.api_calls == 0
          and not g_state.degraded and counter["n"] == before_guard)
    check("Agent 轨迹仍留一条可审计的 query_guard 记录",
          len(g_state.tool_calls) == 1
          and g_state.tool_calls[0].action == "query_guard"
          and g_state.tool_calls[0].step == 0)
    check("AgentState 脱敏导出暴露入口过滤且不回显提示词",
          g_state.to_dict()["guard_blocked"] is True
          and OUT_Q not in json.dumps(g_state.to_dict(), ensure_ascii=False)
          and OUT_Q not in trace_mod.render_text(g_state))
    check("Agent 入口拦截同样不写入会话记忆",
          len(p.memory.history("selfcheck-guard-agent")) == 0)

    normal_guard = p.ask(QueryRequest(raw_query=NORMAL_Q, session_id="selfcheck-guard-normal"))
    check("正常校规提问不受入口过滤影响",
          not normal_guard.guard_blocked and normal_guard.evidences
          and normal_guard.to_dict()["guard_blocked"] is False)

    r_case = next(c for c in g_cases if c["id"] == "R01")
    r_rec = harness._run_case(p, None, {"n": 0}, r_case, harness.EvalOptions())
    check("评测记录入口拒答：citation_ok=None / 0 调用 / 拒答判定正确",
          r_rec["guard_blocked"] is True and r_rec["citation_ok"] is None
          and r_rec["api_calls"] == 0 and r_rec["refusal_correct"] is True
          and "guard_rules" in r_rec
          and OUT_Q not in json.dumps(r_rec, ensure_ascii=False))
    check("聚合指标单列 guard_blocked_cases 与零调用题数",
          harness._aggregate([r_rec])["totals"]["guard_blocked_cases"] == 1
          and harness._aggregate([r_rec])["totals"]["guard_zero_call_cases"] == 1)

    # 11. 评测框架
    print("\n[11] 评测框架 / 金标题库")
    cases = harness.load_cases()
    cats = {c["category"] for c in cases}
    check("25 题 / 5 类", len(cases) == 25 and cats == set(harness.CATEGORIES))
    check("全部标记 needs_review（金标题非金答案）", all(c["needs_review"] for c in cases))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = harness.run(harness.EvalOptions(mode="agent", decision="llm", limit=2))
    check("dry-run 返回 0 且打印调用计划",
          rc == 0 and "[DRY-RUN]" in buf.getvalue() and "接口" in buf.getvalue())
    with contextlib.redirect_stderr(io.StringIO()):
        gate = harness.run(harness.EvalOptions(run=True, confirm_api_calls=False, limit=1))
    check("真实运行闸门拒绝未授权调用", gate == 2)

    # 12. API 端点
    print("\n[12] FastAPI 端点（只读端点）")
    from fastapi.testclient import TestClient  # noqa: E402
    from campus_rules.api import create_app  # noqa: E402

    # 数据中台自检用内存事件库：既能验证真实回放，又绝不污染 var/analytics.db
    mem_analytics = AnalyticsStore(":memory:")
    seed_info = ensure_seeded(mem_analytics, root=ROOT)
    app = create_app(p, analytics=mem_analytics)
    client = TestClient(app)
    check("GET /health 含 agent_tools",
          client.get("/health").json().get("agent_tools") is not None)
    check("GET /agent/tools 返回 6 个工具", len(client.get("/agent/tools").json()) == 6)
    rules_resp = client.get("/guard/rules")
    check("GET /guard/rules 返回规则清单与保守放行策略",
          rules_resp.status_code == 200
          and rules_resp.json()["strategy"] == "conservative_pass"
          and len(rules_resp.json()["rules"]) == 6)
    blocked_resp = client.post("/guard/check", json={"query": OUT_Q})
    passed_resp = client.post("/guard/check", json={"query": NORMAL_Q})
    check("POST /guard/check 零调用判定：域外拦 / 正常放",
          blocked_resp.status_code == 200
          and blocked_resp.json()["blocked"] is True
          and blocked_resp.json()["api_calls"] == 0
          and passed_resp.json()["blocked"] is False)
    check("/guard/check 响应不回显用户提示词",
          OUT_Q not in json.dumps(blocked_resp.json(), ensure_ascii=False)
          and counter["n"] == 0)
    ask_guard = client.post(
        "/ask", json={"query": OUT_Q, "session_id": "selfcheck-http-guard"}
    )
    check("POST /ask 端到端入口拦截：固定拒答 / 无证据 / 不回显 / 0 调用",
          ask_guard.status_code == 200
          and ask_guard.json()["guard_blocked"] is True
          and ask_guard.json()["answer"] == GUARD_REFUSAL
          and ask_guard.json()["evidences"] == []
          and OUT_Q not in json.dumps(ask_guard.json(), ensure_ascii=False)
          and counter["n"] == 0)
    agent_guard = client.post(
        "/agent", json={"query": OUT_Q, "session_id": "selfcheck-http-guard-agent", "mode": "llm"}
    )
    check("POST /agent 端到端入口拦截：step=0 / api_calls=0 / 专用终止原因",
          agent_guard.status_code == 200
          and agent_guard.json()["guard_blocked"] is True
          and agent_guard.json()["stop_reason"] == "guard_blocked"
          and agent_guard.json()["api_calls"] == 0
          and agent_guard.json()["steps"] == 0
          and counter["n"] == 0)
    demo = client.get("/demo")
    check("GET /demo 返回单文件页面", demo.status_code == 200 and "SAMPLE" in demo.text)
    bad = client.post("/agent", json={"query": "x", "ablations": ["unknown"]})
    check("非法消融开关返回 422", bad.status_code == 422)
    check("AnswerResult 新增 version_conflicts 字段",
          "version_conflicts" in p.ask(
              QueryRequest(raw_query="自检", session_id="selfcheck-ask")
          ).to_dict())

    # 12b. 问答机器人界面 / 「答案依据」数据契约
    print("\n[12b] /chat 界面与答案依据契约")
    chat = client.get("/chat")
    page = chat.text
    check("GET /chat 返回问答机器人界面",
          chat.status_code == 200 and "答案依据" in page and "DEMO_RUNS" in page)
    check("/chat 依据区渲染入口过滤徽标与零调用说明",
          "guard_blocked" in page and "入口过滤 · 与校规无关" in page
          and "模型调用 0 次 · 未进入检索链路" in page)
    check("现有 Agent 轨迹演示页未被破坏", client.get("/demo").status_code == 200)
    check("真实回放数据已注入（无占位符）", "/*__DEMO__*/" not in page)

    matched = re.search(r"const DEMO_RUNS = (\[[\s\S]*?\]);", page)
    runs = json.loads(matched.group(1)) if matched else []
    check("回放为真实历史记录且显式标注",
          len(runs) == 2 and all(r.get("recorded") is True for r in runs)
          and all(r.get("recorded_at") for r in runs))
    check("每轮回放都带 5 条证据原文与溯源",
          all(len(r.get("evidences", [])) == 5
              and all(e.get("excerpt") and e.get("doc_title")
                      and e.get("page_no") is not None for e in r["evidences"])
              for r in runs))
    check("有依据的答案含可定位的 [E1] 引用",
          bool(runs) and "[E1]" in runs[0]["answer"]
          and runs[0]["verification"]["ok"] is True)
    check("页面不落盘用户提示词（无浏览器存储调用）",
          "localStorage" not in page and "sessionStorage" not in page)
    check("界面不提供推荐语句 / 快捷问题（学生自己提问）",
          "const QUICK" not in page and 'id="chips"' not in page
          and "例如：" not in page and "快捷问题试试" not in page
          and 'placeholder="请输入你的问题…"' in page)
    check("发送各分支都在对话流内给出可见反馈",
          all(t in page for t in (
              "role === 'status'", "function pushStatus(", "function dropStatus(",
              "请先输入问题再发送", "当前是离线回放模式",
              "正在检索校规文件并生成答案", "请求失败，未产生答案",
              "上一条问题还在处理中", "本地服务连接失败"))
          and "sysmsg" in page)
    check("等待反馈可移除，失败时不伪造助手答案",
          "pending: true" in page and "dropStatus(waiting)" in page
          and "push('assistant', '请求失败" not in page)

    chat_ev = p.ask(QueryRequest(
        raw_query="学生违纪处分有哪几种？", session_id="selfcheck-chat"
    )).to_dict()["evidences"][0]
    check("/ask 证据带原文 / 标题 / 页码 / 汇编标记",
          all(k in chat_ev for k in ("excerpt", "doc_title", "page_no", "is_compilation"))
          and len(chat_ev["excerpt"]) > 0)
    agent_evs = st_red.to_dict()["evidences"]
    check("Agent 轨迹证据同样带原文片段",
          len(agent_evs) == 5 and all("excerpt" in e for e in agent_evs))
    p.memory.clear("selfcheck-chat")

    # 12c. 数据中台：结构化事件 / 三指标口径 / 隐私 / 看板
    print("\n[12c] 数据中台 analytics")
    import inspect  # noqa: E402
    import sqlite3  # noqa: E402

    ddl = mem_analytics.rows(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='events'"
    )[0]["sql"]
    check("事件表无提示词字段（query / content / prompt 均不建列）",
          all(w not in ddl.lower() for w in ("query", "content", "prompt")),
          ddl[:64])
    sig = inspect.signature(AnalyticsStore.record)
    check("record() 签名不接受提示词原文参数",
          all(w not in sig.parameters for w in ("query", "content", "prompt")),
          ",".join(sig.parameters))
    check("主题词典多标签命中；域外题归为未归类",
          classify("学生对处分不服的申诉期限是多久？") == ["纪律处分", "申诉复核"]
          and classify(OUT_Q) == ["未归类"] and classify("") == ["未归类"])

    seed_again = ensure_seeded(mem_analytics, root=ROOT)
    rows_before = build_report(mem_analytics)["overview"]["event_rows"]
    seed_from_history(mem_analytics, root=ROOT)
    check("历史回放读真实产物且 dedup_key 保证幂等",
          seed_info.get("result_files", 0) >= 1
          and seed_info.get("cases", 0) >= 1
          and seed_info.get("recomputed_guard", 0) >= 1
          and seed_again["skipped"] is True
          and build_report(mem_analytics)["overview"]["event_rows"] == rows_before,
          str(seed_info)[:90])

    rep = build_report(mem_analytics)
    ov = rep["overview"]
    check("主题输入率分母统一且逐项可复核（多标签口径）",
          ov["total_queries"] > 0 and rep["topics"]["multi_label"] is True
          and all(x["rate"] == round(x["count"] / ov["total_queries"], 4)
                  for x in rep["topics"]["items"]))
    check("拦截率以被拦请求数为分子，按规则拆分保留全部命中",
          ov["guard_rate"] == round(ov["guard_blocked"] / ov["total_queries"], 4)
          and ov["guard_rate"] is not None and ov["guard_rate"] <= 1
          and sum(x["count"] for x in rep["guard"]["by_rule"]) == ov["guard_rule_hits"]
          and ov["guard_rule_hits"] >= ov["guard_blocked"])

    # 回归：一次请求同时命中多条软规则时，历史上会把拦截率算成 >100%
    multi = AnalyticsStore(":memory:")
    multi.record("query_received", session_id="multi", topic="未归类",
                 meta={"rid": "req-1"})
    for _rule in ("dining_service", "venue_hours", "off_campus_transit"):
        multi.record("guard_blocked", session_id="multi", rule=_rule,
                     meta={"rid": "req-1"})
    mrep = build_report(multi)
    mo = mrep["overview"]
    check("一次请求命中多条软规则不会把拦截率推过 100%",
          mo["guard_blocked"] == 1 and mo["guard_rule_hits"] == 3
          and mo["guard_rate"] == 1.0
          and sum(x["count"] for x in mrep["guard"]["by_rule"]) == 3)
    docs = rep["documents"]["items"]
    check("被索引排行按清洗标题归并，无重复文档",
          bool(docs) and len(docs) == len({d["display_title"] for d in docs})
          and all(d["count"] > 0 for d in docs)
          and sum(d["count"] for d in docs) <= ov["evidence_events"])
    check("隐私声明明确不存提示词原文",
          rep["privacy"]["stores_raw_prompt"] is False
          and "doc_title" in rep["privacy"]["fields"])

    golden_queries = [c["query"] for c in g_cases]
    blob = mem_analytics.all_text()
    check("事件库不含任何金标题库 query 原文",
          not any(q in blob for q in golden_queries))

    summary = client.get("/analytics/summary")
    check("GET /analytics/summary 返回三类指标",
          summary.status_code == 200
          and all(k in summary.json() for k in
                  ("overview", "topics", "guard", "documents", "privacy")))
    dash = client.get("/dashboard")
    check("GET /dashboard 注入真实报表、无占位符残留、三大模块齐全",
          dash.status_code == 200 and "const DATA =" in dash.text
          and PLACEHOLDER not in dash.text
          and all(t in dash.text for t in
                  ("提示词关键词输入率", "拦截率", "被索引最多的数据")))
    tpl = (ROOT / "web" / "dashboard.html").read_text(encoding="utf-8")
    check("未注入时页面显示诚实空状态而非伪造数据",
          PLACEHOLDER in tpl and "typeof DATA === 'undefined'" in tpl
          and "不会伪造任何指标" in tpl)
    html_snapshot = render_dashboard_html(rep)
    check("HTML 快照可脱离服务独立渲染",
          "const DATA =" in html_snapshot and "校规问答 Agent · 数据中台" in html_snapshot
          and PLACEHOLDER not in html_snapshot)
    check("CLI 文本报表含三个模块与隐私说明",
          all(t in render_text(rep) for t in
              ("提示词关键词输入率", "拦截率", "被索引最多的数据", "不存用户提示词原文")))

    before_live = build_report(mem_analytics)["overview"]["live_events"]
    before_guard = build_report(mem_analytics)["overview"]["guard_blocked"]
    live_ask = client.post(
        "/ask", json={"query": OUT_Q, "session_id": "selfcheck-analytics-live"}
    )
    check("真实 HTTP 问答写入 live 事件且入口拦截被计入",
          live_ask.status_code == 200
          and build_report(mem_analytics)["overview"]["live_events"] > before_live
          and build_report(mem_analytics)["overview"]["guard_blocked"] == before_guard + 1)
    check("HTTP 响应与事件库都不回显该提示词",
          OUT_Q not in json.dumps(live_ask.json(), ensure_ascii=False)
          and OUT_Q not in mem_analytics.all_text())
    rows_ro = build_report(mem_analytics)["overview"]["event_rows"]
    client.get("/analytics/summary")
    client.get("/dashboard")
    client.post("/guard/check", json={"query": OUT_Q})
    check("只读端点不写事件（统计不被探测流量污染）",
          build_report(mem_analytics)["overview"]["event_rows"] == rows_ro)

    real_db = s.var_dir / "analytics.db"
    real_blob = ""
    if real_db.exists():
        con = sqlite3.connect(str(real_db))
        real_blob = "\n".join(
            str(v) for row in con.execute("SELECT * FROM events")
            for v in row if isinstance(v, str)
        )
        con.close()
    check("自检使用内存库，未污染真实 var/analytics.db",
          OUT_Q not in real_blob and "selfcheck-analytics-live" not in real_blob)
    check("数据中台全程零模型调用", counter["n"] == 0, f"counter={counter['n']}")

    # 12d. 一键启动：只做静态契约与本地探测，不弹浏览器、不占用服务端口
    print("\n[12d] 一键启动脚本与 serve --open")
    from campus_rules.cli import _health_ok, _service_ready, build_parser  # noqa: E402

    launcher = ROOT / "start.bat"
    launcher_text = launcher.read_text(encoding="utf-8") if launcher.exists() else ""
    check("一键启动脚本存在且调用 serve --open",
          launcher.exists()
          and "-m campus_rules serve --open" in launcher_text
          and r".venv\Scripts\python.exe" in launcher_text
          and "cd /d" in launcher_text)
    s_args = build_parser().parse_args(["serve", "--open", "--open-page", "chat"])
    check("serve 支持 --open / --open-page 且默认本机 8000",
          s_args.open is True and s_args.open_page == "chat"
          and s_args.port == 8000 and s_args.host == "127.0.0.1")
    check("服务探测只读，未启动端口返回 False",
          _service_ready("http://127.0.0.1:65534", timeout=0.2) is False)

    # 回归：8000 曾被 v0.3 遗留服务占用（有 /health、没有 /analytics/summary），
    # 旧逻辑会误判「已就绪」并把浏览器带到 404 页面。
    import http.server  # noqa: E402
    import threading  # noqa: E402

    class _LegacyHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args):  # noqa: A002
            pass

    fake = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LegacyHandler)
    threading.Thread(target=fake.serve_forever, daemon=True).start()
    fake_base = f"http://127.0.0.1:{fake.server_address[1]}"
    try:
        check("旧版服务（仅有 /health）不会被误判为就绪",
              _health_ok(fake_base, timeout=1) is True
              and _service_ready(fake_base, timeout=1) is False)
    finally:
        fake.shutdown()
        fake.server_close()

    check("一键启动断言全程零模型调用", counter["n"] == 0, f"counter={counter['n']}")

    # 12e. README 编码卫生。历史上文件末尾曾混入一段 UTF-16LE 残留（含 NUL），
    # Git 会把 Markdown 判成二进制（i/-text），GitHub 渲染即乱码。此处从字节层拦住回归。
    print("\n[12e] README 编码与仓库文本卫生")
    readme_bytes = (ROOT / "README.md").read_bytes()
    try:
        readme_text = readme_bytes.decode("utf-8")
        readme_utf8 = True
    except UnicodeDecodeError:
        readme_text, readme_utf8 = "", False
    check("README 严格 UTF-8、无 BOM、无 NUL、无替换符",
          readme_utf8
          and not readme_bytes.startswith(b"\xef\xbb\xbf")
          and "\x00" not in readme_text
          and "\ufffd" not in readme_text,
          f"{len(readme_bytes)} bytes")
    check("README 结尾无 UTF-16 残留，正文结构完整",
          readme_text.startswith("# campus-rules-agent")
          and "## 10." in readme_text
          and readme_text.rstrip().endswith("本仓库不含任何真实密钥。"))
    import subprocess as _subprocess  # noqa: E402
    try:
        _tracked = _subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-files"], text=True,
            stderr=_subprocess.DEVNULL,
        ).splitlines()
        _nul_files = [
            f for f in _tracked
            if not f.lower().endswith((".png", ".jpg", ".jpeg", ".ico", ".db", ".npy"))
            and (ROOT / f).is_file()
            and b"\x00" in (ROOT / f).read_bytes()
        ]
    except (OSError, _subprocess.SubprocessError):
        _nul_files = []
        check("仓库文本文件扫描跳过（git 不可用）", True)
    else:
        check("所有已跟踪文本文件均无 NUL 污染", not _nul_files, ",".join(_nul_files))

    # 13. 清理 + 零调用断言
    print("\n[13] 清理与零 API 调用断言")
    for sid in ("offline-smoke", "offline-cli", "selfcheck-run", "selfcheck-guard",
                "selfcheck-ask", "selfcheck-redline", "selfcheck-redline-agent",
                "selfcheck-chat", "selfcheck-guard-ask", "selfcheck-guard-agent",
                "selfcheck-guard-normal", "selfcheck-http-guard",
                "selfcheck-http-guard-agent", "selfcheck-analytics-live"):
        p.memory.clear(sid)
    check("自检会话已清理", True)
    check("★ 全程模型 API 调用 = 0", counter["n"] == 0, f"counter={counter['n']}")

    failed = [n for n, ok, _ in CHECKS if not ok]
    print("\n" + "=" * 72)
    print(f"结果：{len(CHECKS) - len(failed)}/{len(CHECKS)} 通过；模型 API 调用 {counter['n']} 次")
    if failed:
        print("失败项：" + "、".join(failed))
        return 1
    print("全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
