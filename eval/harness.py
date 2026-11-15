"""评测框架：金标题库 + baseline/agent 对照 + 消融实验。

API 调用治理（项目硬性约定）：
- 默认 dry-run，只打印「接口 / 次数 / 用途」调用计划，**绝不发起任何模型 API 调用**；
- 真实运行必须同时给出 --run 与 --confirm-api-calls；
- 即便加了这两个参数，运行前仍须由人先审阅调用清单并明确批准。

结果文件不保存任何用户提示词原文，也不保存金标题库里的 query 文本、
答案全文或 key_facts 原文，只记录 case_id、命中布尔与指标 ——
与「本期不采集提示词数据」的边界保持一致。

v0.4：新增答案级金标 key_facts 与 fact_recall，用于解决 title_hit_rate
六组全 1.00 饱和后指标失去区分度的问题（只校验答案是否命中真实原文锚点）。

v0.5：入口过滤 query_guard 会在检索前拦掉域外问题，此类题 0 次模型调用、
不执行引用核查，因此 citation_ok 记 None 不进分母；dry-run 计划里会
直接打印「预计入口拦截 N 题」及扣除后的调用上限，便于估算省下的调用。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campus_rules.agent import AgentLoop  # noqa: E402
from campus_rules.config import Settings  # noqa: E402
from campus_rules.pipeline import QAPipeline  # noqa: E402
from campus_rules.query_guard import guard_query  # noqa: E402
from campus_rules.schema import QueryRequest  # noqa: E402

GOLDEN = ROOT / "eval" / "golden.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
ABLATIONS = ("no_vector", "no_expand", "no_rerank", "no_verify")
CATEGORIES = ("fact", "refusal", "version_conflict", "multi_turn", "cross_article")
_SCHEMA = "campus-rules-agent/eval-v1"


@dataclass
class EvalOptions:
    mode: str = "baseline"                 # baseline | agent
    decision: str = "policy"               # llm | policy（仅 agent 生效）
    ablations: tuple[str, ...] = ()
    max_steps: int = 6
    categories: tuple[str, ...] = ()
    ids: tuple[str, ...] = ()              # 只跑指定 case_id，用于诊断复测
    limit: int = 0                         # 0 = 全部
    run: bool = False                      # True 才会真实执行
    confirm_api_calls: bool = False        # 真实执行的第二道闸门
    out: Path | None = None
    golden: Path = GOLDEN


def load_cases(path: Path | None = None) -> list[dict[str, Any]]:
    target = Path(path or GOLDEN)
    cases: list[dict[str, Any]] = []
    for lineno, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        missing = {"id", "category", "query", "expected_titles", "expect_refusal"} - case.keys()
        if missing:
            raise ValueError(f"{target.name}:{lineno} 缺字段 {sorted(missing)}")
        if case["category"] not in CATEGORIES:
            raise ValueError(f"{target.name}:{lineno} 未知类别 {case['category']}")
        case.setdefault("prelude", [])
        case.setdefault("note", "")
        case.setdefault("key_facts", [])
        case.setdefault("facts_source", "")
        if not isinstance(case["key_facts"], list) or not all(
            isinstance(f, str) and f for f in case["key_facts"]
        ):
            raise ValueError(f"{target.name}:{lineno} key_facts 必须是非空字符串数组")
        if case["expect_refusal"] and case["key_facts"]:
            raise ValueError(f"{target.name}:{lineno} 拒答题不应配置 key_facts")
        cases.append(case)
    if not cases:
        raise ValueError(f"金标题库为空：{target}")
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("金标题库存在重复 id")
    return cases


def _validate(options: EvalOptions) -> None:
    if options.mode not in {"baseline", "agent"}:
        raise ValueError("--mode 只能是 baseline 或 agent")
    if options.decision not in {"llm", "policy"}:
        raise ValueError("--decision 只能是 llm 或 policy")
    unknown = set(options.ablations) - set(ABLATIONS)
    if unknown:
        raise ValueError(f"未知消融开关 {sorted(unknown)}；可用 {list(ABLATIONS)}")
    unknown_cat = set(options.categories) - set(CATEGORIES)
    if unknown_cat:
        raise ValueError(f"未知类别 {sorted(unknown_cat)}；可用 {list(CATEGORIES)}")


def _select_cases(cases: list[dict[str, Any]], options: EvalOptions) -> list[dict[str, Any]]:
    picked = [
        c for c in cases
        if (not options.categories or c["category"] in options.categories)
        and (not options.ids or c["id"] in options.ids)
    ]
    return picked[: options.limit] if options.limit > 0 else picked


def _call_plan(options: EvalOptions, s: Settings) -> dict[str, Any]:
    """按配置与消融开关估算单题调用上限（保守估计，含重试）。"""
    vector = int(
        s.enable_vector_recall and s.embedding_ready and "no_vector" not in options.ablations
    )
    rerank = int(s.rerank_ready and "no_rerank" not in options.ablations)
    rewrite = int(s.llm_ready and options.mode == "baseline")  # baseline 多轮改写
    decision = 0 if options.mode == "baseline" or options.decision == "policy" else options.max_steps
    # baseline 与 agent 现在都支持「核查不通过 → 带意见重生成」，
    # 生成与核查各按 retry 次估算，属于保守上限（实际只有核查不过才用满）。
    retry = s.max_generation_retry + 1
    generate = int(s.llm_ready) * retry
    verify = int(s.llm_ready and "no_verify" not in options.ablations) * retry
    parts = {
        "LLM 决策": decision,
        "embedding（查询向量）": vector,
        "rerank（候选精排）": rerank,
        "LLM 查询改写": rewrite,
        "LLM 答案生成": generate,
        "LLM 引用核查": verify,
    }
    return {"parts": parts, "per_case_upper": sum(parts.values())}


def print_plan(options: EvalOptions, cases: list[dict[str, Any]], s: Settings) -> None:
    plan = _call_plan(options, s)
    counts = {c: sum(1 for x in cases if x["category"] == c) for c in CATEGORIES}

    print("[DRY-RUN] 零 API 调用：以下只是调用计划，本命令不会发起任何请求")
    print(f"模式={options.mode} | 决策={options.decision} | "
          f"消融={','.join(options.ablations) or '无'} | max_steps={options.max_steps}")
    review = sum(1 for c in cases if c.get("needs_review"))
    with_facts = sum(1 for c in cases if c.get("key_facts"))
    print(f"金标题库：{len(cases)} 题（needs_review={review}；{with_facts} 题带答案级关键事实锚点，"
          f"{len(cases) - with_facts} 题拒答类不参与事实召回）  "
          + " ".join(f"{k}={v}" for k, v in counts.items()))
    print("\n单题模型调用上限估算（含生成重试）：")
    for k, v in plan["parts"].items():
        print(f"  {k:<22} {v:>3} 次")
    print(f"  {'合计':<22} {plan['per_case_upper']:>3} 次/题 × {len(cases)} 题 "
          f"= 最多 {plan['per_case_upper'] * len(cases)} 次")
    blocked_ids = [c["id"] for c in cases if guard_query(c["query"]).blocked]
    effective = plan["per_case_upper"] * (len(cases) - len(blocked_ids))
    print("\n入口过滤（query_guard，纯本地零调用）：")
    print(f"  预计拦截 {len(blocked_ids)} 题（0 次模型调用）："
          f"{','.join(blocked_ids) or '无'}")
    print(f"  扣除后实际上限 ≈ {effective} 次，较原上限省 "
          f"{plan['per_case_upper'] * len(blocked_ids)} 次")
    print("\n涉及的接口（只列路径，不含任何密钥）：")
    if plan["parts"]["embedding（查询向量）"]:
        print(f"  embedding : {s.embedding_style} / "
              f"{s.embedding_path if s.embedding_style == 'openai' else '/services/embeddings/text-embedding/text-embedding'}"
              "  用途：把查询编码成向量")
    if plan["parts"]["rerank（候选精排）"]:
        print(f"  rerank    : {s.rerank_style} / {s.rerank_path or '(完整 URL)'}"
              "  用途：对候选原文做精排")
    if any(v for k, v in plan["parts"].items() if k.startswith("LLM")):
        print(f"  llm       : {s.llm_path}  用途：决策 / 改写 / 生成 / 引用核查")
    print("\n用例清单（只列 id 与类别，不输出 query 原文）：")
    for c in cases:
        print(f"  {c['id']:<4} {c['category']:<17} "
              f"expect_refusal={str(c['expect_refusal']):<5} "
              f"titles={len(c['expected_titles'])} key_facts={len(c.get('key_facts', []))}")
    print("\n⚠️ 真实运行会消耗模型调用：须先审阅以上清单并明确批准，"
          "再同时加 --run --confirm-api-calls。")


def _wrap_adapters(pipeline: QAPipeline) -> dict[str, int]:
    """在适配器实例上挂调用计数器，作为两种模式统一的 api_calls 口径。"""
    counter = {"n": 0}
    for obj, name in (
        (pipeline.embedding, "embed"),
        (pipeline.rerank, "rerank"),
        (pipeline.llm, "chat"),
    ):
        original = getattr(obj, name)

        def make(orig):
            def inner(*args, **kwargs):
                counter["n"] += 1
                return orig(*args, **kwargs)
            return inner

        setattr(obj, name, make(original))
    return counter


def _citation_ok(verification: dict[str, Any] | None) -> bool | None:
    """引用核查是否通过；没跑核查器时返回 None（不可评，不进分母）。

    早期实现给「未执行核查」的题默认 True，导致 no_verify 消融组的
    citation_pass_rate 恒为 1.00 的假象。这里显式区分「通过」与「没测」。
    """
    if not verification:
        return None
    method = verification.get("method", "skipped")
    if method in ("ablated", "skipped"):
        return None
    return bool(verification.get("ok", False))


def _norm_for_match(text: str) -> str:
    """NFKC 归一 + 去空白 + 小写，容忍全角数字／百分号与 PDF 抽取的排版空格。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).lower()


def _fact_hits(answer: str, key_facts: list[str]) -> list[bool]:
    """答案是否逐条命中答案级金标。只返回布尔，绝不把答案文本写进结果文件。"""
    norm_answer = _norm_for_match(answer)
    return [_norm_for_match(f) in norm_answer for f in key_facts]


def _run_case(
    pipeline: QAPipeline,
    loop: AgentLoop | None,
    counter: dict[str, int],
    case: dict[str, Any],
    options: EvalOptions,
) -> dict[str, Any]:
    session = f"eval::{case['id']}::{options.mode}::{options.decision}"
    pipeline.memory.clear(session)
    for turn in case.get("prelude", []):
        pipeline.memory.append(session, turn["role"], turn["content"])

    before = counter["n"]
    t0 = time.perf_counter()
    if options.mode == "baseline":
        result = pipeline.ask(QueryRequest(raw_query=case["query"], session_id=session))
        citations = [e.chunk.citation for e in result.evidences]
        best_score = max((e.score for e in result.evidences), default=0.0)
        refused, verification = result.refused, result.verification
        conflicts, steps = result.version_conflicts, len(result.trace)
        stop, degraded = "baseline", result.degraded
        reasons = list(result.degrade_reasons)
        relevance = dict(result.relevance)
        answer_text = result.answer
        steps_metric = "pipeline_trace_steps"
        guard_blocked = bool(result.guard_blocked)
        guard_rules = list(result.guard.get("matched_rules", []))
    else:
        assert loop is not None
        state = loop.run(QueryRequest(raw_query=case["query"], session_id=session))
        citations = [e.chunk.citation for e in state.evidences]
        best_score = max((e.score for e in state.evidences), default=0.0)
        refused, verification = state.refused, state.verification
        conflicts, steps = state.conflicts, state.step
        stop = state.stop_reason.value if state.stop_reason else "-"
        degraded = state.degraded
        reasons = list(state.degrade_reasons)
        relevance = dict(state.relevance)
        answer_text = state.answer
        steps_metric = "agent_tool_calls"
        guard_blocked = bool(state.guard_blocked)
        guard_rules = list(state.guard.get("matched_rules", []))
    ms = round((time.perf_counter() - t0) * 1000, 1)
    api_calls = counter["n"] - before
    pipeline.memory.clear(session)  # 评测会话不留存

    titles = case.get("expected_titles", [])
    expect_refusal = bool(case.get("expect_refusal"))
    v_method = verification.get("method", "skipped") if verification else "skipped"
    # 入口拦截题根本没跑核查器，citation_ok 必须记 None，不能算作「通过」
    citation_ok = None if guard_blocked else _citation_ok(verification)

    # 答案级金标：只算命中布尔，不把答案全文 / key_facts 原文写进结果文件。
    # 拒答题（key_facts 为空）记 None 不进分母；正常题被拒答记 0 分，
    # 这样 title_hit 饱和（六组全 1.00）时仍能靠事实召回区分链路优劣。
    key_facts = case.get("key_facts", [])
    if key_facts and not expect_refusal:
        hits = _fact_hits(answer_text, key_facts)
        fact_recall = round(sum(hits) / len(hits), 4)
    else:
        hits, fact_recall = [], None
    return {
        "id": case["id"],
        "category": case["category"],
        "title_hit": (not titles) or any(t in c for t in titles for c in citations),
        "scored_for_title": bool(titles),
        "refusal_correct": refused == expect_refusal,
        "citation_ok": citation_ok,
        "fact_hits": hits,
        "fact_total": len(key_facts),
        "fact_recall": fact_recall,
        "guard_blocked": guard_blocked,
        "guard_rules": guard_rules,
        "verification_method": v_method,
        "verification_issues": list(verification.get("issues", []))[:3] if verification else [],
        "conflict_detected": bool(conflicts),
        "steps": steps,
        "steps_metric": steps_metric,
        "api_calls": api_calls,
        "ms": ms,
        "degraded": degraded,
        "degrade_reasons": reasons,
        "relevance": relevance,
        "best_evidence_score": round(best_score, 6),
        "stop_reason": stop,
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(rows, key):
        # None = 本题不适用该指标（例如 no_verify 组没跑核查器），不进分母
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(bool(v) for v in vals) / len(vals), 4) if vals else None

    title_rows = [r for r in records if r["scored_for_title"]]
    fact_vals = [r["fact_recall"] for r in records if r.get("fact_recall") is not None]
    def fact_rate(rows: list[float]) -> float | None:
        return round(sum(rows) / len(rows), 4) if rows else None
    totals = {
        "cases": len(records),
        "title_hit_rate": rate(title_rows, "title_hit"),
        "fact_recall_rate": fact_rate(fact_vals),
        "fact_scored_cases": len(fact_vals),
        "refusal_accuracy": rate(records, "refusal_correct"),
        "citation_pass_rate": rate(records, "citation_ok"),
        "citation_scored_cases": sum(
            1 for r in records if r.get("citation_ok") is not None
        ),
        "guard_blocked_cases": sum(1 for r in records if r.get("guard_blocked")),
        "guard_zero_call_cases": sum(
            1 for r in records if r.get("guard_blocked") and r["api_calls"] == 0
        ),
        "conflict_detect_rate": rate(
            [r for r in records if r["category"] == "version_conflict"], "conflict_detected"
        ),
        "avg_steps": round(sum(r["steps"] for r in records) / len(records), 2),
        "avg_api_calls": round(sum(r["api_calls"] for r in records) / len(records), 2),
        "total_api_calls": sum(r["api_calls"] for r in records),
        "degraded_rate": rate(records, "degraded"),
    }
    by_category = {}
    for cat in CATEGORIES:
        rows = [r for r in records if r["category"] == cat]
        if not rows:
            continue
        cat_facts = [r["fact_recall"] for r in rows if r.get("fact_recall") is not None]
        by_category[cat] = {
            "cases": len(rows),
            "title_hit_rate": rate([r for r in rows if r["scored_for_title"]], "title_hit"),
            "fact_recall_rate": fact_rate(cat_facts),
            "refusal_accuracy": rate(rows, "refusal_correct"),
            "avg_steps": round(sum(r["steps"] for r in rows) / len(rows), 2),
            "avg_api_calls": round(sum(r["api_calls"] for r in rows) / len(rows), 2),
        }
    return {"totals": totals, "by_category": by_category}


def _print_report(report: dict[str, Any]) -> None:
    t = report["totals"]
    print("\n评测结果")
    print(f"  用例数              {t['cases']}")
    print(f"  金标题命中率        {t['title_hit_rate']}")
    fact = t.get("fact_recall_rate")
    print(
        f"  答案事实召回率      "
        f"{fact if fact is not None else 'N/A（无答案级金标）'}"
        f"   可评 {t.get('fact_scored_cases', 0)}/{t['cases']} 题"
    )
    print(f"  拒答判定准确率      {t['refusal_accuracy']}")
    citation = t["citation_pass_rate"]
    print(
        f"  引用校验通过率      "
        f"{citation if citation is not None else 'N/A（未执行核查）'}"
        f"   可评 {t['citation_scored_cases']}/{t['cases']} 题"
    )
    print(f"  版本冲突检出率      {t['conflict_detect_rate']}")
    print(f"  平均步数*           {t['avg_steps']}")
    print("    * baseline=流水线 trace 步骤数 / agent=工具调用次数，口径不同不可直接比较")
    print(f"  平均/总模型调用数   {t['avg_api_calls']} / {t['total_api_calls']}")
    print(f"  入口拦截 / 零调用   {t.get('guard_blocked_cases', 0)} / "
          f"{t.get('guard_zero_call_cases', 0)} 题（query_guard，v0.5）")
    print(f"  降级比例            {t['degraded_rate']}")
    print("\n分类型：")
    for cat, v in report["by_category"].items():
        print(f"  {cat:<17} n={v['cases']:<3} title={v['title_hit_rate']} "
              f"fact={v.get('fact_recall_rate')} "
              f"refusal={v['refusal_accuracy']} steps={v['avg_steps']}")


def run(options: EvalOptions) -> int:
    _validate(options)
    cases = _select_cases(load_cases(options.golden), options)
    if not cases:
        raise ValueError("筛选后无用例，请检查 --categories / --ids / --limit")
    s = Settings.load()

    if not options.run:
        print_plan(options, cases, s)
        return 0

    if not options.confirm_api_calls:
        print("拒绝执行：真实运行会发起模型 API 调用。", file=sys.stderr)
        print("请先审阅 `python -m eval.harness ...` 的 dry-run 调用清单并获得批准，"
              "然后同时加 --run --confirm-api-calls。", file=sys.stderr)
        return 2

    print_plan(options, cases, s)
    print("\n[REAL RUN] 已声明获得调用授权，开始执行…", flush=True)

    pipeline = QAPipeline(
        s,
        ablations=options.ablations if options.mode == "baseline" else (),
    )
    counter = _wrap_adapters(pipeline)
    loop = (
        AgentLoop(pipeline, decision_mode=options.decision,
                  max_steps=options.max_steps, ablations=options.ablations)
        if options.mode == "agent" else None
    )
    records = [
        _run_case(pipeline, loop, counter, case, options)
        for case in cases
    ]
    report = _aggregate(records)
    _print_report(report)

    out = options.out or RESULTS_DIR / (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
        f"{options.mode}_{options.decision}.json"
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "schema": _SCHEMA,
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "options": {**asdict(options), "out": str(out), "golden": str(options.golden)},
                "config": s.safe_summary(),
                "metric_notes": {
                    "title_hit": "金标题库只校验检索来源文档，不校验答案全文；refusal 类不参与统计",
                    "fact_recall": (
                        "答案级金标：逐条核对自只读 meta.jsonl 的关键事实锚点（条款号/类别词/数值/年份），"
                        "只存命中布尔与比例，不存答案全文或锚点原文；拒答题记 None 不进分母，"
                        "正常题被拒答记 0 分，用于解决 title_hit_rate 饱和后指标失去区分度的问题"
                    ),
                    "steps": "baseline=流水线 trace 步骤数，agent=工具调用次数，口径不同不可直接比较",
                    "degraded": "仅统计能力缺失 / 调用失败 / 护栏触发；policy 决策模式、消融开关与红线主动拒答均不计入",
                    "citation_pass_rate": (
                        "只统计真正执行了引用核查的题（citation_scored_cases 为分母）；"
                        "no_verify 消融组、红线拒答题与 v0.5 入口拦截题记 None 不进分母，"
                        "因此不再出现假的 1.00"
                    ),
                    "guard_blocked_cases": (
                        "v0.5 入口过滤在检索前判定为域外问题的题数；这些题 0 次模型调用、"
                        "无证据、无引用核查，refusal_correct 仍按 expect_refusal 判定。"
                        "guard_rules 只记录命中的内置规则名，不存用户提示词原文"
                    ),
                    "conflict_detect_rate": "按去日期前缀的标题分组，括号年份已归一化，不再视为不同主题",
                },
                **report,
                "cases": records,
            },
            ensure_ascii=False, indent=2, default=str,
        ),
        encoding="utf-8",
    )
    print(f"\n结果已写入：{out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="campus-rules eval",
        description="校规问答 Agent 评测 / 消融（默认 dry-run，零 API 调用）",
    )
    p.add_argument("--mode", choices=("baseline", "agent"), default="baseline")
    p.add_argument("--decision", choices=("llm", "policy"), default="policy")
    p.add_argument("--ablate", default="", help="逗号分隔：no_vector,no_expand,no_rerank,no_verify")
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--categories", default="", help="逗号分隔，只跑指定类别")
    p.add_argument("--ids", default="", help="逗号分隔的 case_id，只跑指定题目（诊断用）")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 题，0=全部")
    p.add_argument("--run", action="store_true", help="真实执行（会发起模型 API 调用）")
    p.add_argument("--confirm-api-calls", action="store_true", help="确认调用清单已获批准")
    p.add_argument("--out", default="", help="结果文件路径")
    p.add_argument("--golden", default="", help="自定义金标题库路径")
    args = p.parse_args(argv)

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
        golden=Path(args.golden) if args.golden else GOLDEN,
    )
    try:
        return run(options)
    except (ValueError, FileNotFoundError) as exc:
        print(f"评测配置错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
