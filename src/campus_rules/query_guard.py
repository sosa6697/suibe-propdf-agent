"""入口提示词过滤：与校规无关的问题在链路最前面被挡掉，**零模型调用**。

为什么需要它
------------
v0.3 的 `relevance.assess_relevance` 已经能拦住域外问题，但它位于检索之后：
域外题仍会走完「查询改写 → embedding → BM25 → RRF → 邻块扩展 → API rerank」
才被判「证据不足」。按 v0.4 实测，5 道拒答题各消耗 2 次模型调用
（embedding 1 + rerank 1），只是为了让系统得出「本来就不该问」的结论。

本模块把「域外判定」前移到 `QAPipeline.ask` / `AgentLoop.run` 的第一行：
命中即返回固定拒答文案，**不进入任何适配器、不写向量、不调 LLM**。
`relevance_gate` 保留为第二道防线，负责入口没拦住的隐晦域外题。

判定策略：保守放行（fail-open）
------------------------------
- 默认放行；只有命中「明确域外信号」才拦截。宁可多跑一次检索，也不误伤合法校规提问。
- 硬信号（外校 / 境外机构名）不可被覆盖；软信号（生活时刻、交通常识、未来预测、
  非校规任务）命中后，若同时出现「校规强信号」则取消拦截，避免把
  「图书馆借阅管理办法规定的开放时间是什么」这类合法提问误杀。
- 一个信号可以由多个正则组成，全部命中才算命中（AND），提高精确率。

隐私边界（与项目长期约定一致）
------------------------------
- 纯本地正则 / 词表，无网络、无配置、无环境变量、无开关。
- 返回值只含**本模块自己定义的规则名与固定文案**，绝不回显用户输入片段，
  因此 `guard.to_dict()` 可以安全进入 API 响应 / trace / 结果文件。
- 不采集、不埋点、不落盘任何用户提示词；后续「提示词数据采集」需求仍只在
  `QueryRequest.raw_query` / `PromptSink` / SQLite `turns` 处保留架构入口。

运行：`python -m campus_rules.query_guard` 可打印规则清单与 25 题金标的离线判定。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

GUARD_REFUSAL = (
    "无法回答与校规无关的问题。本助手仅依据上海对外经贸大学校规文件回答与校规相关的提问。"
)

_PASS_NO_SIGNAL = "未命中明确域外信号，保守放行"
_PASS_DOMAIN = "命中校规强信号，软域外信号被覆盖，保守放行"


@dataclass(frozen=True, slots=True)
class _Signal:
    """一个可解释的域外信号。exprs 中的正则全部命中才算命中（AND）。"""

    name: str
    label: str
    exprs: tuple[str, ...]
    hard: bool = False

    def matches(self, text: str) -> bool:
        # 正则在模块加载时由 _COMPILED 预编译一次；本方法只做匹配，无编译开销。
        return all(p.search(text) for p in _COMPILED[self.name])


# --------------------------------------------------------------------------- 硬信号
# 显式指向另一所学校 / 境外机构：这类问题与本校校规语料无关，优先级最高。
_HARD_SIGNALS: tuple[_Signal, ...] = (
    _Signal(
        "foreign_institution",
        "外校 / 境外机构名",
        (
            r"哈佛|斯坦福|牛津|剑桥|耶鲁|麻省理工|普林斯顿|东京大学|"
            r"北京大学|清华大学|复旦大学|上海交通大学|浙江大学|南京大学|"
            r"武汉大学|中山大学|外校|他校|其他高校|别的学校",
        ),
        hard=True,
    ),
)

# --------------------------------------------------------------------------- 软信号
# 都是「问法本身就说明答案不在校规文件里」的高精度标记。
_SOFT_SIGNALS: tuple[_Signal, ...] = (
    _Signal(
        "dining_service",
        "餐饮 / 生活服务时刻",
        (r"食堂|供餐|就餐|打烊|饭菜|餐券|窗口菜价",),
    ),
    _Signal(
        "venue_hours",
        "场所开放 / 营业时间",
        (r"阅览室|开放时间|营业时间|几点开门|几点关门|周末开放|自习室.*座位|图书馆.*几点",),
    ),
    _Signal(
        "off_campus_transit",
        "校外交通 / 城市常识",
        (r"地铁站|地铁口|出入口|公交站|公交线路|地铁.*号线|首班车|末班车",),
    ),
    _Signal(
        "future_projection",
        "未来预测 / 统计外推",
        (r"预计|预测|预估|推算|外推|保研率|就业率|升学率|录取率|考研率|考研成功率",),
    ),
    _Signal(
        "off_topic_task",
        "非校规类任务",
        (
            r"写一首诗|写篇作文|讲个笑话|今天天气|股票行情|彩票|星座运势|"
            r"推荐电影|推荐餐厅|翻译一下|帮我写代码|做一道菜|算一下.*运势",
        ),
    ),
)

# 校规强信号：出现即认为提问指向规章文本，软域外信号让位（硬信号不让位）。
# 刻意收窄到「文档类型词 + 明确规章概念」，不收「毕业 / 课程 / 学校」等泛词，
# 否则 R02「2027 年我校本科毕业生的保研率预计是多少」会被误放行。
_DOMAIN_STRONG = re.compile(
    r"办法|规定|条例|细则|章程|公约|条款|汇编|申诉|处分|学籍|评审|认定|答辩|"
    r"报销|差旅|退宿|勤工助学|违纪|校规|管理文件|评分标准|"
    r"第[一二三四五六七八九十百零〇0-9]+条"
)

_ALL_SIGNALS: tuple[_Signal, ...] = _HARD_SIGNALS + _SOFT_SIGNALS

# 模块加载时一次性编译，matches() 只做 search；IGNORECASE 兼容中英混排的外校名。
_COMPILED: dict[str, tuple[re.Pattern[str], ...]] = {
    s.name: tuple(re.compile(e, re.IGNORECASE) for e in s.exprs)
    for s in _ALL_SIGNALS
}


def signal_catalog() -> list[dict[str, Any]]:
    """规则清单：供 /guard/rules 与文档展示，不含任何用户输入。"""
    return [
        {"name": s.name, "label": s.label, "hard": s.hard, "expr_count": len(s.exprs)}
        for s in _ALL_SIGNALS
    ]


@dataclass(frozen=True, slots=True)
class GuardResult:
    """入口判定结果。所有字段均为本模块内置常量，不含用户提示词原文。"""

    blocked: bool
    reason: str
    matched_rules: tuple[str, ...] = ()
    domain_override: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocked": self.blocked,
            "reason": self.reason,
            "matched_rules": list(self.matched_rules),
            "domain_override": self.domain_override,
            "api_calls": 0,  # 恒定：本判定器不发起任何模型调用
        }


def _normalize(query: str) -> str:
    """NFKC 归一 + 去空白 + 小写，容忍全角字符与 PDF / 输入法排版空格。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", query)).lower()


def _blocked(signals: list[_Signal]) -> GuardResult:
    labels = "、".join(f"{s.name}（{s.label}）" for s in signals)
    return GuardResult(
        blocked=True,
        reason=f"入口过滤命中明确域外信号：{labels}；按红线拒答，不进入检索与生成链路",
        matched_rules=tuple(s.name for s in signals),
    )


def guard_query(query: str) -> GuardResult:
    """纯本地、零 API、可解释的入口域外判定；保守放行（fail-open）。

    - 空输入交由上层（Pydantic / CLI）校验，这里不越权报错。
    - 硬信号优先且不可被校规强信号覆盖。
    - 软信号命中但存在校规强信号时取消拦截，并留痕 domain_override。
    - 其余一律放行，交给第二道防线 `relevance_gate` 处理。
    """
    if not query or not query.strip():
        return GuardResult(blocked=False, reason="空输入交由上层校验，保守放行")

    text = _normalize(query)

    hard_hits = [s for s in _HARD_SIGNALS if s.matches(text)]
    if hard_hits:
        return _blocked(hard_hits)

    soft_hits = [s for s in _SOFT_SIGNALS if s.matches(text)]
    if not soft_hits:
        return GuardResult(blocked=False, reason=_PASS_NO_SIGNAL)

    if _DOMAIN_STRONG.search(text):
        return GuardResult(
            blocked=False,
            reason=_PASS_DOMAIN,
            matched_rules=tuple(s.name for s in soft_hits),
            domain_override=True,
        )
    return _blocked(soft_hits)


if __name__ == "__main__":  # pragma: no cover - 纯本地诊断，零模型调用
    import json
    import sys
    from pathlib import Path

    print("入口过滤规则清单（零模型调用）：")
    for item in signal_catalog():
        print(f"  [{'HARD' if item['hard'] else 'SOFT'}] {item['name']:<22} {item['label']}")

    golden = Path(__file__).resolve().parents[2] / "eval" / "golden.jsonl"
    if not golden.exists():
        print(f"\n金标题库不存在：{golden}", file=sys.stderr)
        sys.exit(1)
    cases = [json.loads(line) for line in golden.read_text(encoding="utf-8").splitlines() if line.strip()]
    blocked = [c["id"] for c in cases if guard_query(c["query"]).blocked]
    expected = [c["id"] for c in cases if c.get("expect_refusal")]
    print(f"\n{len(cases)} 题离线判定：拦截 {len(blocked)} 题 -> {blocked}")
    print(f"金标应拒答     {len(expected)} 题 -> {expected}")
    print("一致" if blocked == expected else "不一致：需复查规则")
