"""生成后引用校验：先验规则（离线、零成本），再叠加 LLM 核查（若已配置）。

不通过时由编排层重试生成一次；仍不通过则降级为拒答，绝不把未经验证的内容交给用户。
"""
from __future__ import annotations

import json
import re

from .adapters.errors import AdapterError
from .adapters.llm import LLMClient
from .logging_conf import get_logger
from .prompts import VERIFY_SYSTEM, build_verify_user
from .schema import Evidence
from .text_utils import normalize_for_prompt

log = get_logger("verification")

_CITE = re.compile(r"\[(E\d+)\]")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_REFUSAL_MARK = "未找到"
_ARTICLE = re.compile(r"第[一二三四五六七八九十百零〇0-9]+条")


def verify_answer(
    answer: str,
    evidences: list[Evidence],
    llm: LLMClient | None = None,
    query: str = "",
) -> dict:
    issues: list[str] = []
    warnings: list[str] = []
    labels = {e.label for e in evidences}
    cited = set(_CITE.findall(answer))
    is_refusal = _REFUSAL_MARK in answer

    unknown = cited - labels
    if unknown:
        issues.append(f"答案引用了不存在的证据编号：{sorted(unknown)}")
    if not cited and not is_refusal:
        issues.append("答案没有任何 [E编号] 引用标记")

    evidence_text = "\n".join(normalize_for_prompt(e.chunk.content) for e in evidences)
    answer_wo_cite = _CITE.sub("", answer)
    numbers = _NUM.findall(answer_wo_cite)
    # 年份是文件 / 版本元数据（如「2019 年管理文件汇编」「2023 年修订」），常被版本提示
    # 带进答案，却未必出现在证据正文里。若把它算进「数值逐字校验」，C05 这类只有一个
    # 年份数字的正确答案会被误判为「数字均系臆造」而拒答。此处只降为 warning。
    year_re = re.compile(r"(?:19|20)\d{2}")
    years = sorted({n for n in numbers if year_re.fullmatch(n)})
    substantive = [n for n in numbers if not year_re.fullmatch(n)]
    missing_numbers = [n for n in substantive if n not in evidence_text]
    if substantive and len(missing_numbers) == len(substantive):
        issues.append(f"答案中的数值 {missing_numbers} 在证据原文中均未出现")
    elif missing_numbers:
        warnings.append(f"以下数值未在证据中直接命中，需人工复核：{missing_numbers}")
    if years:
        warnings.append(f"答案含年份 {years}（版本 / 文件元数据，不参与数值逐字校验）")

    answer_articles = set(_ARTICLE.findall(answer_wo_cite))
    evidence_articles = set(_ARTICLE.findall(evidence_text))
    unknown_articles = answer_articles - evidence_articles
    if unknown_articles:
        warnings.append(f"答案提到的条款号未在证据中找到：{sorted(unknown_articles)}")

    method = "rules"
    if llm is not None:
        try:
            raw = llm.chat(
                system=VERIFY_SYSTEM,
                user=build_verify_user(answer, evidences, query),
                temperature=0.0,
                timeout=60,
            )
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0 and end > start:
                payload = json.loads(raw[start : end + 1])
                if not bool(payload.get("supported", True)):
                    issues.append("LLM 核查不通过：" + "；".join(map(str, payload.get("issues", [])))[:300])
                method = "rules+llm"
        except (AdapterError, json.JSONDecodeError, ValueError) as exc:
            warnings.append(f"LLM 核查不可用，仅执行规则校验：{exc}")

    result = {
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
        "method": method,
        "citations": sorted(cited),
        "refusal": is_refusal,
    }
    log.info("引用校验：ok=%s method=%s issues=%d warnings=%d", result["ok"], method, len(issues), len(warnings))
    return result
