"""文本清洗：只用于喂给 LLM 的展示，绝不改写 Chunk.content 里的原文。

PDF 抽取带来的页码噪声（如 " 1 / 6 "）会干扰条款定位，但溯源必须保留原始文本。
"""
from __future__ import annotations

import re

_PAGE_NOISE = re.compile(r"^[ \t\u00a0]*\d+[ \t\u00a0]*/[ \t\u00a0]*\d+[ \t\u00a0]*$", re.MULTILINE)
_BLANKS = re.compile(r"[ \t\u00a0]+")
_MULTI_NL = re.compile(r"\n{3,}")
_ARTICLE = re.compile(r"第[一二三四五六七八九十百零〇0-9]+条")


def normalize_for_prompt(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = _PAGE_NOISE.sub("", text)
    text = _BLANKS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def articles_in(text: str) -> set[str]:
    """提取文本里出现的条款号，用于条款级匹配与版本冲突判断。"""
    return set(_ARTICLE.findall(text))


def query_articles(query: str) -> set[str]:
    return articles_in(query)
