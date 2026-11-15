"""校规主题词典：把用户提示词在**内存里**映射成主题标签。

为什么不用 LLM 分类 / 自动分词 Top-N（用户确认「主题词典命中率」口径）：
- 主题词典口径稳定、可解释、可复现，管理者看到「纪律处分 18.5%」能直接对上业务；
- 不需要任何模型调用，也不依赖 jieba 的停用词质量；
- 只输出**内置标签名**，标签本身就是可安全落盘的元数据，不含用户输入片段。

一条输入可以命中多个主题（例如「违纪处分的申诉期限」），因此看板里
各主题占比之和可能超过 100%，口径说明里会明确标注。
"""
from __future__ import annotations

import re
import unicodedata

# 顺序即看板默认展示顺序；键是唯一允许落盘的主题标签。
TOPICS: dict[str, tuple[str, ...]] = {
    "纪律处分": ("处分", "违纪", "警告", "记过", "留校察看", "开除学籍"),
    "申诉复核": ("申诉", "复核", "不服"),
    "奖学金": ("奖学金", "国家奖学金", "励志奖学金", "学业奖学金"),
    "学籍选课": ("学籍", "选课", "学分", "第二学士学位"),
    "毕业论文": ("毕业论文", "答辩", "评分标准", "论文管理条例"),
    "资助勤工": ("勤工助学", "家庭经济困难", "困难认定", "认定办法", "岗位设置"),
    "住宿管理": ("退宿", "住宿", "宿舍", "作息", "住宿公约"),
    "财务报销": ("差旅", "报销", "出差", "经费"),
    "竞赛奖励": ("竞赛", "奖励办法", "学科学术类"),
    "章程汇编": ("章程", "管理文件汇编", "汇编"),
    "版本时效": ("版本", "修订", "核准", "最新版", "先后发布"),
}

NONE_TOPIC = "未归类"


def normalize(text: str) -> str:
    """NFKC 归一 + 去空白 + 小写，容忍全角字符与输入法排版空格。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).lower()


def classify(query: str) -> list[str]:
    """返回命中的主题标签；无命中返回 [NONE_TOPIC]。

    输入原文只在这个函数内部使用，返回后调用方即可丢弃，绝不进入事件表。
    """
    text = normalize(query)
    hits = [name for name, words in TOPICS.items()
            if any(normalize(w) in text for w in words)]
    return hits or [NONE_TOPIC]


def catalog() -> list[dict[str, object]]:
    """供 /analytics/summary 与看板展示词典口径，不含任何用户输入。"""
    return [
        {"topic": name, "keywords": list(words), "keyword_count": len(words)}
        for name, words in TOPICS.items()
    ]
