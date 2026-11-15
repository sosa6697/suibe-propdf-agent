"""确定性兜底策略：LLM 决策失败 / 超时 / JSON 解析失败时接管。

同时作为消融实验里「固定流水线」对照组 —— 它的动作序列与原 QAPipeline
的顺序一致，便于量化「让模型自己决定下一步」到底带来多少收益。
"""
from __future__ import annotations

from .state import AgentState, Decision

BASELINE_PLAN = ("retrieve", "rerank", "check_version", "pack_evidence", "answer")


def default_plan(state: AgentState) -> Decision:
    index = state.step
    if index >= len(BASELINE_PLAN):
        return Decision(
            thought="[规则策略] 固定流程已走完，仍未得到已验证答案，收敛为拒答",
            action="refuse",
            args={"reason": "固定流程结束，证据不足以作答"},
            reason="baseline_exhausted",
        )
    action = BASELINE_PLAN[index]
    if action == "answer" and not state.evidences:
        return Decision(
            thought="[规则策略] 无证据，跳过生成直接拒答",
            action="refuse",
            args={"reason": "检索无证据"},
            reason="no_evidence",
        )
    return Decision(
        thought=f"[规则策略] 第 {index + 1} 步，按固定顺序执行 {action}",
        action=action,
        args={},
        reason="deterministic_baseline",
    )
