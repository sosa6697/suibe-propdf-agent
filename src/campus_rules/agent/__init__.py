"""手写 Agent 编排层：状态机 + 工具注册 + 决策循环 + 追踪。

刻意不依赖 LangGraph / LangChain：状态、终止条件、重复动作护栏、
决策失败回退全部显式可见，便于审计与讲解。
"""
from .loop import AgentLoop
from .state import AgentState, Decision, StopReason, ToolCall
from .tools import ToolRegistry

__all__ = ["AgentLoop", "AgentState", "Decision", "StopReason", "ToolCall", "ToolRegistry"]

