"""适配器统一异常类型。"""


class AdapterError(RuntimeError):
    """外部模型调用失败（未配置、鉴权失败、超时、响应格式异常）。"""

    def __init__(self, component: str, reason: str):
        self.component = component
        self.reason = reason
        super().__init__(f"[{component}] {reason}")
