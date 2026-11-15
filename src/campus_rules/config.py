"""集中配置：唯一读取环境变量的地方，其他模块不得直接 os.getenv。

设计原则：
- 真实密钥只存在于项目根 .env（已 gitignore），代码与日志不输出任何密钥值；
- 三项外部能力（embedding / llm / rerank）各自独立可配，缺失时明确降级，不静默跳过；
- 数据目录只读引用，程序不向其中写入任何文件。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _str(key: str, default: str = "") -> str | None:
    value = os.getenv(key, default).strip()
    return value or None


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)).strip())
    except ValueError:
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)).strip())
    except ValueError:
        return default


def _bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _path(key: str, default: str) -> str:
    """端点路径：未设置用默认后缀；显式留空表示 URL 已是完整端点，不再拼接。"""
    raw = os.getenv(key)
    return default if raw is None else raw.strip()


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    data_dir: Path
    var_dir: Path

    llm_api_key: str | None
    llm_api_url: str | None
    llm_model: str | None
    llm_path: str
    llm_max_tokens: int

    embedding_api_key: str | None
    embedding_api_url: str | None
    embedding_model: str | None
    embedding_path: str
    embedding_style: str

    rerank_api_url: str | None
    rerank_api_key: str | None
    rerank_model: str | None
    rerank_path: str
    rerank_style: str
    rerank_top_n: int
    rerank_timeout: int

    top_k_vector: int
    top_k_bm25: int
    enable_vector_recall: bool
    rrf_k: int
    final_evidence: int
    min_evidence_score: float
    min_relevance_score: float
    min_semantic_score: float
    neighbor_window: int
    max_per_doc: int
    memory_turns: int
    max_generation_retry: int
    log_level: str

    @classmethod
    def load(cls, env_file: Path | None = None) -> "Settings":
        path = env_file or (PROJECT_ROOT / ".env")
        if path.exists():
            load_dotenv(dotenv_path=str(path), override=False)

        data_dir = Path(
            _str("DATA_DIR") or str(PROJECT_ROOT.parent)
        ).expanduser()
        var_dir = Path(_str("VAR_DIR") or str(PROJECT_ROOT / "var")).expanduser()
        var_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            project_root=PROJECT_ROOT,
            data_dir=data_dir,
            var_dir=var_dir,
            llm_api_key=_str("BASE_LLM_API_KEY"),
            llm_api_url=_str("BASE_LLM_API_URL"),
            llm_model=_str("BASE_LLM_API_MODEL"),
            llm_path=_path("BASE_LLM_API_PATH", "/chat/completions"),
            llm_max_tokens=_int("LLM_MAX_TOKENS", 2400),
            embedding_api_key=_str("BASE_EMBEDDING_API_KEY"),
            embedding_api_url=_str("BASE_EMBEDDING_API_URL"),
            embedding_model=_str("BASE_EMBEDDING_API_MODEL"),
            embedding_path=_path("BASE_EMBEDDING_API_PATH", "/embeddings"),
            embedding_style=(_str("EMBEDDING_STYLE") or "openai").lower(),
            rerank_api_url=_str("RERANK_API_URL"),
            rerank_api_key=_str("RERANK_API_KEY"),
            rerank_model=_str("RERANK_API_MODEL"),
            rerank_path=_path("RERANK_API_PATH", "/rerank"),
            rerank_style=(_str("RERANK_STYLE") or "cohere").lower(),
            rerank_top_n=_int("RERANK_TOP_N", 8),
            rerank_timeout=_int("RERANK_TIMEOUT", 20),
            top_k_vector=_int("TOP_K_VECTOR", 50),
            top_k_bm25=_int("TOP_K_BM25", 50),
            enable_vector_recall=_bool("ENABLE_VECTOR_RECALL", True),
            rrf_k=_int("RRF_K", 60),
            final_evidence=_int("FINAL_EVIDENCE", 5),
            min_evidence_score=_float("MIN_EVIDENCE_SCORE", 0.0),
            min_relevance_score=_float("MIN_RELEVANCE_SCORE", 0.30),
            min_semantic_score=_float("MIN_SEMANTIC_SCORE", 0.80),
            neighbor_window=_int("NEIGHBOR_WINDOW", 1),
            max_per_doc=_int("MAX_PER_DOC", 2),
            memory_turns=_int("MEMORY_TURNS", 6),
            max_generation_retry=_int("MAX_GENERATION_RETRY", 1),
            log_level=_str("LOG_LEVEL") or "INFO",
        )

    # ---------- 能力探测：缺失即降级，绝不假装已执行 ----------
    @property
    def embedding_ready(self) -> bool:
        return all((self.embedding_api_key, self.embedding_api_url, self.embedding_model))

    @property
    def llm_ready(self) -> bool:
        return all((self.llm_api_key, self.llm_api_url, self.llm_model))

    @property
    def rerank_ready(self) -> bool:
        return all((self.rerank_api_url, self.rerank_api_key, self.rerank_model))

    def degrade_reasons(self) -> list[str]:
        reasons: list[str] = []
        if not self.enable_vector_recall:
            reasons.append("向量召回已按配置关闭（ENABLE_VECTOR_RECALL=false），仅走 BM25")
        elif not self.embedding_ready:
            reasons.append("embedding 未配置：查询向量不可用，向量召回关闭，仅走 BM25")
        if not self.rerank_ready:
            reasons.append("rerank 未配置：降级为规则重排（标题/关键词/条款/时效/去重）")
        if not self.llm_ready:
            reasons.append("LLM 未配置：无法生成答案，仅返回检索证据")
        return reasons

    def safe_summary(self) -> dict[str, str]:
        """可打印的能力摘要，不含任何密钥值。"""
        return {
            "data_dir": str(self.data_dir),
            "var_dir": str(self.var_dir),
            "embedding": f"ready={self.embedding_ready}, model={self.embedding_model or '-'}, style={self.embedding_style}",
            "llm": f"ready={self.llm_ready}, model={self.llm_model or '-'}",
            "rerank": (
                f"ready={self.rerank_ready}, style={self.rerank_style}, "
                f"path={self.rerank_path!r}, model={self.rerank_model or '-'}"
            ),
            "vector_recall": f"enabled={self.enable_vector_recall}, bm25_top_k={self.top_k_bm25}",
            "relevance_gate": (
                f"lexical>={self.min_relevance_score} (主判) | "
                f"semantic>={self.min_semantic_score} (仅补位, 仅 API rerank 可用)"
            ),
        }
