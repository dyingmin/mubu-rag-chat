from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma_db"
CACHE_DIR = DATA_DIR / "cache"
KB_DIR = BASE_DIR / "知识库"
DEFAULT_DOC_DIR = Path(os.getenv("DOC_DIR", str(KB_DIR if KB_DIR.exists() else BASE_DIR / "知识库")))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "1200"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "120"))
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", "8"))
TOP_K_VECTOR = int(os.getenv("TOP_K_VECTOR", "8"))
HYBRID_TOP_K = int(os.getenv("HYBRID_TOP_K", "20"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
TOP_K_CONTEXT = int(os.getenv("TOP_K_CONTEXT", str(RERANK_TOP_K)))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.45"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.55"))
MIN_CONTEXT_SCORE = float(os.getenv("MIN_CONTEXT_SCORE", "0.15"))

QWEN_API_KEY = os.getenv("QWEN_API_KEY", "").strip()
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").strip().strip('"').strip("'")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus").strip().strip('"').strip("'")
QWEN_TIMEOUT = int(os.getenv("QWEN_TIMEOUT", "90"))

SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。
要求：
1. 只能根据给定的知识库上下文回答。
2. 如果上下文不足，明确说明知识库未覆盖，不要编造。
3. 尽量输出结构化、清晰、简洁的答案。
4. 回答中尽量引用文档标题路径作为依据。
"""


@dataclass(frozen=True)
class AppPaths:
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    raw_dir: Path = RAW_DIR
    chroma_dir: Path = CHROMA_DIR
    cache_dir: Path = CACHE_DIR
    doc_dir: Path = DEFAULT_DOC_DIR


def ensure_dirs() -> None:
    for path in (DATA_DIR, RAW_DIR, CHROMA_DIR, CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)
