from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Sequence

import chromadb
import numpy as np
import requests
from rank_bm25 import BM25Okapi
from config import (
    AppPaths,
    BM25_WEIGHT,
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP_CHARS,
    HYBRID_TOP_K,
    MIN_CONTEXT_SCORE,
    QWEN_API_KEY,
    QWEN_BASE_URL,
    QWEN_MODEL,
    QWEN_TIMEOUT,
    RERANK_TOP_K,
    SYSTEM_PROMPT,
    TOP_K_BM25,
    TOP_K_CONTEXT,
    TOP_K_VECTOR,
    VECTOR_WEIGHT,
    ensure_dirs,
)

try:
    from docx import Document as DocxDocument  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    DocxDocument = None

try:
    from pypdf import PdfReader  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None

TITLE_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\.])\s*")
WORD_RE = re.compile(r"[\w\u4e00-\u9fff]+")
STOPWORDS = {"的", "了", "和", "是", "在", "请问", "什么", "如何", "怎么", "为什么", "吗", "呢", "吧"}
TERM_TITLE_RE = re.compile(r"^(?:\d+[\.)、]|[（(]?\d+[）)]?\s*)?(.{2,40}?)(?:[:：\-—]\s*(.*))?$")
DOCX_CHUNK_MAX_CHARS = 180
DOCX_CHUNK_OVERLAP = 30
TERM_MERGE_MIN_CHARS = 40


@dataclass
class SearchHit:
    chunk: DocumentChunk
    bm25_score: float = 0.0
    vector_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class DocumentChunk:
    chunk_id: str
    source_file: str
    title_path: str
    content: str
    heading_level: int
    order: int

    def metadata(self) -> dict:
        return asdict(self)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = WORD_RE.findall(text)
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def stable_hash_token(token: str) -> int:
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)


def split_by_title(text: str, source_file: str) -> List[DocumentChunk]:
    lines = normalize_text(text).split("\n")
    chunks: List[DocumentChunk] = []
    title_stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    order = 0

    def current_title_path() -> str:
        return " > ".join(title for _, title in title_stack) if title_stack else Path(source_file).stem

    def flush() -> None:
        nonlocal buffer, order
        content = normalize_text("\n".join(buffer))
        if content:
            order += 1
            chunks.append(
                DocumentChunk(
                    chunk_id=make_chunk_id(source_file, order, 0, content),
                    source_file=source_file,
                    title_path=current_title_path(),
                    content=content,
                    heading_level=title_stack[-1][0] if title_stack else 0,
                    order=order,
                )
            )
        buffer = []

    for line in lines:
        match = TITLE_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            while title_stack and title_stack[-1][0] >= level:
                title_stack.pop()
            title_stack.append((level, title))
        else:
            buffer.append(line)
    flush()
    return chunks


def make_chunk_id(source_file: str, order: int, part: int = 0, content: str = "") -> str:
    """生成稳定且尽量避免碰撞的 chunk ID。"""
    normalized_source = str(Path(source_file).resolve())
    payload = f"{normalized_source}:{order}:{part}:{normalize_text(content)}"
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
    return f"chunk_{digest[:16]}"


def further_chunk_long_text(content: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> List[str]:
    content = normalize_text(content)
    if len(content) <= max_chars:
        return [content]
    sentences = SENTENCE_SPLIT_RE.split(content)
    parts: List[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        candidate = (current + " " + sentence).strip() if current else sentence.strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                parts.append(current)
            current = sentence.strip()
    if current:
        parts.append(current)
    if not parts:
        step = max(1, max_chars - overlap)
        parts = [content[i : i + max_chars] for i in range(0, len(content), step)]
    return parts


def read_text_file(file_path: Path) -> str:
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return file_path.read_text(encoding="utf-8", errors="ignore")


def keep_chinese_text(text: str) -> str:
    lines = []
    for raw_line in normalize_text(text).split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        chinese_only = "".join(re.findall(r"[\u4e00-\u9fff\s，。！？；：、（）《》“”‘’—…·]+", line))
        chinese_only = re.sub(r"\s+", " ", chinese_only).strip()
        if chinese_only:
            lines.append(chinese_only)
    return "\n".join(lines)


def read_docx_file(file_path: Path) -> str:
    if DocxDocument is None:
        raise ImportError("缺少 python-docx 依赖，无法解析 Word 文档。")
    doc = DocxDocument(str(file_path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return keep_chinese_text("\n".join(paragraphs))


def chunk_title_from_text(text: str, fallback: str, index: int) -> str:
    cleaned = normalize_text(text)
    if not cleaned:
        return f"{fallback}-{index}"
    first_line = cleaned.split("\n", 1)[0].strip()
    first_line = re.sub(r"^[\W_\d]+", "", first_line)
    if len(first_line) > 24:
        first_line = first_line[:24].rstrip()
    return first_line or f"{fallback}-{index}"


def split_docx_forced_chunks(text: str, source_file: str) -> List[DocumentChunk]:
    lines = [line.strip() for line in normalize_text(text).split("\n") if line.strip()]
    if not lines:
        return []

    paragraphs: list[str] = []
    buffer: list[str] = []
    for line in lines:
        if TITLE_RE.match(line):
            if buffer:
                paragraphs.append(normalize_text("\n".join(buffer)))
                buffer = []
            paragraphs.append(line)
        else:
            buffer.append(line)
    if buffer:
        paragraphs.append(normalize_text("\n".join(buffer)))

    if not paragraphs:
        paragraphs = [normalize_text(text)]

    chunks: list[DocumentChunk] = []
    order = 0
    current_title = Path(source_file).stem

    for block in paragraphs:
        if TITLE_RE.match(block):
            current_title = TITLE_RE.match(block).group(2).strip()
            continue
        for piece in further_chunk_long_text(block, max_chars=DOCX_CHUNK_MAX_CHARS, overlap=DOCX_CHUNK_OVERLAP):
            piece = normalize_text(piece)
            if not piece:
                continue
            order += 1
            title = chunk_title_from_text(piece, current_title, order)
            chunks.append(
                DocumentChunk(
                    chunk_id=make_chunk_id(source_file, order, 0, f"{title}\n{piece}"),
                    source_file=source_file,
                    title_path=title,
                    content=piece,
                    heading_level=1,
                    order=order,
                )
            )
    return chunks


def read_pdf_file(file_path: Path) -> str:
    if PdfReader is None:
        raise ImportError("缺少 pypdf 依赖，无法解析 PDF 文档。")
    reader = PdfReader(str(file_path))
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append(text)
    return "\n".join(pages)


def load_file_content(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        return read_text_file(file_path)
    if suffix == ".docx":
        return read_docx_file(file_path)
    if suffix == ".pdf":
        return read_pdf_file(file_path)
    return ""


def load_documents(doc_dir: Path) -> List[DocumentChunk]:
    documents: List[DocumentChunk] = []
    supported_suffixes = {".md", ".markdown", ".txt", ".docx", ".pdf"}
    for file_path in sorted(doc_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in supported_suffixes:
            continue
        text = load_file_content(file_path)
        if not text.strip():
            continue
        if file_path.suffix.lower() == ".docx":
            base_chunks = split_docx_forced_chunks(text, str(file_path))
        else:
            base_chunks = split_by_title(text, str(file_path))
        if not base_chunks:
            base_chunks = [
                DocumentChunk(
                    chunk_id=make_chunk_id(str(file_path), 1, 0, text),
                    source_file=str(file_path),
                    title_path=file_path.stem,
                    content=normalize_text(text),
                    heading_level=0,
                    order=1,
                )
            ]
        for chunk in base_chunks:
            sub_chunks = further_chunk_long_text(chunk.content)
            if len(sub_chunks) == 1:
                documents.append(chunk)
            else:
                for idx, sub in enumerate(sub_chunks, start=1):
                    documents.append(
                        DocumentChunk(
                            chunk_id=make_chunk_id(chunk.source_file, chunk.order, idx, sub),
                            source_file=chunk.source_file,
                            title_path=chunk.title_path,
                            content=sub,
                            heading_level=chunk.heading_level,
                            order=chunk.order * 100 + idx,
                        )
                    )
    return documents


class EmbeddingClient:
    """轻量兼容的本地/HTTP embedding 客户端。

    优先尝试调用环境变量 EMBEDDING_API_URL 指定的服务；
    若未配置，则使用纯词袋向量作为兜底，保证项目可运行。
    """

    def __init__(self) -> None:
        self.api_url = os.getenv( "").strip()
        self.api_key = os.getenv( "").strip()
        self.dimension = int(os.getenv("EMBEDDING_DIM", "384"))

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if self.api_url:
            return self._embed_via_http(texts)
        return [self._fallback_embed(text) for text in texts]

    def _embed_via_http(self, texts: Sequence[str]) -> List[List[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"texts": list(texts)}
        resp = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["embeddings"]

    def _fallback_embed(self, text: str) -> List[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        for token in tokenize(text):
            idx = stable_hash_token(token) % self.dimension
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()


class RAGIndex:
    def __init__(self, paths: AppPaths | None = None) -> None:
        ensure_dirs()
        self.paths = paths or AppPaths()
        self.embedder = EmbeddingClient()
        self.chroma_client = chromadb.PersistentClient(path=str(self.paths.chroma_dir))
        self.collection = self.chroma_client.get_or_create_collection(name="rag_knowledge_base")
        self.chunks: List[DocumentChunk] = []
        self.bm25: BM25Okapi | None = None
        self._load_cache()

    def _cache_path(self) -> Path:
        return self.paths.cache_dir / "chunks.json"

    def _load_cache(self) -> None:
        cache_path = self._cache_path()
        if cache_path.exists():
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            self.chunks = [DocumentChunk(**item) for item in raw]
            self._build_bm25()
            self._restore_collection_from_cache()

    def _restore_collection_from_cache(self) -> None:
        if not self.chunks:
            return
        try:
            existing = self.collection.get(include=[])
            if existing.get("ids"):
                return
        except Exception:
            pass
        embeddings = self.embedder.embed([f"{c.title_path}\n{c.content}" for c in self.chunks])
        self.collection.add(
            ids=[c.chunk_id for c in self.chunks],
            documents=[c.content for c in self.chunks],
            embeddings=embeddings,
            metadatas=[c.metadata() for c in self.chunks],
        )

    def _reset_collection(self) -> None:
        try:
            self.chroma_client.delete_collection(name=self.collection.name)
        except Exception:
            pass
        self.collection = self.chroma_client.get_or_create_collection(name="rag_knowledge_base")

    def _save_cache(self) -> None:
        self._cache_path().write_text(
            json.dumps([asdict(chunk) for chunk in self.chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _build_bm25(self) -> None:
        tokenized = [tokenize(f"{c.title_path} {c.content}") for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def build(self, doc_dir: Path | None = None) -> int:
        doc_dir = doc_dir or self.paths.doc_dir
        docs = load_documents(doc_dir)
        self.chunks = docs
        self._build_bm25()
        self._reset_collection()
        if docs:
            embeddings = self.embedder.embed([f"{c.title_path}\n{c.content}" for c in docs])
            self.collection.add(
                ids=[c.chunk_id for c in docs],
                documents=[c.content for c in docs],
                embeddings=embeddings,
                metadatas=[c.metadata() for c in docs],
            )
        self._save_cache()
        return len(docs)

    def _bm25_search(self, query: str, top_k: int = TOP_K_BM25) -> list[tuple[float, DocumentChunk]]:
        if not self.bm25 or not self.chunks:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(scores, self.chunks), key=lambda x: x[0], reverse=True)
        return [(float(score), chunk) for score, chunk in ranked[:top_k] if score > 0]

    def _vector_search(self, query: str, top_k: int = TOP_K_VECTOR) -> list[tuple[float, DocumentChunk]]:
        if not self.chunks:
            return []
        embedding = self.embedder.embed([query])[0]
        result = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        scores: list[tuple[float, DocumentChunk]] = []
        ids = result.get("ids", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distance_list = result.get("distances", [[]])[0]
        for idx, _chunk_id in enumerate(ids):
            if idx >= len(metadatas):
                continue
            meta = metadatas[idx]
            if not meta:
                continue
            chunk = DocumentChunk(**meta)
            dist = float(distance_list[idx]) if idx < len(distance_list) else 0.0
            score = 1.0 / (1.0 + max(dist, 0.0))
            scores.append((score, chunk))
        return scores

    def _build_hybrid_candidates(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchHit]:
        bm25_hits = self._bm25_search(query, top_k=TOP_K_BM25)
        vec_hits = self._vector_search(query, top_k=TOP_K_VECTOR)

        merged: dict[str, SearchHit] = {}
        for score, chunk in bm25_hits:
            hit = merged.setdefault(chunk.chunk_id, SearchHit(chunk=chunk))
            hit.bm25_score = score
        for score, chunk in vec_hits:
            hit = merged.setdefault(chunk.chunk_id, SearchHit(chunk=chunk))
            hit.vector_score = score

        results: list[SearchHit] = []
        for hit in merged.values():
            hit.hybrid_score = BM25_WEIGHT * hit.bm25_score + VECTOR_WEIGHT * hit.vector_score
            if hit.hybrid_score >= MIN_CONTEXT_SCORE:
                results.append(hit)
        results.sort(key=lambda x: x.hybrid_score, reverse=True)
        return results[:top_k]

    def _rerank_hits(self, query: str, hits: list[SearchHit], top_k: int = RERANK_TOP_K) -> list[SearchHit]:
        if not hits:
            return []

        query_tokens = set(tokenize(query))
        if not query_tokens:
            query_tokens = set(tokenize(query.lower()))

        for hit in hits:
            title_tokens = set(tokenize(hit.chunk.title_path))
            content_tokens = set(tokenize(hit.chunk.content))
            overlap = len(query_tokens & (title_tokens | content_tokens))
            title_overlap = len(query_tokens & title_tokens)
            length_bonus = min(len(hit.chunk.content) / 2000.0, 1.0) * 0.05
            source_bonus = 0.12 if Path(hit.chunk.source_file).suffix.lower() in {'.md', '.markdown'} else 0.0
            hit.rerank_score = (
                hit.hybrid_score * 0.55
                + overlap * 0.20
                + title_overlap * 0.16
                + length_bonus
                + source_bonus
            )

        hits.sort(key=lambda x: x.rerank_score, reverse=True)
        return hits[:top_k]

    def hybrid_search(self, query: str, top_k: int = TOP_K_CONTEXT) -> list[dict]:
        candidates = self._build_hybrid_candidates(query, top_k=HYBRID_TOP_K)
        reranked = self._rerank_hits(query, candidates, top_k=top_k)
        return [
            {
                "chunk_id": hit.chunk.chunk_id,
                "title_path": hit.chunk.title_path,
                "source_file": hit.chunk.source_file,
                "content": hit.chunk.content,
                "score": float(hit.rerank_score),
                "hybrid_score": float(hit.hybrid_score),
                "bm25_score": float(hit.bm25_score),
                "vector_score": float(hit.vector_score),
            }
            for hit in reranked
        ]

    def answer(self, question: str) -> dict:
        contexts = self.hybrid_search(question)
        if not contexts:
            return {
                "answer": "知识库中未找到足够相关的内容，请尝试更具体的问题，或先执行知识库重建。",
                "source": "knowledge_base_miss",
                "contexts": [],
            }
        prompt_context = self._build_context(contexts)
        answer = call_qwen_api(question, prompt_context)
        return {"answer": answer, "source": "knowledge_base", "contexts": contexts}

    def _build_context(self, contexts: list[dict]) -> str:
        blocks = []
        for i, ctx in enumerate(contexts, start=1):
            blocks.append(f"[文档{i}]\n标题路径：{ctx['title_path']}\n来源：{ctx['source_file']}\n内容：{ctx['content']}")
        return "\n\n".join(blocks)


def call_qwen_api(question: str, context: str) -> str:
    if not QWEN_API_KEY:
        return (
            "当前未配置 QWEN_API_KEY，以下为基于检索上下文的离线回答。\n\n"
            f"问题：{question}\n\n"
            f"检索到的上下文：\n{context[:2000]}\n\n"
            "请在环境变量中配置千问 API Key 后启用真实大模型回答。"
        )

    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"问题：{question}\n\n可用知识库上下文：\n{context}\n\n请基于上下文回答。",
            },
        ],
        "temperature": 0.2,
    }
    base_url = QWEN_BASE_URL.rstrip('/')
    if base_url.endswith('/chat/completions'):
        url = base_url
    else:
        url = f"{base_url}/chat/completions"
    resp = requests.post(url, json=payload, headers=headers, timeout=QWEN_TIMEOUT)
    if not resp.ok:
        raise RuntimeError(f"QWEN API 请求失败: {resp.status_code} {resp.text[:1000]}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]
