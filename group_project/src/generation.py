"""
Generation + Citation — src/generation.py
Person 2: Mã Vĩnh Lộc

Responsibility: LLM call + answer với citation.

Dependencies from teammates:
  - Person 1 (retrieval.py): search(query, top_k) → list[dict]
  - Person 3 (memory.py):    history: list[dict] (injected, not managed here)
  - Person 4 (app.py):       calls generate() / generate_stream()

Public API:
    generate(query, chunks, history) → GenerationResult
    generate_stream(query, chunks, history) → Iterator[str]
    reorder_for_llm(chunks) → list[dict]
    format_context(chunks) → str
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterator

from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

GROQ_MODEL = "llama-3.3-70b-versatile"
OPENAI_MODEL = "gpt-4o-mini"  # fallback — đổi thành gpt-4o nếu cần

TOP_K = 5
TOP_P = 0.9       # nucleus sampling: đủ diverse, không quá random
TEMPERATURE = 0.3  # RAG cần factual → thấp
MAX_TOKENS = 2048

SYSTEM_PROMPT = """Bạn là trợ lý chuyên về pháp luật ma tuý Việt Nam và các tin tức liên quan.

Quy tắc trả lời:
1. Chỉ sử dụng thông tin từ context được cung cấp.
2. Mỗi luận điểm phải có citation ngay sau, ví dụ:
   - [Luật Phòng chống ma tuý 2021, Điều 3]
   - [VnExpress, 2024]
   - [Thanh Niên, 2023]
3. Nếu thông tin không có trong context → trả lời:
   "Tôi không thể xác minh thông tin này từ nguồn hiện có."
4. Trả lời bằng tiếng Việt, rõ ràng, có cấu trúc đoạn văn.
5. Nếu có lịch sử hội thoại, sử dụng để hiểu ngữ cảnh câu hỏi tiếp theo."""


# =============================================================================
# RESULT TYPE
# =============================================================================

@dataclass
class GenerationResult:
    answer: str
    sources: list[dict]
    retrieval_source: str  # 'hybrid' | 'pageindex' | 'none'
    query: str
    reranked_chunks: list[dict] = field(default_factory=list)

    @property
    def unique_sources(self) -> list[str]:
        """Deduplicated source names — for UI display."""
        seen: set[str] = set()
        out: list[str] = []
        for chunk in self.sources:
            src = chunk.get("metadata", {}).get("source", "Unknown")
            if src not in seen:
                seen.add(src)
                out.append(src)
        return out

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "retrieval_source": self.retrieval_source,
            "unique_sources": self.unique_sources,
            "query": self.query,
        }


# =============================================================================
# DOCUMENT REORDERING — tránh "lost in the middle"
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    LLM nhớ tốt đầu và cuối prompt → đặt chunks quan trọng nhất ở đó.

    Input (best → worst): [0, 1, 2, 3, 4]
    Output:               [0, 2, 4, 3, 1]
      - Even indices (0, 2, 4) → đầu (ascending)
      - Odd indices  (1, 3)    → cuối (reversed, nên 1 là cuối cùng)

    Ref: Liu et al. (2023), "Lost in the Middle"
    """
    if len(chunks) <= 2:
        return chunks
    even = [chunks[i] for i in range(0, len(chunks), 2)]
    odd = [chunks[i] for i in range(1, len(chunks), 2)]
    return even + odd[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context block cho prompt.
    Mỗi chunk được label rõ để LLM biết cite từ đâu.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", f"Source {i}")
        doc_type = meta.get("type", "unknown")
        content = chunk.get("content", "")
        parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n{content}"
        )
    return "\n\n---\n\n".join(parts)


# =============================================================================
# LLM CALLS
# =============================================================================

def _call_groq(messages: list[dict]) -> str | None:
    """Gọi Groq API. Trả về None nếu không có key hoặc lỗi."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[Groq] {e}")
        return None


def _stream_groq(messages: list[dict]) -> Iterator[str]:
    """Stream Groq tokens. Yields empty nếu lỗi."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        print(f"[Groq stream] {e}")


def _call_openai(messages: list[dict]) -> str | None:
    """Fallback: OpenAI API. Đặt OPENAI_API_KEY trong .env để dùng."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[OpenAI] {e}")
        return None


def _stream_openai(messages: list[dict]) -> Iterator[str]:
    """Stream OpenAI tokens."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception as e:
        print(f"[OpenAI stream] {e}")


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

def _build_messages(
    query: str,
    context: str,
    history: list[dict] | None = None,
) -> list[dict]:
    """
    Build messages list: [system] + history + [user with context+query].

    history format (from memory.py):
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    user_content = f"Context:\n{context}\n\n---\n\nCâu hỏi: {query}"
    messages.append({"role": "user", "content": user_content})
    return messages


def generate(
    query: str,
    chunks: list[dict],
    history: list[dict] | None = None,
) -> GenerationResult:
    """
    Generate answer có citation từ retrieved chunks.

    Args:
        query:   Câu hỏi của user.
        chunks:  Kết quả từ retrieval.py — list[{'content', 'score', 'metadata', 'source'}].
        history: Conversation history từ memory.py (optional).

    Returns:
        GenerationResult với .answer, .sources, .unique_sources
    """
    if not chunks:
        return GenerationResult(
            answer="Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            sources=[],
            retrieval_source="none",
            query=query,
        )

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    messages = _build_messages(query, context, history)
    retrieval_source = chunks[0].get("source", "hybrid")

    answer = _call_groq(messages) or _call_openai(messages)
    if answer is None:
        answer = (
            "GROQ_API_KEY chưa được cấu hình. "
            "Vui lòng thêm vào file .env (hoặc thêm OPENAI_API_KEY để dùng fallback)."
        )

    return GenerationResult(
        answer=answer,
        sources=chunks,
        retrieval_source=retrieval_source,
        query=query,
        reranked_chunks=reordered,
    )


def generate_stream(
    query: str,
    chunks: list[dict],
    history: list[dict] | None = None,
) -> Iterator[str]:
    """
    Stream answer token by token — dùng trong Streamlit/Chainlit.

    Usage (Streamlit):
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full = ""
            for token in generate_stream(query, chunks, history):
                full += token
                placeholder.markdown(full + "▌")
            placeholder.markdown(full)

    Usage (Chainlit):
        msg = cl.Message(content="")
        async for token in generate_stream(query, chunks, history):
            await msg.stream_token(token)

    Yields: str (text tokens)
    After last yield: full answer has been generated
    """
    if not chunks:
        yield "Tôi không thể xác minh thông tin này từ nguồn hiện có."
        return

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    messages = _build_messages(query, context, history)

    has_output = False
    for token in _stream_groq(messages):
        has_output = True
        yield token

    if not has_output:
        for token in _stream_openai(messages):
            has_output = True
            yield token

    if not has_output:
        yield "GROQ_API_KEY chưa được cấu hình. Thêm OPENAI_API_KEY để dùng fallback."


# =============================================================================
# DEMO — chạy trực tiếp: python -m group_project.src.generation
# =============================================================================

if __name__ == "__main__":
    import sys
    import pathlib

    # Thêm group_project/src vào path để import retrieval
    sys.path.insert(0, str(pathlib.Path(__file__).parent))

    from retrieval import search, build_vector_db

    # Build DB nếu chưa có
    print("Kiểm tra / build vector DB...")
    build_vector_db()

    def retrieve_fn(q: str, k: int = TOP_K) -> list[dict]:
        return search(q, top_k=k)

    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý?",
        "Quy trình cai nghiện bắt buộc là gì?",
        "Danh mục chất ma tuý nhóm I gồm những gì?",
    ]

    history: list[dict] = []

    for q in test_queries:
        print(f"\n{'='*70}\nQ: {q}\n{'='*70}")
        chunks = retrieve_fn(q)
        print(f"[Retrieval] Lấy được {len(chunks)} chunks")
        result = generate(q, chunks, history=history)
        print(f"A: {result.answer}")
        print(f"\n[Sources: {result.unique_sources} | via {result.retrieval_source}]")

        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": result.answer})
        history = history[-12:]  # keep last 6 turns
