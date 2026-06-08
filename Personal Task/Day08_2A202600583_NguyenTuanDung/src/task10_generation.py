"""
Task 10 - Generation with Citation.

This module completes the RAG pipeline:
    1. Retrieve relevant chunks.
    2. Reorder chunks to reduce the "lost in the middle" effect.
    3. Format context with citation-friendly source labels.
    4. Generate an answer through OpenRouter.
    5. Return answer + source chunks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


# =============================================================================
# CONFIGURATION
# =============================================================================

# top_k: number of chunks injected into the prompt. 5 is enough evidence for
# concise legal/news answers without making the context too long.
TOP_K = 5

# top_p: nucleus sampling. 0.9 keeps wording natural while still constrained.
TOP_P = 0.9

# temperature: low value because RAG answers should be factual.
TEMPERATURE = 0.3

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
)
# Set TASK10_USE_OPENROUTER=1 in .env to call the real OpenRouter model.
USE_OPENROUTER = os.getenv("TASK10_USE_OPENROUTER", "0") == "1"


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý RAG trả lời bằng tiếng Việt.

Chỉ sử dụng thông tin trong CONTEXT được cung cấp. Mỗi nhận định thực tế phải có
citation ngay sau câu, dùng nhãn nguồn có trong context như [D1], [D2].

Nếu context không đủ bằng chứng để trả lời, hãy nói:
"Tôi không thể xác minh thông tin này từ nguồn hiện có."

Quy tắc:
- Không bịa thông tin ngoài context.
- Trả lời trực tiếp, rõ ràng.
- Mọi câu có dữ kiện pháp luật, sự kiện, con số, tên văn bản hoặc tên người đều phải có citation.
- Nếu các nguồn mâu thuẫn hoặc không đủ chi tiết, nêu rõ giới hạn đó."""


# =============================================================================
# DOCUMENT REORDERING
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Reorder chunks to reduce "lost in the middle".

    Given chunks sorted by score descending, keep the best chunk at the start,
    place odd-ranked chunks after it, and move even-ranked chunks to the end in
    reverse order. Example:
        [1, 2, 3, 4, 5] -> [1, 3, 5, 4, 2]
    """
    if len(chunks) <= 2:
        return chunks

    reordered: list[dict] = []

    for i in range(0, len(chunks), 2):
        reordered.append(chunks[i])

    last_even_index = len(chunks) - 1 if len(chunks) % 2 == 0 else len(chunks) - 2
    for i in range(last_even_index, 0, -2):
        reordered.append(chunks[i])

    return reordered


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def _metadata_value(metadata: dict[str, Any], *keys: str, default: str = "unknown") -> str:
    for key in keys:
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a citation-ready context string.

    Each chunk receives a stable label [D1], [D2], ... so the LLM can cite it.
    """
    if not chunks:
        return "Không có context phù hợp."

    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}
        source = _metadata_value(metadata, "source", "filename", default=f"Source {i}")
        doc_type = _metadata_value(metadata, "type", "doc_type", default="unknown")
        chunk_index = _metadata_value(metadata, "chunk_index", "page", "page_num", default="n/a")
        score = float(chunk.get("score", 0.0) or 0.0)
        content = str(chunk.get("content", "")).strip()

        context_parts.append(
            f"[D{i}] Source: {source} | Type: {doc_type} | Chunk/Page: {chunk_index} | "
            f"Score: {score:.3f}\n{content}"
        )

    return "\n\n---\n\n".join(context_parts)


def _build_fallback_answer(query: str, chunks: list[dict]) -> str:
    """
    Deterministic fallback when the LLM API is unavailable.
    """
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    first = chunks[0].get("content", "").strip()
    if not first:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    snippet = first[:700]
    return (
        "Dựa trên nguồn truy xuất được, thông tin liên quan nhất là: "
        f"{snippet} [D1]\n\n"
        "Tôi không thể xác minh thêm các chi tiết ngoài đoạn context hiện có."
    )


def _retrieve_chunks(query: str, top_k: int) -> list[dict]:
    """
    Import retrieval lazily so helper functions stay fast to import and test.
    """
    try:
        try:
            from .task9_retrieval_pipeline import retrieve
        except ImportError:
            from task9_retrieval_pipeline import retrieve

        return retrieve(query, top_k=top_k)
    except Exception as exc:
        print(f"Retrieval failed: {exc}")
        return []


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation with citations.

    Returns:
        {
            'answer': str,
            'sources': list[dict],
            'retrieval_source': str
        }
    """
    chunks = _retrieve_chunks(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    user_message = f"""CONTEXT:
{context}

QUESTION:
{query}

Hãy trả lời bằng tiếng Việt và cite bằng các nhãn [D1], [D2], ... tương ứng."""

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    answer: str

    if not USE_OPENROUTER:
        answer = _build_fallback_answer(query, reordered)
    elif not api_key:
        answer = _build_fallback_answer(query, reordered)
    else:
        try:
            import requests

            response = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "Day08 RAG Pipeline",
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                },
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"].get("content") or ""
            if not answer.strip():
                answer = _build_fallback_answer(query, reordered)
        except Exception as exc:
            print(f"OpenRouter generation failed: {exc}")
            answer = _build_fallback_answer(query, reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
    }


if __name__ == "__main__":
    import sys

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma túy theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma túy?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma túy?",
    ]

    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
