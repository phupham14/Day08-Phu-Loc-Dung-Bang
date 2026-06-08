"""
Task 7 - Jina Reranking Module.

This implementation uses only Jina Reranker API.

Input:
    query + candidates from retrieval

Output:
    top_k candidates sorted by Jina relevance score descending
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
JINA_RERANK_MODEL = "jina-reranker-v2-base-multilingual"


def rerank_cross_encoder(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Rerank candidates using Jina Reranker v2 multilingual.

    Args:
        query: User query.
        candidates: List of {'content': str, 'score': float, 'metadata': dict}.
        top_k: Number of results after reranking.

    Returns:
        List of top_k candidates with:
            - score: Jina relevance score
            - rerank_score: same Jina relevance score
            - retrieval_score: original retrieval score, if available
    """
    if not candidates or top_k <= 0:
        return []

    jina_api_key = os.getenv("JINA_API_KEY")
    if not jina_api_key:
        raise ValueError("Missing JINA_API_KEY in .env")

    documents = [candidate.get("content", "") for candidate in candidates]

    response = requests.post(
        JINA_RERANK_URL,
        headers={
            "Authorization": f"Bearer {jina_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": JINA_RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": min(top_k, len(candidates)),
        },
        timeout=30,
    )
    response.raise_for_status()

    reranked_items = response.json().get("results", [])

    results: list[dict[str, Any]] = []
    for item in reranked_items:
        original = candidates[item["index"]].copy()
        jina_score = float(item["relevance_score"])

        if "score" in original:
            original["retrieval_score"] = original["score"]

        original["rerank_score"] = jina_score
        original["score"] = jina_score
        results.append(original)

    return results


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Main reranking interface for Task 7.

    This project is configured to use only Jina reranking, so rerank() simply
    delegates to rerank_cross_encoder().
    """
    return rerank_cross_encoder(query, candidates, top_k)


if __name__ == "__main__":
    dummy_candidates = [
        {
            "content": "Dieu 248: Toi tang tru trai phep chat ma tuy",
            "score": 0.8,
            "metadata": {"source": "dummy_legal"},
        },
        {
            "content": "Nghe si X bi bat vi su dung ma tuy",
            "score": 0.7,
            "metadata": {"source": "dummy_news"},
        },
        {
            "content": "Hinh phat tu 2-7 nam cho toi tang tru",
            "score": 0.6,
            "metadata": {"source": "dummy_legal"},
        },
    ]

    results = rerank("hinh phat tang tru ma tuy", dummy_candidates, top_k=2)
    for result in results:
        print(f"[{result['score']:.3f}] {result['content']}")
