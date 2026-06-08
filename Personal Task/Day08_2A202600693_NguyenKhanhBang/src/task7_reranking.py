"""
Task 7 — Reranking Module.
"""

from __future__ import annotations

from typing import Optional
import math


# =============================================================================
# Utils
# =============================================================================

def cosine_sim(vec1: list[float], vec2: list[float]) -> float:
    """Cosine similarity."""

    dot = sum(a * b for a, b in zip(vec1, vec2))

    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


# =============================================================================
# Cross Encoder
# =============================================================================

def rerank_cross_encoder(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Cross-encoder reranking.

    Expected candidate format:
    {
        "content": str,
        "score": float,
        "metadata": dict
    }

    Notes:
        - Jina API version can be plugged in here.
        - Local Qwen3-Reranker can also be used.
    """

    try:
        import os
        import requests

        jina_api_key = os.getenv("JINA_API_KEY")

        if not jina_api_key:
            raise ValueError("JINA_API_KEY not found")

        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={
                "Authorization": f"Bearer {jina_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": [
                    c["content"]
                    for c in candidates
                ],
                "top_n": min(top_k, len(candidates)),
            },
            timeout=30,
        )

        response.raise_for_status()

        reranked = response.json()["results"]

        results = []

        for r in reranked:
            item = candidates[r["index"]].copy()
            item["score"] = r["relevance_score"]
            results.append(item)

        return results

    except Exception as e:
        raise NotImplementedError(
            f"Cross-encoder unavailable: {e}"
        )


# =============================================================================
# MMR
# =============================================================================

def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance.

    candidate format:
    {
        "content": str,
        "score": float,
        "embedding": list[float],
        "metadata": dict
    }
    """

    if not candidates:
        return []

    selected = []
    remaining = list(range(len(candidates)))

    max_select = min(top_k, len(candidates))

    for _ in range(max_select):

        best_idx = None
        best_mmr_score = float("-inf")

        for idx in remaining:

            doc_embedding = candidates[idx]["embedding"]

            relevance = cosine_sim(
                query_embedding,
                doc_embedding,
            )

            if not selected:
                diversity_penalty = 0.0

            else:
                diversity_penalty = max(
                    cosine_sim(
                        doc_embedding,
                        candidates[sel]["embedding"],
                    )
                    for sel in selected
                )

            mmr_score = (
                lambda_param * relevance
                - (1 - lambda_param) * diversity_penalty
            )

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    results = []

    for idx in selected:
        item = candidates[idx].copy()

        item["mmr_score"] = (
            lambda_param
            * cosine_sim(
                query_embedding,
                item["embedding"],
            )
        )

        results.append(item)

    return results


# =============================================================================
# RRF
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion.

    RRF(d) = Σ 1/(k + rank)
    """

    rrf_scores = {}

    content_map = {}

    for ranked_list in ranked_lists:

        for rank, item in enumerate(
            ranked_list,
            start=1,
        ):

            key = item["content"]

            rrf_scores[key] = (
                rrf_scores.get(key, 0.0)
                + 1.0 / (k + rank)
            )

            content_map[key] = item

    sorted_docs = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []

    for content, score in sorted_docs[:top_k]:

        item = content_map[content].copy()

        item["score"] = score

        results.append(item)

    return results


# =============================================================================
# Main Interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
):
    """
    Unified interface.
    """

    if method == "cross_encoder":
        return rerank_cross_encoder(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )

    raise ValueError(
        "Use rerank_mmr() or rerank_rrf() directly."
    )


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":

    candidates = [
        {
            "content": "Điều 248: Tội tàng trữ trái phép chất ma tuý",
            "score": 0.8,
            "embedding": [0.9, 0.1, 0.2],
            "metadata": {},
        },
        {
            "content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý",
            "score": 0.7,
            "embedding": [0.1, 0.8, 0.3],
            "metadata": {},
        },
        {
            "content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ",
            "score": 0.6,
            "embedding": [0.85, 0.15, 0.25],
            "metadata": {},
        },
    ]

    query_embedding = [1.0, 0.0, 0.0]

    print("\n=== MMR ===")

    mmr_results = rerank_mmr(
        query_embedding=query_embedding,
        candidates=candidates,
        top_k=2,
    )

    for r in mmr_results:
        print(r["content"])

    print("\n=== RRF ===")

    lexical_rank = [
        candidates[0],
        candidates[2],
        candidates[1],
    ]

    semantic_rank = [
        candidates[2],
        candidates[0],
        candidates[1],
    ]

    rrf_results = rerank_rrf(
        [lexical_rank, semantic_rank],
        top_k=3,
    )

    for r in rrf_results:
        print(
            f"{r['score']:.5f}",
            r["content"],
        )