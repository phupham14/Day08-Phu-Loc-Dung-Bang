"""
Task 9 - Complete Retrieval Pipeline.

Pipeline:
    1. Run semantic search and lexical search in parallel.
    2. Merge both result lists with RRF (Reciprocal Rank Fusion).
    3. Rerank merged results with Jina reranker from Task 7.
    4. If results are weak or unavailable, fallback to PageIndex from Task 8.
    5. Return top_k final results.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError
from typing import Any

try:
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank
    from .task8_pageindex_vectorless import pageindex_search
except ImportError:
    from task6_lexical_search import lexical_search
    from task7_reranking import rerank
    from task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RRF_K = 60
SEARCH_TIMEOUT_SECONDS = 12
RERANK_TIMEOUT_SECONDS = 15
USE_SEMANTIC_SEARCH = os.getenv("TASK9_USE_SEMANTIC", "0") == "1"
USE_JINA_RERANKING = os.getenv("TASK9_USE_JINA", "0") == "1"


def _doc_key(item: dict[str, Any]) -> tuple[Any, ...]:
    """
    Build a stable identity for deduplication while merging rankers.
    """
    metadata = item.get("metadata") or {}
    return (
        metadata.get("source") or metadata.get("filename"),
        metadata.get("chunk_index"),
        item.get("content", ""),
    )


def _safe_search(search_name: str, search_fn, query: str, top_k: int) -> list[dict]:
    """
    Run a retriever safely so one failing backend does not break the pipeline.
    """
    try:
        return search_fn(query, top_k=top_k)
    except Exception as exc:
        print(f"  {search_name} failed: {exc}")
        return []


def _semantic_search(query: str, top_k: int) -> list[dict]:
    try:
        try:
            from .task5_semantic_search import semantic_search
        except ImportError:
            from task5_semantic_search import semantic_search

        return semantic_search(query, top_k=top_k)
    except Exception as exc:
        print(f"  Semantic search failed: {exc}")
        return []


def _merge_rrf(
    ranked_lists: list[list[dict]],
    top_k: int,
    k: int = RRF_K,
) -> list[dict]:
    """
    Merge ranked results using Reciprocal Rank Fusion.

    RRF(d) = sum(1 / (k + rank_r(d)))

    The raw RRF score is small, so score is normalized to 0-1 for easier
    thresholding. The original raw score is kept as rrf_score.
    """
    rrf_scores: dict[tuple[Any, ...], float] = {}
    item_map: dict[tuple[Any, ...], dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = _doc_key(item)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            item_map.setdefault(key, item)

    if not rrf_scores:
        return []

    max_score = max(rrf_scores.values())
    sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

    merged = []
    for key in sorted_keys[:top_k]:
        original = item_map[key].copy()
        raw_score = rrf_scores[key]
        if "score" in original:
            original["retrieval_score"] = original["score"]
        original["rrf_score"] = raw_score
        original["score"] = raw_score / max_score if max_score > 0 else 0.0
        original["source"] = "hybrid"
        merged.append(original)

    return merged


def _fallback_pageindex(query: str, top_k: int, reason: str) -> list[dict]:
    print(f"  {reason}. Fallback -> PageIndex")
    return pageindex_search(query, top_k=top_k)


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Complete retrieval pipeline with PageIndex fallback.

    Args:
        query: User query.
        top_k: Number of final results.
        score_threshold: Minimum acceptable top score before fallback.
        use_reranking: Whether to apply Jina reranking after hybrid merge.

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'hybrid' | 'pageindex'
        }
    """
    if top_k <= 0:
        return []

    retrieval_top_k = top_k * 2

    if USE_SEMANTIC_SEARCH:
        executor = ThreadPoolExecutor(max_workers=2)
        dense_future = executor.submit(
            _safe_search, "Semantic search", _semantic_search, query, retrieval_top_k
        )
        sparse_future = executor.submit(
            _safe_search, "Lexical search", lexical_search, query, retrieval_top_k
        )

        try:
            sparse_results = sparse_future.result(timeout=SEARCH_TIMEOUT_SECONDS)
        except TimeoutError:
            print("  Lexical search timed out")
            sparse_results = []

        try:
            dense_results = dense_future.result(timeout=SEARCH_TIMEOUT_SECONDS)
        except TimeoutError:
            print("  Semantic search timed out")
            dense_results = []

        executor.shutdown(wait=False, cancel_futures=True)
    else:
        dense_results = []
        sparse_results = _safe_search(
            "Lexical search", lexical_search, query, retrieval_top_k
        )

    merged_results = _merge_rrf(
        [dense_results, sparse_results],
        top_k=retrieval_top_k,
    )

    if not merged_results:
        return _fallback_pageindex(query, top_k, "Hybrid retrieval returned no results")

    if use_reranking and USE_JINA_RERANKING:
        rerank_executor = ThreadPoolExecutor(max_workers=1)
        rerank_future = rerank_executor.submit(rerank, query, merged_results, top_k)
        try:
            final_results = rerank_future.result(timeout=RERANK_TIMEOUT_SECONDS)
            for item in final_results:
                item["source"] = "hybrid"
        except TimeoutError:
            print("  Reranking timed out")
            final_results = merged_results[:top_k]
        except Exception as exc:
            print(f"  Reranking failed: {exc}")
            final_results = merged_results[:top_k]
        finally:
            rerank_executor.shutdown(wait=False, cancel_futures=True)
    else:
        final_results = merged_results[:top_k]

    if not final_results:
        return _fallback_pageindex(query, top_k, "Reranking returned no results")

    best_score = float(final_results[0].get("score", 0.0))
    if best_score < score_threshold:
        return _fallback_pageindex(
            query,
            top_k,
            f"Hybrid score {best_score:.3f} < threshold {score_threshold:.3f}",
        )

    return final_results[:top_k]


if __name__ == "__main__":
    import sys

    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    test_queries = [
        "Hinh phat cho toi tang tru trai phep chat ma tuy",
        "Nghe si nao bi bat vi su dung ma tuy nam 2024",
        "Luat phong chong ma tuy quy dinh gi ve cai nghien",
    ]

    for query_text in test_queries:
        print(f"\nQuery: {query_text}")
        print("-" * 60)
        results = retrieve(query_text, top_k=3)
        for i, result in enumerate(results, start=1):
            print(
                f"  {i}. [{result['score']:.3f}] "
                f"[{result['source']}] {result['content'][:80]}..."
            )
