"""
Task 6 — Lexical Search Module (BM25).

Cài đặt:
    pip install rank-bm25

Corpus được load từ:
    data/standardized/*.md

Output:
    List[{
        "content": str,
        "score": float,
        "metadata": dict
    }]
"""

from pathlib import Path

from rank_bm25 import BM25Okapi
import numpy as np


# ==========================================================
# Config
# ==========================================================

DATA_DIR = Path(__file__).parent.parent / "data" / "standardized"

CORPUS: list[dict] = []
BM25_INDEX = None


# ==========================================================
# Load Corpus
# ==========================================================

def load_corpus() -> list[dict]:
    """
    Load markdown files từ data/standardized.

    Returns:
        List[{
            "content": str,
            "metadata": dict
        }]
    """

    corpus = []

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục: {DATA_DIR}"
        )

    files = list(DATA_DIR.rglob("*.md"))
    print(f"Found {len(files)} markdown files")
    for file_path in files:
        try:
            content = file_path.read_text(
                encoding="utf-8"
            ).strip()

            if not content:
                continue

            corpus.append(
                {
                    "content": content,
                    "metadata": {
                        "source": str(file_path),
                        "filename": file_path.name,
                    },
                }
            )

        except Exception as e:
            print(f"Lỗi đọc {file_path}: {e}")

    return corpus


# ==========================================================
# BM25 Index
# ==========================================================

def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {
            'content': str,
            'metadata': dict
        }

    Returns:
        BM25Okapi object
    """
    if len(corpus) == 0:
        raise ValueError(
            "Corpus is empty. No markdown files loaded."
        )
    tokenized_corpus = [
        doc["content"].lower().split()
        for doc in corpus
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25


# ==========================================================
# Search
# ==========================================================

def lexical_search(
    query: str,
    top_k: int = 10
) -> list[dict]:

    global BM25_INDEX
    global CORPUS

    if BM25_INDEX is None:
        CORPUS = load_corpus()
        BM25_INDEX = build_bm25_index(CORPUS)

    query_tokens = query.lower().split()

    scores = BM25_INDEX.get_scores(query_tokens)

    ranked = sorted(
        zip(CORPUS, scores),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for doc, score in ranked[:top_k]:
        results.append({
            "content": doc["content"],
            "score": float(score),
            "metadata": doc.get("metadata", {})
        })

    return results

# ==========================================================
# Initialize
# ==========================================================

def initialize():
    global CORPUS
    global BM25_INDEX

    CORPUS = load_corpus()

    print(
        f"Loaded {len(CORPUS)} documents."
    )

    BM25_INDEX = build_bm25_index(
        CORPUS
    )

    print("BM25 index ready.")


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    initialize()

    query = (
        "Điều 248 tàng trữ trái phép chất ma tuý"
    )

    results = lexical_search(
        query=query,
        top_k=5
    )

    print("\nRESULTS")
    print("=" * 80)

    for i, r in enumerate(results, start=1):

        print(
            f"\n#{i}"
        )
        print(
            f"Score: {r['score']:.4f}"
        )
        print(
            f"File : {r['metadata']['filename']}"
        )
        print(
            f"Text : {r['content'][:200]}..."
        )