"""
Task 5 — Semantic Search Module.

Dense Retrieval trên Weaviate Vector Store.
"""

from typing import List, Dict

import weaviate
from sentence_transformers import SentenceTransformer
from weaviate.classes.query import MetadataQuery

# =========================
# Config
# =========================

COLLECTION_NAME = "DrugLawDocs"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Load model 1 lần
_model = SentenceTransformer(EMBEDDING_MODEL)


def semantic_search(query: str, top_k: int = 10) -> List[Dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict
        }
    """

    if not query.strip():
        return []

    # =========================
    # Bước 1: Embed query
    # =========================
    query_embedding = _model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # =========================
    # Bước 2: Connect Weaviate
    # =========================
    client = weaviate.connect_to_local()

    try:
        collection = client.collections.get(COLLECTION_NAME)

        # =========================
        # Bước 3: Vector Search
        # =========================
        results = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True)
        )

        output = []

        for obj in results.objects:

            distance = (
                obj.metadata.distance
                if obj.metadata and obj.metadata.distance is not None
                else 1.0
            )

            similarity = 1.0 - distance

            output.append(
                {
                    "content": obj.properties.get("content", ""),
                    "score": float(similarity),
                    "metadata": {
                        "source": obj.properties.get("source"),
                        "doc_type": obj.properties.get("doc_type"),
                        "chunk_index": obj.properties.get("chunk_index"),
                    },
                }
            )

        # =========================
        # Bước 4: Sort descending
        # =========================
        output.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return output

    finally:
        client.close()


if __name__ == "__main__":
    results = semantic_search(
        "hình phạt cho tội tàng trữ ma tuý",
        top_k=5
    )

    print("\n=== RESULTS ===\n")

    for i, r in enumerate(results, start=1):
        print(
            f"{i}. Score={r['score']:.4f} | "
            f"Source={r['metadata']['source']}"
        )
        print(r["content"][:200])
        print("-" * 80)