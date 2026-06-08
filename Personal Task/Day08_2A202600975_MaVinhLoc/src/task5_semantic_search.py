"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

from pathlib import Path

# Mirrors task4 constants so both modules share the same index
_CHROMA_DIR = Path(__file__).parent.parent / "data" / "chromadb"
_COLLECTION_NAME = "drug_law_docs"
_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embedding_model = None
_chroma_collection = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(_EMBEDDING_MODEL)
    return _embedding_model


def _get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
        _chroma_collection = client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _chroma_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score [0, 1]
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    model = _get_embedding_model()
    collection = _get_collection()

    # Return empty if collection is not populated
    count = collection.count()
    if count == 0:
        return []

    # Embed query with same model used during indexing
    query_embedding = model.encode(query).tolist()

    # Query ChromaDB (returns distances, lower = more similar for cosine)
    n = min(top_k, count)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"]
    )

    output = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        # Convert cosine distance [0,2] to similarity [0,1]
        score = 1.0 - distance / 2.0
        output.append({
            "content": results["documents"][0][i],
            "score": float(score),
            "metadata": results["metadatas"][0][i] or {}
        })

    return sorted(output, key=lambda x: x["score"], reverse=True)


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
