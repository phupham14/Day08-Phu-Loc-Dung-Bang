"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    Nếu không có API key, bỏ qua.
    """
    if not PAGEINDEX_API_KEY:
        print("⚠ PAGEINDEX_API_KEY không được cấu hình. Bỏ qua upload.")
        return

    try:
        from pageindex import PageIndex

        pi = PageIndex(api_key=PAGEINDEX_API_KEY)

        for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8", errors="ignore")
                if not content.strip():
                    continue
                pi.upload(
                    content=content,
                    metadata={"filename": md_file.name, "type": md_file.parent.name}
                )
                print(f"  ✓ Uploaded: {md_file.name}")
            except Exception as e:
                print(f"  Warning: failed to upload {md_file.name}: {e}")
    except Exception as e:
        print(f"PageIndex upload error: {e}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Nếu PAGEINDEX_API_KEY không được cấu hình, fallback sang
    đọc trực tiếp từ data/standardized/ với keyword scoring đơn giản.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if PAGEINDEX_API_KEY:
        try:
            from pageindex import PageIndex
            pi = PageIndex(api_key=PAGEINDEX_API_KEY)
            results = pi.query(query=query, top_k=top_k)
            return [
                {
                    "content": r.text,
                    "score": r.score,
                    "metadata": r.metadata or {},
                    "source": "pageindex"
                }
                for r in results
            ]
        except Exception as e:
            print(f"PageIndex API error: {e}. Falling back to local search.")

    # Fallback: keyword scoring over local standardized files
    return _local_fallback_search(query, top_k)


def _local_fallback_search(query: str, top_k: int) -> list[dict]:
    """Tìm kiếm đơn giản trên file local khi không có PageIndex API."""
    query_lower = query.lower()
    query_words = set(query_lower.split())

    candidates = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8", errors="ignore")
            if len(content.strip()) < 50:
                continue

            content_lower = content.lower()
            word_hits = sum(1 for w in query_words if w in content_lower)
            score = word_hits / max(len(query_words), 1)

            doc_type = "legal" if "legal" in str(md_file) else "news"
            candidates.append({
                "content": content[:600],
                "score": float(score),
                "metadata": {"source": md_file.name, "type": doc_type},
                "source": "pageindex"
            })
        except Exception:
            continue

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
        print("\nThử fallback search:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] [{r['source']}] {r['content'][:100]}...")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
