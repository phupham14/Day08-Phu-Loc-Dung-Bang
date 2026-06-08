"""
Group Project - Document Loading & Vector Store.

Load PDFs từ group_project/docs/, chunk, embed, và index vào ChromaDB.
Hỗ trợ OCR cho PDF scan (ảnh) với EasyOCR + cache kết quả.

Chạy lần đầu để build DB (lần đầu sẽ lâu do OCR):
    python -m group_project.src.retrieval

Rebuild từ đầu:
    python -m group_project.src.retrieval --force
"""

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path(__file__).parent.parent / "docs"
CHROMA_DB_DIR = Path(__file__).parent.parent / "chroma_db"
OCR_CACHE_DIR = Path(__file__).parent.parent / "ocr_cache"
COLLECTION_NAME = "drug_law_docs"

# Chunk lớn hơn bài cá nhân vì văn bản pháp luật có điều khoản dài
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "BAAI/bge-m3"

# Ngưỡng: nếu pypdf trích được ít hơn N ký tự/trang → coi là scan, dùng OCR
MIN_CHARS_PER_PAGE = 50

# Chỉ load các file có stem trong danh sách này (None = load tất cả PDF trong docs/).
DOCS_FILTER: list[str] | None = [
    "luat_phong_chong_ma_tuy_20_10_2025",
]

# Giới hạn trang OCR cho từng file (tính từ 0).
# None = lấy hết trang; list[int] = chỉ lấy các trang đó.
DOCS_CONFIG: dict[str, list[int] | None] = {
    "luat_phong_chong_ma_tuy_20_10_2025": None,  # 33 trang, lấy hết
}

# Module-level cache để tránh load lại model mỗi lần gọi search()
_embedding_model: SentenceTransformer | None = None
_chroma_collection = None
_ocr_reader = None


def _get_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        _chroma_collection = client.get_collection(COLLECTION_NAME)
    return _chroma_collection


# =============================================================================
# OCR helpers
# =============================================================================

def _ocr_cache_path(pdf_path: Path, pages: list[int] | None) -> Path:
    if pages is None:
        suffix = "all"
    else:
        suffix = f"p{pages[0]}-{pages[-1]}"
    return OCR_CACHE_DIR / f"{pdf_path.stem}_{suffix}.txt"


def _load_ocr_cache(pdf_path: Path, pages: list[int] | None) -> str | None:
    cache = _ocr_cache_path(pdf_path, pages)
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    return None


def _save_ocr_cache(pdf_path: Path, pages: list[int] | None, text: str):
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _ocr_cache_path(pdf_path, pages).write_text(text, encoding="utf-8")


def _ocr_pdf(pdf_path: Path, pages: list[int] | None = None) -> str:
    """
    OCR các trang chỉ định trong PDF scan bằng EasyOCR + PyMuPDF.

    Args:
        pdf_path: Đường dẫn tới file PDF.
        pages: Danh sách index trang (0-based) cần OCR.
               None = lấy hết. Ví dụ: list(range(13)) = 13 trang đầu.
    """
    import io
    import numpy as np
    import fitz  # pymupdf
    import easyocr
    from PIL import Image

    cached = _load_ocr_cache(pdf_path, pages)
    if cached is not None:
        print(f"  [cache] {pdf_path.name}")
        return cached

    doc = fitz.open(str(pdf_path))
    page_indices = pages if pages is not None else list(range(len(doc)))
    print(f"  [OCR] {pdf_path.name} — {len(page_indices)}/{len(doc)} trang...")

    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["vi", "en"], gpu=False, verbose=False)
    reader = _ocr_reader

    page_texts = []
    for i, page_idx in enumerate(page_indices, start=1):
        page = doc[page_idx]
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        img_np = np.array(img)

        lines = reader.readtext(img_np, detail=0, paragraph=True)
        page_texts.append("\n".join(lines))

        if i % 5 == 0 or i == len(page_indices):
            print(f"    ... {i}/{len(page_indices)} trang xong")

    full_text = "\n\n".join(page_texts)
    _save_ocr_cache(pdf_path, pages, full_text)
    print(f"  [OK] OCR xong, da cache: {_ocr_cache_path(pdf_path, pages).name}")
    return full_text


# =============================================================================
# STEP 1: Load PDFs (với OCR fallback)
# =============================================================================

def load_pdfs() -> list[dict]:
    """
    Load tất cả PDF trong docs/.

    Chiến lược:
    - Thử pypdf trước (nhanh, cho text PDF).
    - Nếu text trung bình < MIN_CHARS_PER_PAGE/trang → PDF scan → dùng OCR.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str, 'filename': str}}
    """
    from pypdf import PdfReader

    pdf_files = list(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"Không tìm thấy PDF nào trong {DOCS_DIR}")

    documents = []
    for pdf_path in pdf_files:
        if DOCS_FILTER is not None and pdf_path.stem not in DOCS_FILTER:
            print(f"  [skip] {pdf_path.name}")
            continue
        reader = PdfReader(str(pdf_path))
        num_pages = len(reader.pages)

        # Thử pypdf
        pages_text = [page.extract_text() or "" for page in reader.pages]
        total_chars = sum(len(t) for t in pages_text)
        avg_chars = total_chars / max(num_pages, 1)

        page_filter = DOCS_CONFIG.get(pdf_path.stem)  # None or list[int]

        if avg_chars >= MIN_CHARS_PER_PAGE:
            if page_filter is not None:
                full_text = "\n\n".join(pages_text[i] for i in page_filter if pages_text[i].strip())
            else:
                full_text = "\n\n".join(t for t in pages_text if t.strip())
            method = "pypdf"
        else:
            # PDF scan → OCR
            full_text = _ocr_pdf(pdf_path, pages=page_filter)
            method = "ocr"

        documents.append({
            "content": full_text,
            "metadata": {
                "source": pdf_path.stem,
                "type": "legal",
                "filename": pdf_path.name,
            },
        })
        print(f"  [OK] [{method}] {pdf_path.name} - {num_pages} trang, {len(full_text):,} ky tu")

    return documents


# =============================================================================
# STEP 2: Chunk
# =============================================================================

def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents dùng RecursiveCharacterTextSplitter.

    Chunk size 800 phù hợp với điều khoản pháp luật (thường ~300-600 ký tự/điều).
    Overlap 150 để không mất ngữ cảnh khi điều khoản bị cắt giữa chừng.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            if chunk_text.strip():
                chunks.append({
                    "content": chunk_text,
                    "metadata": {**doc["metadata"], "chunk_index": i},
                })

    return chunks


# =============================================================================
# STEP 3: Build Vector DB
# =============================================================================

def build_vector_db(force_rebuild: bool = False):
    """
    Build ChromaDB từ tất cả PDFs trong docs/.

    Nếu collection đã tồn tại, skip (không build lại) trừ khi force_rebuild=True.

    Args:
        force_rebuild: Xoá và build lại từ đầu.
    """
    import chromadb

    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    existing_names = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_names:
        if not force_rebuild:
            print(f"[OK] Collection '{COLLECTION_NAME}' da ton tai. Dung --force de rebuild.")
            return
        client.delete_collection(COLLECTION_NAME)
        print("  Da xoa collection cu.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Load → chunk
    print("\n[1/3] Loading PDFs...")
    docs = load_pdfs()
    print(f"\n[2/3] Chunking {len(docs)} documents...")
    chunks = chunk_documents(docs)
    print(f"  -> {len(chunks)} chunks")

    # Embed
    print(f"\n[3/3] Embedding với {EMBEDDING_MODEL}...")
    model = _get_model()
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=16)

    # Re-fetch collection reference (tránh stale reference sau long-running OCR/embed)
    collection = client.get_collection(COLLECTION_NAME)

    # Index
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[c["metadata"] for c in chunks],
    )

    print(f"\n[OK] Indexed {len(chunks)} chunks vao ChromaDB tai: {CHROMA_DB_DIR}")

    # Reset cache để dùng collection mới
    global _chroma_collection
    _chroma_collection = collection


# =============================================================================
# STEP 4: Search
# =============================================================================

def search(query: str, top_k: int = 5) -> list[dict]:
    """
    Semantic search trên vector DB.

    Args:
        query: Câu hỏi của user
        top_k: Số kết quả trả về

    Returns:
        List of {
            'content': str,
            'score': float,     # cosine similarity [0, 1]
            'metadata': dict,
            'source': str       # tên file PDF (không có extension)
        }
    """
    collection = _get_collection()
    model = _get_model()

    query_embedding = model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "content": doc,
            "score": round(1 - dist, 4),  # cosine distance → similarity
            "metadata": meta,
            "source": meta.get("source", "unknown"),
        })

    return output


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    print("=" * 60)
    print("Group Project — Build Vector DB")
    print("=" * 60)
    build_vector_db(force_rebuild=force)

    print("\n--- Test search ---")
    test_queries = [
        "Hình phạt tàng trữ trái phép chất ma tuý",
        "Quy trình cai nghiện bắt buộc",
        "Danh mục chất ma tuý nhóm I",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = search(q, top_k=2)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
