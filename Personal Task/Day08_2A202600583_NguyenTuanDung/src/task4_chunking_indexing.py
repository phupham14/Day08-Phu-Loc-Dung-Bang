"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Lựa chọn chunking strategy MarkdownHeaderTextSplitter vì input có định dạng .md
CHUNK_SIZE = 500        # Vì sao chọn 500? ...
CHUNK_OVERLAP = 50      # Vì sao chọn 50? ...
CHUNKING_METHOD = "markdown_header"  # "recursive" | "markdown_header" | "semantic"

# TODO: Chọn embedding model và giải thích
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Vì sao? Multilingual, tốt cho tiếng Việt
EMBEDDING_DIM = 384

# TODO: Chọn vector store
VECTOR_STORE = "weaviate"  # "weaviate" | "chromadb" | "faiss"


# =============================================================================
# IMPLEMENTATIONs
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    # TODO: Iterate qua STANDARDIZED_DIR, đọc .md files
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return documents
    # raise NotImplementedError("Implement load_documents")


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    # TODO: Implement chunking
    
    from langchain_text_splitters import MarkdownHeaderTextSplitter
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, split_doc in enumerate(splits):
            chunks.append({
                "content": split_doc.page_content,
                "metadata": {
                    **doc["metadata"],
                    **split_doc.metadata,
                    "chunk_index": i
                }
            })
    return chunks
    # raise NotImplementedError("Implement chunk_documents")


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    # TODO: Implement embedding
    #
    # Ví dụ với sentence-transformers:
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks
    # raise NotImplementedError("Implement embed_chunks")


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào Weaviate Cloud v4.
    """
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType

    # 1. Lấy thông tin cấu hình từ file .env
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    
    if not weaviate_url or not weaviate_api_key:
        raise ValueError("❌ Thiếu cấu hình WEAVIATE_URL hoặc WEAVIATE_API_KEY trong file .env")

    print(f"🔌 Đang kết nối tới Weaviate Cloud tại: {weaviate_url} ...")
    
    # 2. Khởi tạo kết nối tới Weaviate Cloud (Client v4)
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key)
    )
    
    try:
        # Kiểm tra nếu kết nối sẵn sàng
        if not client.is_ready():
            print("⚠️ Máy chủ Weaviate Cloud phản hồi nhưng chưa sẵn sàng.")
            return

        collection_name = "DrugLawDocs"

        # 3. [Tùy chọn nhưng nên có] Xóa collection cũ nếu đã tồn tại để tránh ghi đè/trùng lặp khi test lại
        if client.collections.exists(collection_name):
            print(f"🗑️ Đã tìm thấy Collection '{collection_name}' cũ. Đang tiến hành xóa để làm sạch dữ liệu...")
            client.collections.delete(collection_name)

        print(f"🏗️ Đang tạo mới Collection: '{collection_name}'...")
        
        # 4. Tạo cấu trúc Collection mới
        collection = client.collections.create(
            name=collection_name,
            vectorizer_config=Configure.Vectorizer.none(),  # Nhận vector tự định nghĩa từ local gửi lên
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
            ]
        )
        
        # 5. Thực hiện Insert Chunks bằng phương thức Batch tối ưu tốc độ mạng
        print(f"📦 Đang tải {len(chunks)} chunks lên Weaviate Cloud...")
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                # Đảm bảo đọc đúng cấu trúc metadata của bạn
                source_val = chunk.get("metadata", {}).get("source", "")
                doc_type_val = chunk.get("metadata", {}).get("type", "")
                
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": source_val,
                        "doc_type": doc_type_val
                    },
                    vector=chunk["embedding"]  # Truyền vector 384 hoặc 1024 chiều đã embed từ local
                )
                
        # 6. Kiểm tra lỗi nếu có trong quá trình đẩy Batch lên Cloud
        if collection.batch.failed_objects:
            print(f"❌ Có {len(collection.batch.failed_objects)} objects lỗi khi lưu lên Cloud.")
        else:
            print("✨ Hoàn thành! Toàn bộ chunks đã được index thành công lên Weaviate Cloud.")
            
    except Exception as e:
        print(f"💥 Đã xảy ra lỗi trong quá trình indexing: {str(e)}")
        
    finally:
        # Luôn đóng kết nối một cách an toàn
        client.close()
        print("🔒 Đã đóng kết nối với Weaviate Cloud.")
    # raise NotImplementedError("Implement index_to_vectorstore")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
