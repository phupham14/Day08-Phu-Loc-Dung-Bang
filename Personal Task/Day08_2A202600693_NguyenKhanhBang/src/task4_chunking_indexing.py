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

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# =============================================================================
# CONFIGURATION
# =============================================================================

# RecursiveCharacterTextSplitter:
# - Hoạt động tốt với mọi loại markdown
# - Không phụ thuộc cấu trúc heading
# - Ổn định và phổ biến nhất trong production RAG

CHUNK_SIZE = 500 # Đủ ngữ cảnh nhưng không quá dài gây nhiễu retrieval
CHUNK_OVERLAP = 50 # Giúp giữ mạch logic giữa các chunks, tránh mất thông tin quan trọng ở ranh giới
CHUNKING_METHOD = "recursive" # Đơn giản, ổn định, hoạt động với mọi markdown

# BAAI/bge-small-en-v1.5:
# - Multilingual mạnh
# - Hỗ trợ tiếng Việt tốt hơn MiniLM
# - Chất lượng retrieval cao
# - 384 dimensions

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# Weaviate:
# - Hybrid search built-in
# - Metadata filtering tốt
# - Scalable hơn Chroma/FAISS

VECTOR_STORE = "weaviate"

COLLECTION_NAME = "DrugLawDocs"

# =============================================================================
# IMPLEMENTATION
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
        content = md_file.read_text(
            encoding="utf-8",
            errors="ignore"
        )
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": str(md_file),
                    "type": doc_type,
                },
            }
        )
    return documents

def chunk_documents(documents: list[dict]) -> list[dict]:

    chunks = []

    if CHUNKING_METHOD == "recursive":

        from langchain_text_splitters import (
            RecursiveCharacterTextSplitter,
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

        for doc in documents:
            splits = splitter.split_text(doc["content"])

            for idx, chunk in enumerate(splits):
                chunks.append(
                    {
                        "content": chunk,
                        "metadata": {
                            **doc["metadata"],
                            "chunk_index": idx,
                        },
                    }
                )

    elif CHUNKING_METHOD == "markdown_header":

        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
        )

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ]
        )

        for doc in documents:

            splits = splitter.split_text(doc["content"])

            for idx, split in enumerate(splits):
                chunks.append(
                    {
                        "content": split.page_content,
                        "metadata": {
                            **doc["metadata"],
                            **split.metadata,
                            "chunk_index": idx,
                        },
                    }
                )

    elif CHUNKING_METHOD == "semantic":

        from langchain_experimental.text_splitter import (
            SemanticChunker,
        )

        from langchain_community.embeddings import (
            HuggingFaceEmbeddings,
        )

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        splitter = SemanticChunker(
            embeddings
        )

        for doc in documents:

            splits = splitter.split_text(
                doc["content"]
            )

            for idx, chunk in enumerate(splits):
                chunks.append(
                    {
                        "content": chunk,
                        "metadata": {
                            **doc["metadata"],
                            "chunk_index": idx,
                        },
                    }
                )

    else:
        raise ValueError(
            f"Unknown chunking method: {CHUNKING_METHOD}"
        )

    return chunks

from fastembed import TextEmbedding
def embed_chunks(chunks: list[dict]) -> list[dict]:

    model = TextEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    texts = [
        c["content"]
        for c in chunks
    ]

    embeddings = list(
    model.embed(
        texts,
        batch_size=8
    ))

    for chunk, emb in zip(
        chunks,
        embeddings
    ):
        chunk["embedding"] = emb.tolist()

    return chunks

def index_to_vectorstore(
    chunks: list[dict]
):
    import weaviate

    from weaviate.classes.config import (
        Configure,
        Property,
        DataType,
    )

    client = weaviate.connect_to_local()

    try:
        existing = list(client.collections.list_all().keys())
        
        if COLLECTION_NAME not in existing:
            client.collections.create(
                name=COLLECTION_NAME,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(
                        name="content",
                        data_type=DataType.TEXT,
                    ),
                    Property(
                        name="source",
                        data_type=DataType.TEXT,
                    ),
                    Property(
                        name="path",
                        data_type=DataType.TEXT,
                    ),
                    Property(
                        name="doc_type",
                        data_type=DataType.TEXT,
                    ),
                    Property(
                        name="chunk_index",
                        data_type=DataType.INT,
                    ),
                ],
            )

        collection = client.collections.get(
            COLLECTION_NAME
        )

        with collection.batch.dynamic() as batch:

            for chunk in chunks:

                batch.add_object(
                    properties={
                        "content":
                            chunk["content"],
                        "source":
                            chunk["metadata"]["source"],
                        "path":
                            chunk["metadata"].get(
    "path",
    "unknown"
),
                        "doc_type":
                            chunk["metadata"]["type"],
                        "chunk_index":
                            chunk["metadata"][
                                "chunk_index"
                            ],
                    },
                    vector=chunk["embedding"],
                )

        failed = (
            collection.batch.failed_objects
        )

        if failed:
            print(
                f"Failed objects: {len(failed)}"
            )

    finally:
        client.close()

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
