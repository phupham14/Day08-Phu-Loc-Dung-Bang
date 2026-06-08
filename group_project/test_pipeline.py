"""
Test tích hợp: retrieval.py (Person 1) + generation.py (Person 2)

Chạy từ thư mục gốc repo:
    python -m group_project.test_pipeline
"""

import sys
from pathlib import Path

# Thêm group_project/src vào path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from retrieval import search, build_vector_db
from generation import generate, TOP_K

# Build DB nếu chưa có (skip nếu đã có)
print("=" * 60)
print("Kiểm tra / build vector DB...")
print("=" * 60)
build_vector_db()

TEST_QUERIES = [
    "Hình phạt cho tội tàng trữ trái phép chất ma tuý?",
    "Quy trình cai nghiện bắt buộc là gì?",
    "Danh mục chất ma tuý nhóm I gồm những gì?",
]

print("\n" + "=" * 60)
print("Bắt đầu test pipeline: Retrieval → Generation")
print("=" * 60)

for q in TEST_QUERIES:
    print(f"\nQ: {q}")
    print("-" * 60)

    # Bước 1: Retrieval
    chunks = search(q, top_k=TOP_K)
    print(f"[Retrieval] {len(chunks)} chunks, scores: {[c['score'] for c in chunks]}")

    # Bước 2: Generation
    result = generate(q, chunks)
    print(f"[Generation] source: {result.retrieval_source}")
    print(f"\nA: {result.answer}")
    print(f"\n[Sources: {result.unique_sources}]")
    print("=" * 60)
