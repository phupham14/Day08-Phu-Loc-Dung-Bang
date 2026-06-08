"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""
import os
import weaviate
from weaviate.classes.query import MetadataQuery
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    
    if not weaviate_url or not weaviate_api_key:
        raise ValueError("❌ Thiếu cấu hình WEAVIATE_URL hoặc WEAVIATE_API_KEY trong file .env")

    # 2. Embed câu truy vấn bằng đúng mô hình ở Task 4
    # (Đảm bảo dùng đúng model tương thích với vector 384 chiều đã lưu trên mây)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode(query).tolist()
    
    # 3. Khởi tạo kết nối tới Weaviate Cloud
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key)
    )
    
    try:
        # Lấy Collection cần truy vấn
        collection = client.collections.get("DrugLawDocs")
        
        # 4. Thực hiện tìm kiếm Vector lân cận (Near Vector)
        results = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True) # Yêu cầu trả về khoảng cách vector
        )
        
        search_results = []
        for obj in results.objects:
            # Quy đổi từ khoảng cách (distance) sang độ tương đồng (similarity score)
            # Weaviate v4 mặc định trả về khoảng cách Cosine (Cosine Distance)
            # Công thức: Similarity = 1 - Distance
            distance = obj.metadata.distance if obj.metadata.distance is not None else 1.0
            score = 1.0 - distance
            
            # Đóng gói dữ liệu đầu ra theo đúng định dạng yêu cầu
            search_results.append({
                "content": obj.properties.get("content", ""),
                "score": score,
                "metadata": {
                    "source": obj.properties.get("source", ""),
                    "doc_type": obj.properties.get("doc_type", "")
                }
            })
            
        # 5. Sắp xếp kết quả giảm dần theo điểm số (Similarity Score Descending)
        # (Mặc định near_vector của Weaviate đã sắp xếp tối ưu, nhưng sorted lại giúp đảm bảo tuyệt đối yêu cầu bài tập)
        search_results.sort(key=lambda x: x["score"], reverse=True)
        
        return search_results

    except Exception as e:
        print(f"💥 Đã xảy ra lỗi khi tìm kiếm ngữ nghĩa: {str(e)}")
        return []
        
    finally:
        # Giải phóng kết nối an toàn sau khi lấy xong dữ liệu
        client.close()


if __name__ == "__main__":
    # Chạy thử nghiệm module tìm kiếm
    query_test = "hình phạt cho tội tàng trữ ma tuý"
    print(f"🔍 Đang tìm kiếm ngữ nghĩa cho câu hỏi: '{query_test}'...\n")
    
    results = semantic_search(query_test, top_k=5)
    
    if not results:
        print("Thông báo: Không tìm thấy kết quả nào hoặc kết nối thất bại.")
    else:
        for i, r in enumerate(results, 1):
            print(f"Top {i} [{r['score']:.3f}]")
            print(f"📄 Nguồn: {r['metadata']['source']} | Loại: {r['metadata']['doc_type']}")
            print(f"📝 Nội dung: {r['content'][:150]}...")
            print("-" * 50)
