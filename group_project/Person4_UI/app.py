"""
RAG Chatbot UI — Person4_UI/app.py
Person 4: Nguyen Khanh Bang

Streamlit chatbot: chat interface + hiển thị sources + ghép pipeline.

Chạy từ thư mục gốc repo (Day08_Group):
    streamlit run group_project/Person4_UI/app.py
"""

import sys
from pathlib import Path

# Trỏ tới group_project/src/ để import retrieval, generation, memory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
from retrieval import search, build_vector_db
from generation import generate, generate_stream, TOP_K
from memory import ConversationMemory

# =============================================================================
# Page config
# =============================================================================

st.set_page_config(
    page_title="Chatbot Pháp Luật Ma Tuý",
    page_icon="⚖️",
    layout="wide",
)

# =============================================================================
# Session state init
# =============================================================================

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()

if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []

if "db_ready" not in st.session_state:
    st.session_state.db_ready = False

# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.title("⚖️ RAG Chatbot")
    st.caption("Pháp luật phòng chống ma tuý Việt Nam")
    st.divider()

    st.subheader("Cài đặt")
    top_k = st.slider("Số chunks retrieval (top_k)", min_value=1, max_value=10, value=TOP_K)
    use_stream = st.toggle("Streaming response", value=True)

    st.divider()

    if st.button("🗑️ Xoá lịch sử hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()

    st.divider()
    st.caption(f"Memory: {st.session_state.memory}")
    st.caption("Nguồn: Luật 120/2025/QH15")

# =============================================================================
# Build DB once
# =============================================================================

if not st.session_state.db_ready:
    with st.spinner("Đang khởi động vector database..."):
        try:
            build_vector_db()
            st.session_state.db_ready = True
        except Exception as e:
            st.error(f"Lỗi khởi động DB: {e}")
            st.stop()

# =============================================================================
# Main header
# =============================================================================

st.title("⚖️ Chatbot Pháp Luật Phòng Chống Ma Tuý")
st.caption("Hỏi đáp dựa trên Luật Phòng chống ma tuý 120/2025/QH15")

# =============================================================================
# Render chat history
# =============================================================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📄 Nguồn tham khảo ({len(msg['sources'])} chunks)"):
                for i, src in enumerate(msg["sources"], 1):
                    score = src.get("score", 0)
                    source_name = src.get("metadata", {}).get("source", "unknown")
                    content_preview = src.get("content", "")[:200]
                    st.markdown(f"**[{i}] {source_name}** — score: `{score:.3f}`")
                    st.caption(content_preview + "...")
                    if i < len(msg["sources"]):
                        st.divider()

# =============================================================================
# Chat input
# =============================================================================

if prompt := st.chat_input("Đặt câu hỏi về pháp luật ma tuý..."):
    # Hiển thị message user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieval
    with st.spinner("Đang tìm kiếm tài liệu liên quan..."):
        chunks = search(prompt, top_k=top_k)

    # Generation
    with st.chat_message("assistant"):
        if use_stream:
            placeholder = st.empty()
            full_answer = ""
            for token in generate_stream(prompt, chunks, history=st.session_state.memory.get()):
                full_answer += token
                placeholder.markdown(full_answer + "▌")
            placeholder.markdown(full_answer)
            answer = full_answer
        else:
            with st.spinner("Đang tạo câu trả lời..."):
                result = generate(prompt, chunks, history=st.session_state.memory.get())
            answer = result.answer
            st.markdown(answer)

        # Hiển thị sources ngay sau câu trả lời
        if chunks:
            unique_src_names = list({c.get("metadata", {}).get("source", "?") for c in chunks})
            st.caption(f"📄 Dựa trên: {', '.join(unique_src_names)}")
            with st.expander(f"📄 Nguồn tham khảo ({len(chunks)} chunks)"):
                for i, src in enumerate(chunks, 1):
                    score = src.get("score", 0)
                    source_name = src.get("metadata", {}).get("source", "unknown")
                    content_preview = src.get("content", "")[:200]
                    st.markdown(f"**[{i}] {source_name}** — score: `{score:.3f}`")
                    st.caption(content_preview + "...")
                    if i < len(chunks):
                        st.divider()

    # Lưu vào session + memory
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": chunks,
    })
    st.session_state.memory.add(prompt, answer)
