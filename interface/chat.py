import streamlit as st
from utils.embedder import Embedder
from utils.vector_store import VectorStore
from utils.rag_chain import RAGChain

st.title("RAG 채팅")

# 채팅 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []


# RAG Chain 초기화
@st.cache_resource
def initialize_rag():
    embedder = Embedder()
    vector_store = VectorStore(embedder.embeddings)
    return RAGChain(vector_store)


rag_chain = initialize_rag()

# 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            response = rag_chain.run(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
