import streamlit as st
from utils.document_processor import DocumentProcessor
from utils.embedder import Embedder
from utils.vector_store import VectorStore
import uuid

st.title("문서 업로드 및 처리")

# 세션 상태 초기화
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

uploaded_files = st.file_uploader(
    "처리할 문서를 업로드하세요", type=["pdf", "txt"], accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        if file.name not in st.session_state.processed_files:
            with st.spinner(f"{file.name} 처리 중..."):
                try:
                    # 문서 처리 파이프라인
                    processor = DocumentProcessor()
                    embedder = Embedder()
                    vector_store = VectorStore(embedder.embeddings)

                    # 문서 로드 및 청킹
                    documents = processor.load_document(file)
                    chunks = processor.split_documents(documents)

                    # Vector DB에 저장
                    ids = [str(uuid.uuid4()) for _ in chunks]
                    vector_store.add_documents(chunks, ids)

                    st.session_state.processed_files.add(file.name)
                    st.success(f"{file.name} 처리 완료!")
                except Exception as e:
                    st.error(f"{file.name} 처리 중 오류 발생: {str(e)}")
