import streamlit as st
from pathlib import Path
import os
from src import Config, DocumentProcessor, RAGChain


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False


def create_sidebar():
    with st.sidebar:
        st.title("계약서 검토 시스템")
        uploaded_file = st.file_uploader(
            "계약서 업로드",
            type=["pdf", "docx", "txt"],
            help="PDF, DOCX, TXT 파일을 업로드해주세요.",
        )
        return uploaded_file


def main():
    init_session_state()

    # 사이드바 생성
    uploaded_file = create_sidebar()

    # 메인 화면 구성
    st.title("계약서 검토 문서")

    # 탭 구성
    tab1, tab2 = st.tabs(["계약서 불러오기", "계약서 검토시작"])

    with tab1:
        if uploaded_file:
            # 문서 처리
            doc_processor = DocumentProcessor()
            documents = doc_processor.process_document(uploaded_file)

            if documents:
                st.session_state.document_processed = True
                st.success("문서 처리가 완료되었습니다!")

                # 문서 내용 표시
                st.markdown("### 문서 내용")
                st.write(documents[0].page_content[:500] + "...")

    with tab2:
        if not st.session_state.document_processed:
            st.warning("먼저 계약서를 업로드해주세요.")
            return

        # 검토 영역 구성
        st.markdown("### 문제점 및 대안제시")

        # RAG Chain 초기화
        rag_chain = RAGChain()

        # 검토 버튼
        if st.button("계약서 검토 시작"):
            with st.spinner("계약서를 검토중입니다..."):
                # 문제점 분석
                analysis = rag_chain.analyze_document()

                # 결과 표시
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 문제점")
                    st.write(analysis.get("issues", "문제점이 발견되지 않았습니다."))

                    st.markdown("#### 대안제시")
                    st.write(analysis.get("suggestions", "대안이 필요하지 않습니다."))

                with col2:
                    st.markdown("#### 문제점")
                    st.write(
                        "이자율: 일일 0.03%의 이자율은 연간 약 10.95%에 해당합니다. 이는 상당히 높은 이자율로, 차후 지연시 큰 재정적 부담이 될 수 있습니다."
                    )

                    st.markdown("#### 대안제시")
                    st.write(
                        "이자율 조정: 일일 이자율을 낮추거나, 연간 이자율로 변경하여 재정적 부담을 줄일 수 있습니다. 예를 들어, 일일 0.01% 또는 연간 5%로 조정할 수 있습니다."
                    )


if __name__ == "__main__":
    main()
