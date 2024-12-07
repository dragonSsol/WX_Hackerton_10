from typing import List
from langchain_community.document_loaders import PDFPlumberLoader, CSVLoader
from langchain_core.documents import Document

import logging
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self):
        pass

    def load_pdf(self, file_path: str) -> List[Document]:
        """PDF 파일을 LlamaIndex를 사용하여 로드하고 특정 섹션만 추출합니다."""
        # 파일 추출기 설정

        loader = PDFPlumberLoader(file_path)
        docs = loader.load()

        filtered_docs = []
        is_target_section = False
        found_first_match = False  # 첫 번째 매칭(목차) 확인용 플래그

        # 디버깅을 위한 출력 추가
        print(f"총 {len(docs)}개의 페이지를 로드했습니다.")

        for doc_idx, doc in enumerate(docs):

            # 연속된 스페이스를 하나로 통일
            doc.page_content = " ".join(doc.page_content.split())

            lines = doc.page_content.split("\n")
            filtered_content = []

            for line_idx, line in enumerate(lines):
                clean_line = line.strip()

                if any(
                    marker in clean_line
                    for marker in ["6.0 특기", "6.0특기", "6. 특기"]
                ):
                    if not found_first_match:
                        # 첫 번째 매칭(목차)는 건너뜀
                        found_first_match = True
                        print(f">>> 목차 발견 (건너뜀): {clean_line}")
                        continue
                    print(f">>> 시작 지점 발견: {clean_line}")
                    is_target_section = True
                elif clean_line.startswith("7."):
                    print(f">>> 종료 지점 발견: {clean_line}")
                    is_target_section = False
                    break

                if is_target_section:
                    filtered_content.append(line)

            if filtered_content:
                # 원본 메타데이터를 유지하면서 새로운 Document 생성
                # page 번호는 0-based이므로 1을 더해 실제 페이지 번호로 변환
                metadata = doc.metadata.copy()
                metadata["page_number"] = (
                    metadata.get("page", 0) + 1
                )  # 실제 페이지 번호

                filtered_doc = Document(
                    page_content="\n".join(filtered_content), metadata=metadata
                )
                filtered_docs.append(filtered_doc)

        # 결과 확인을 위한 출력
        print(f"\n추출된 문서 수: {len(filtered_docs)}")
        if filtered_docs:
            print("\n첫 번째 추출 문서의 시작 부분:")
            print(f"페이지 번호: {filtered_docs[0].metadata['page_number']}")
            print(filtered_docs[0].page_content[:200])
        else:
            print("\n추출된 문서가 없습니다!")

        return filtered_docs

    def load_csv(self, file_path: str) -> List[Document]:
        """CSV 파일을 로드합니다."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {file_path}")

            logger.info(f"파일 경로: {os.path.abspath(file_path)}")

            # 파일 내용 �리보기
            with open(file_path, "rb") as f:
                preview = f.read(1024)
                logger.info(f"파일 미리보기: {preview[:100]}")

            loader = CSVLoader(file_path)
            documents = loader.load()
            return documents

        except Exception as e:
            logger.error(f"CSV 로드 에러: {str(e)}")
            logger.error(f"에러 타입: {type(e)}")
            import traceback

            logger.error(f"스택 트레이스: {traceback.format_exc()}")
            raise

    def show_metadata(self, docs: List[Document]) -> None:
        """문서의 메타데이터를 출력합니다."""
        if not docs:
            return

        print("[metadata]")
        print(list(docs[0].metadata.keys()))
        print("\n[examples]")
        max_key_length = max(len(k) for k in docs[0].metadata.keys())
        for k, v in docs[0].metadata.items():
            print(f"{k:<{max_key_length}} : {v}")
