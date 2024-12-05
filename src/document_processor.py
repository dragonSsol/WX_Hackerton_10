from typing import List
from langchain_community.document_loaders import PDFPlumberLoader, CSVLoader
from langchain_core.documents import Document


class DocumentProcessor:
    def __init__(self):
        pass

    def load_pdf(self, file_path: str) -> List[Document]:
        """PDF 파일을 로드합니다."""
        loader = PDFPlumberLoader(file_path)
        return loader.load()

    def load_csv(self, file_path: str) -> List[Document]:
        """CSV 파일을 로드합니다."""
        loader = CSVLoader(file_path)
        return loader.load()

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
