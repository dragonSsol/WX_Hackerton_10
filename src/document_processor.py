from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import pdfplumber
import tempfile
import os


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def process_document(self, file) -> List[Document]:
        # 임시 파일로 저장 후 처리
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            text_content = ""
            with pdfplumber.open(tmp_file_path) as pdf:
                for page in pdf.pages:
                    text_content += page.extract_text() + "\n"

            # Document 객체 생성
            metadata = {"source": file.name}
            document = Document(page_content=text_content, metadata=metadata)

            # 문서 청킹
            return self.split_documents([document])

        finally:
            os.unlink(tmp_file_path)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(documents)
