from langchain.vectorstores import FAISS
from langchain.schema import Document
from typing import List, Optional
import os


class VectorStore:
    def __init__(self, embedding_function, index_name: str = "document_store"):
        self.embedding_function = embedding_function
        self.index_name = index_name
        self.store_path = f"./faiss_index/{index_name}"

        # 기존 인덱스가 있으면 로드, 없으면 새로 생성
        if os.path.exists(f"{self.store_path}.faiss"):
            self.db = FAISS.load_local(self.store_path, self.embedding_function)
        else:
            self.db = FAISS.from_documents([], self.embedding_function)

    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None):
        if not documents:
            return

        # 문서 추가
        self.db.add_documents(documents)

        # 인덱스 저장
        os.makedirs("./faiss_index", exist_ok=True)
        self.db.save_local(self.store_path)

    def similarity_search_with_score(self, query: str, k: int = 4):
        return self.db.similarity_search_with_score(query, k=k)

    def get_relevant_documents(self, query: str, k: int = 4):
        return self.db.similarity_search(query, k=k)
