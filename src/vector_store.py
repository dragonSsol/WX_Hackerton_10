import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from typing import List, Tuple
from langchain_core.documents import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.retriever import BaseRetriever
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.store = None
        logger.info("벡터 스토어 초기화")

    def initialize_store(self, documents: List[Dict]):
        """문서로 벡터 스토어 초기화"""
        try:
            self.store = FAISS.from_documents(documents, self.embeddings)
            logger.info(f"{len(documents)}개 문서로 벡터 스토어 초기화 완료")
        except Exception as e:
            logger.error(f"벡터 스토어 초기화 중 오류: {str(e)}")
            raise

    def save_local(self, path: str):
        """로컬에 벡터 스토어 저장"""
        if self.store:
            self.store.save_local(path)
            logger.info(f"벡터 스토어 저장 완료: {path}")
        else:
            raise ValueError("초기화되지 않은 벡터 스토어는 저장할 수 없습니다")

    def load_local(self, path: str):
        """로컬에서 벡터 스토어 로드"""
        try:
            self.store = FAISS.load_local(
                path, self.embeddings, allow_dangerous_deserialization=True
            )
            logger.info(f"벡터 스토어 로드 완료: {path}")
        except Exception as e:
            logger.error(f"벡터 스토어 로드 중 오류: {str(e)}")
            raise

    def get_retriever(self, search_kwargs: Optional[Dict] = None) -> BaseRetriever:
        """벡터 스토어의 retriever 반환"""
        if not self.store:
            raise ValueError("벡터 스토어가 초기화되지 않았습니다")

        search_kwargs = search_kwargs or {"k": 4}
        return self.store.as_retriever(search_kwargs=search_kwargs)

    def similarity_search(self, query: str, k: int = 4) -> List[Dict]:
        """유사도 검색 수행"""
        if not self.store:
            raise ValueError("벡터 스토어가 초기화되지 않았습니다")
        return self.store.similarity_search(query, k=k)
