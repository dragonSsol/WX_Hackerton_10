from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from typing import List
import numpy as np
from .config import Config


class Embedder:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,  # OpenAI의 최신 임베딩 모델
            dimensions=1536,  # 모델의 차원 수
        )

    def embed_documents(self, documents: List[Document]) -> List[np.ndarray]:
        texts = [doc.page_content for doc in documents]
        return self.embeddings.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        return self.embeddings.embed_query(query)
