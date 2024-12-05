from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from typing import List, Any


class Embedder:
    def __init__(
        self, model_type: str = "huggingface", model_name: str = "BAAI/bge-m3"
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.model_kwargs = {"device": "mps"}
        self.encode_kwargs = {"normalize_embeddings": True}

        if self.model_type == "huggingface":
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs=self.model_kwargs,
                encode_kwargs=self.encode_kwargs,
            )
        elif self.model_type == "openai":
            self.embeddings = OpenAIEmbeddings(model=self.model_name)
        else:
            raise ValueError(
                "지원하지 않는 모델 타입입니다. 'huggingface' 또는 'openai'를 사용하세요."
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 리스트를 임베딩합니다."""
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """쿼리 텍스트를 임베딩합니다."""
        return self.embeddings.embed_query(text)


"""
사용 방법
HuggingFace 모델 사용:
embedder = Embedder(model_type="huggingface", model_name="BAAI/bge-m3")
document_embeddings = embedder.embed_documents(["문서 내용"])

OpenAI 모델 사용:
embedder = Embedder(model_type="openai", model_name="text-embedding-3-large")
document_embeddings = embedder.embed_documents(["문서 내용"])
"""
