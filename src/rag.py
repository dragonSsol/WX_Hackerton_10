from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from .vector_store import VectorStore  # 상대 경로로 수정
from .embedder import Embedder  # Embedder 추가
from .config import Config  # 상대 경로로 수정


class RAGChain:
    def __init__(self):
        self.llm = ChatOpenAI(model_name=Config.MODEL_NAME, temperature=0)
        embedder = Embedder()
        self.vector_store = VectorStore(embedder.embeddings)  # embeddings 전달
        self.setup_chain()

    def setup_chain(self):
        template = """
        다음 계약서를 검토하고 문제점과 대안을 제시해주세요.
        
        계약서 내용:
        {context}
        
        다음 형식으로 답변해주세요:
        1. 문제점: 계약서의 잠재적 문제점들을 나열
        2. 대안제시: 각 문제점에 대한 구체적인 해결방안 제시
        """

        self.prompt = ChatPromptTemplate.from_template(template)

        self.chain = (
            {"context": self.vector_store.get_relevant_documents}
            | self.prompt
            | self.llm
        )

    def analyze_document(self) -> dict:
        response = self.chain.invoke("계약서 검토를 시작합니다.")

        # 응답 파싱 및 결과 반환
        return {
            "issues": response.content.split("문제점:")[1]
            .split("대안제시:")[0]
            .strip(),
            "suggestions": response.content.split("대안제시:")[1].strip(),
        }
