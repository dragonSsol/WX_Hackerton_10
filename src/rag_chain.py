from langchain_core.prompts import load_prompt
from langchain_core.callbacks import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict
import logging
from langchain_teddynote import logging as langsmith_logging
from openai import OpenAI
from langsmith.wrappers import wrap_openai
import json


logger = logging.getLogger(__name__)


class RAGChain:
    def __init__(self):
        logger.info("RAG Chain 초기화 시작")
        self.prompt = load_prompt("prompt.yaml", encoding="utf-8")
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        # LangSmith 설정
        langsmith_logging.langsmith("hackerton")

        # OpenRouter 클라이언트 설정
        self.client = wrap_openai(
            OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key="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",
            )
        )

        logger.info("RAG Chain 초기화 완료")

    def format_docs(self, docs: List[Dict]) -> str:
        """검색된 문서들을 하나의 문자열로 포맷팅"""
        return "\n\n".join(doc.page_content for doc in docs)

    def get_openrouter_response(self, prompt: str) -> dict:
        """Get response through OpenRouter API"""
        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                extra_headers={"X-Title": "hackerton"},
                temperature=0,
                timeout=30,
                max_tokens=500,
            )
            content = response.choices[0].message.content

            try:
                # 1. 작은따옴표를 큰따옴표로 변경
                content = content.replace("'", '"')

                # 2. JSON 파싱 시도
                json_data = json.loads(content)
                return json_data

            except json.JSONDecodeError as je:
                # 3. 파싱 실패 시 문자열을 직접 파싱
                try:
                    # 문자열을 파이썬 딕셔너리로 변환
                    import ast

                    dict_data = ast.literal_eval(content)
                    return dict_data

                except Exception as e:
                    logger.error(f"String parsing error: {str(e)}")
                    return {
                        "error": f"Parsing error: {str(e)}",
                        "raw_response": content,
                    }

        except Exception as e:
            logger.error(f"OpenRouter API error: {str(e)}")
            return {"error": f"Error occurred: {str(e)}"}

    def run_rag_chain(self, question: str, retriever) -> str:
        """RAG 체인 실행"""
        try:
            context = self.format_docs(retriever.invoke(question))
            prompt_value = self.prompt.format(context=context, question=question)

            return self.get_openrouter_response(prompt_value)
        except Exception as e:
            logger.error(f"RAG 체인 오류: {str(e)}")
            return f"체인 실행 오류: {str(e)}"

    def analyze_documents(self, query: str, retriever) -> Dict:
        """문서 분석 실행"""
        try:
            logger.info(f"문서 분석 시작: {query[:100]}...")
            response = self.run_rag_chain(query, retriever)
            logger.info("문서 분석 완료")
            return {"query": query, "response": response, "status": "success"}
        except Exception as e:
            logger.error(f"문서 분석 중 오류 발생: {str(e)}")
            return {"query": query, "error": str(e), "status": "error"}
