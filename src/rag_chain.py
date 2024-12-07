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
from dotenv import load_dotenv
import os


logger = logging.getLogger(__name__)


class RAGChain:
    def __init__(self):
        logger.info("RAG Chain 초기화 시작")
        self.prompt = load_prompt("prompt.yaml", encoding="utf-8")
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        # LangSmith 설정
        langsmith_logging.langsmith("hackerton")

        # 환경 변수 로드
        load_dotenv()
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        openrouter_api_base = os.getenv("OPENROUTER_API_BASE")
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY가 환경 변수에 설정되지 않았습니다.")

        # OpenRouter 클라이언트 설정
        self.client = wrap_openai(
            OpenAI(
                base_url=openrouter_api_base,
                api_key=openrouter_api_key,
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

    def normalize_text(self, text: str) -> str:
        """GPT-4o mini를 사용하여 텍스트 정규화"""
        try:
            prompt = """당신은 한국어 문장 교정 전문가입니다. 
                        주어진 문장을 읽기 쉽도록 적절한 띄어쓰기와 문장 부호를 교정해주세요.
                        원래 문장의 의미는 그대로 유지하면서 띄어쓰기만 교정하여 한 문장으로 출력해주세요.
                        다른 설명은 하지 말고 교정된 문장만 출력해주세요.

                        입력 문장: {text}"""

            response = self.client.chat.completions.create(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": prompt.format(text=text)}],
                extra_headers={"X-Title": "hackerton"},
                temperature=0,
                timeout=30,
                max_tokens=200,
            )

            normalized_text = response.choices[0].message.content.strip()
            logger.info(f"원본 문장: {text}")
            logger.info(f"정규화된 문장: {normalized_text}")

            return normalized_text

        except Exception as e:
            logger.error(f"텍스트 정규화 중 오류: {str(e)}")
            return text  # 오류 발생 시 원본 텍스트 반환

    def run_rag_chain(self, question: str, retriever) -> str:
        """RAG 체인 실행"""
        try:
            context = self.format_docs(retriever.invoke(question))
            prompt_value = self.prompt.format(context=context, question=question)
            # 질문 텍스트 정규화
            #normalized_question = self.normalize_text(question)

            # 정규화된 텍스트로 검색 수행
            #context = self.format_docs(retriever.invoke(normalized_question))
            #prompt_value = self.prompt.format(
             #   context=context, question=normalized_question
            #)

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
