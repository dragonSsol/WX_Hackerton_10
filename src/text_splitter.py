from langchain.text_splitter import TextSplitter
from typing import List, Dict
import kss
import re
import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class KoreanSentenceSplitter(TextSplitter):
    def __init__(self):
        super().__init__()
        logger.info("한국어 문장 분할기 초기화")

    def clean_text(self, text: str) -> str:
        """텍스트 정리를 위한 헬퍼 함수"""
        text = " ".join(text.split())
        return text.strip()

    def split_documents(self, documents) -> List[Dict]:
        """문서를 문장 단위로 분할하고 메타데이터를 포함하여 반환"""
        logger.debug(f"문서 분할 시작: {len(documents)}개 문서")
        result = []
        section_number = 1

        for doc in documents:
            try:
                page_number = doc.metadata.get("page", 0)
                sentences = self.split_text(doc.page_content)

                for sentence in sentences:
                    if sentence.strip():
                        result.append(
                            {
                                "content": sentence.strip(),
                                "page_number": page_number + 1,
                                "section_number": section_number,
                                "metadata": doc.metadata,
                            }
                        )
                        section_number += 1
            except Exception as e:
                logger.error(f"문서 분할 중 오류 발생: {str(e)}")
                continue

        logger.info(f"문서 분할 완료: {len(result)}개 문장 생성")
        return result

    def split_text(self, text: str) -> List[str]:
        """텍스트를 문장 단위로 분할"""
        logger.debug("텍스트 분할 시작")

        try:
            # 줄바꿈을 스페이스로 대체
            text = text.replace("\n", " ")

            # 연속된 스페이스를 하나로 통일
            text = " ".join(text.split())

            sentences = []

            # 큰따옴표로 둘러싸인 부분을 임시 마커로 대체
            quote_parts = []

            def quote_replacer(match):
                quote_parts.append(match.group(0))
                return f"QUOTE_{len(quote_parts)-1}_QUOTE"

            # 큰따옴표 내용 임시 저장
            processed_text = re.sub(r'"[^"]+?"', quote_replacer, text)

            # KSS로 기본 문장 분할
            current_sentences = kss.split_sentences(processed_text.strip())

            for sent in current_sentences:
                # 1. 숫자.숫자 다음에 한글이 오는 경우
                # 2. "다." 다음에 숫자.숫자가 오는 경우
                parts = re.split(r"(?<=다\.)\s*(?=\d+\.\d+)", sent)

                for part in parts:
                    if not part.strip():
                        continue

                    # 임시 마커를 원래 큰따옴표 내용으로 복원
                    restored_part = re.sub(
                        r"QUOTE_(\d+)_QUOTE",
                        lambda m: quote_parts[int(m.group(1))],
                        part,
                    )

                    # 숫자.숫자 패턴이 단독으로 있는 경우 다음 문장과 병합
                    if re.match(r"^\d+\.\d+$", restored_part.strip()):
                        if sentences:
                            sentences[-1] = f"{sentences[-1]} {restored_part}"
                        else:
                            sentences.append(restored_part)
                    else:
                        sentences.append(restored_part.strip())

            logger.debug(f"텍스트 분할 완료: {len(sentences)}개 문장 생성")
            return sentences

        except Exception as e:
            logger.error(f"텍스트 분할 중 오류 발생: {str(e)}")
            # 오류 발생 시 원본 텍스트를 하나의 문장으로 반환
            if text.strip():
                return [text.strip()]
            return []

    def split_by_numbering(self, documents: List[Document]) -> List[Document]:
        """문서를 번호 매기기 방식으로 분할"""
        split_docs = []
        for doc in documents:
            content = doc.page_content
            # 번호 매기기 패턴으로 분할
            sections = re.split(r'(\d+[\)）\.])\s*', content)
            
            current_number = 0
            for i in range(1, len(sections), 2):
                if i+1 < len(sections):
                    number_part = sections[i].strip('.)）')
                    text_part = sections[i+1].strip()
                    
                    if text_part:  # 빈 텍스트가 아닌 경우만 처리
                        current_number += 1
                        # 메타데이터에 section_number와 page_number추가
                        metadata = {
                            "section_number": current_number,
                            "page_number": doc.metadata.get("page", 1),
                            "source": doc.metadata.get("source", ""),
                        }
                        
                        split_docs.append(Document(
                            page_content=text_part,
                            metadata=metadata
                        ))
        
        logger.info(f"문서 분할 완료: {len(split_docs)}개 섹션 생성")
        return split_docs
