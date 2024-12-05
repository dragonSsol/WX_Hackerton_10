from datetime import datetime
import json
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self):
        self.cache = {}
        logger.info("캐시 매니저 초기화")

    def add_result(self, 
                  sentence_number: int, 
                  page_number: int, 
                  content: str, 
                  analysis: str) -> None:
        """분석 결과를 캐시에 추가"""
        try:
            self.cache[sentence_number] = {
                "sentence_number": sentence_number,
                "page_number": page_number,
                "original": content,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            logger.debug(f"캐시에 결과 추가: 문장 번호 {sentence_number}")
        except Exception as e:
            logger.error(f"캐시 추가 중 오류 발생: {str(e)}")

    def get_result(self, sentence_number: int) -> Dict[str, Any]:
        """캐시에서 결과 조회"""
        return self.cache.get(sentence_number)

    def export_results(self, file_path: str) -> None:
        """캐시 결과를 JSON 파일로 내보내기"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logger.info(f"캐시 결과 저장 완료: {file_path}")
        except Exception as e:
            logger.error(f"캐시 내보내기 중 오류 발생: {str(e)}") 