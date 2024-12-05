from flask import Flask
from config import API_CONFIG
import logging

logger = logging.getLogger(__name__)

def create_app():
    """Flask 애플리케이션 팩토리"""
    app = Flask(__name__)
    
    # 설정 로드
    app.config.from_object(API_CONFIG)
    
    # 로깅 설정
    if not app.debug:
        logger.info("프로덕션 모드로 API 서버 시작")
    else:
        logger.info("디버그 모드로 API 서버 시작")
    
    return app

__all__ = ['create_app'] 