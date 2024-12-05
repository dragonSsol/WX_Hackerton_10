import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 프로젝트 메타데이터
__version__ = '0.1.0'
__author__ = 'Your Name'
__description__ = 'Contract Analysis System using RAG'

# 프로젝트 루트 디렉토리 경로
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 중요 경로들
PROMPT_PATH = os.path.join(PROJECT_ROOT, 'prompts')
CACHE_PATH = os.path.join(PROJECT_ROOT, 'cache')
LOG_PATH = os.path.join(PROJECT_ROOT, 'logs')

# 필요한 디렉토리 생성
for path in [PROMPT_PATH, CACHE_PATH, LOG_PATH]:
    if not os.path.exists(path):
        os.makedirs(path) 