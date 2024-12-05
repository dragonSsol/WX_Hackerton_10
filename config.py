import os
from dotenv import load_dotenv
import logging.config

# 환경 변수 로드
load_dotenv()

# 기본 설정
DEFAULT_CONFIG = {
    "MODEL_NAME": "gpt-4",
    "EMBEDDING_MODEL": "BAAI/bge-m3",
    "CACHE_DIR": "cache",
    "LOG_LEVEL": "INFO",
    "RETRIEVER_K": 4,
    "OPENAI_MODEL": "text-embedding-3-large",
}

# 로깅 설정
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "app.log",
            "formatter": "standard",
            "level": "DEBUG",
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
}

# 로깅 설정 적용
logging.config.dictConfig(LOGGING_CONFIG)

# API 설정
API_CONFIG = {
    "HOST": "0.0.0.0",
    "PORT": 5003,
    "VECTOR_STORE_API_PORT": 5001,  # 벡터 스토어 API용 포트
    "DEBUG": False,
}

# LangSmith 설정
LANGSMITH_CONFIG = {
    "API_KEY": os.getenv("LANGCHAIN_API_KEY"),
    "PROJECT": "hackerton",
}
