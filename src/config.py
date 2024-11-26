import os
from dotenv import load_dotenv


load_dotenv()


class Config:
    # OpenAI API Key (from .env)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Document Processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # Storage
    VECTOR_DB_PATH = "./faiss_index"

    # Models
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
