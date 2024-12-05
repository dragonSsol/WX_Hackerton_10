import sys
import os
import logging
from pathlib import Path
from datetime import datetime

#######
# 스크립트 실행 방법
# python scripts/create_vector_store.py --csv_path /path/to/your/data.csv
# python scripts/create_vector_store.py --csv_path /path/to/your/data.csv --output_dir vector_stores --model_type huggingface --model_name BAAI/bge-m3
# python scripts/create_vector_store.py --csv_path hackerton/data/poc.csv --output_dir vector_stores --model_type openai --model_name text-embedding-3-large
#
# 생성된 벡터 저장소는 다음과 같은 구조로 저장
# vector_stores/
# └── store_20240301_123456/
#     ├── faiss_store/
#     │   ├── index.faiss
#     │   └── index.pkl
#     └── metadata.json
#
# 이렇게 생성된 벡터 저장소는 나중에 API 서버에서 로드하여 사용할 수 있음.
#########

# 프로젝트 루트 경로를 Python 경로에 추가
project_root = str(Path(__file__).parents[1])
sys.path.append(project_root)

from src import DocumentProcessor, Embedder, VectorStore
from config import DEFAULT_CONFIG
import json

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VectorStoreCreator:
    def __init__(
        self, csv_path: str, output_dir: str, model_type: str, model_name: str = None
    ):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.model_type = model_type
        self.model_name = model_name or DEFAULT_CONFIG["EMBEDDING_MODEL"]

        # 컴포넌트 초기화
        self.document_processor = DocumentProcessor()
        self.embedder = Embedder(model_type=model_type, model_name=self.model_name)
        self.vector_store = VectorStore(self.embedder.embeddings)

        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

    def process(self):
        """CSV 파일을 처리하고 벡터 저장소 생성"""
        try:
            # 1. CSV 파일 로드
            logger.info(f"CSV 파일 로딩 시작: {self.csv_path}")
            documents = self.document_processor.load_csv(self.csv_path)
            logger.info(f"로드된 문서 수: {len(documents)}")

            # 2. 벡터 저장소 초기화 및 문서 저장
            logger.info(
                f"{self.model_type} 모델({self.model_name})을 사용하여 벡터 저장소 초기화 및 문서 임베딩 시작"
            )
            self.vector_store.initialize_store(documents)

            # 3. 벡터 저장소 저장
            store_path = os.path.join(self.output_dir, "faiss_store")
            self.vector_store.save_local(
                store_path, allow_dangerous_deserialization=True
            )
            logger.info(f"벡터 저장소 저장 완료: {store_path}")

            # 4. 메타데이터 저장
            metadata = {
                "document_count": len(documents),
                "embedding_model": self.model_name,
                "model_type": self.model_type,
                "created_at": datetime.now().isoformat(),
            }

            metadata_path = os.path.join(self.output_dir, "metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"메타데이터 저장 완료: {metadata_path}")

            return True

        except Exception as e:
            logger.error(f"처리 중 오류 발생: {str(e)}")
            raise


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="CSV 파일을 벡터 저장소로 변환")
    parser.add_argument("--csv_path", required=True, help="입력 CSV 파일 경로")
    parser.add_argument(
        "--output_dir", default="vector_stores", help="출력 디렉토리 경로"
    )
    parser.add_argument(
        "--model_type",
        choices=["huggingface", "openai"],
        default="huggingface",
        help="사용할 임베딩 모델 타입 (huggingface 또는 openai)",
    )
    parser.add_argument(
        "--model_name",
        help="사용할 임베딩 모델 이름 (huggingface: BAAI/bge-m3, openai: text-embedding-3-large)",
    )

    args = parser.parse_args()

    try:
        # 타임스탬프를 포함한 출력 디렉토리 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_short = (
            args.model_name.split("/")[-1] if args.model_name else "default"
        )
        output_dir = os.path.join(
            args.output_dir, f"store_{args.model_type}_{model_name_short}_{timestamp}"
        )

        # 벡터 저장소 생성
        logger.info(f"{args.model_type} 모델을 사용하여 벡터 저장소 생성 시작")
        creator = VectorStoreCreator(
            args.csv_path, output_dir, args.model_type, args.model_name
        )
        creator.process()

        logger.info("벡터 저장소 생성 완료")

    except Exception as e:
        logger.error(f"실행 중 오류 발생: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
