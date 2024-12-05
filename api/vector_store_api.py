from flask import Flask, request, jsonify
from datetime import datetime
import logging
import os
from pathlib import Path
import tempfile
import json
import sys

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 프로젝트 루트 경로를 Python 경로에 추가
project_root = str(Path(__file__).parents[1])
sys.path.append(project_root)

from src import DocumentProcessor, Embedder, VectorStore
from config import API_CONFIG, DEFAULT_CONFIG


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
            self.vector_store.save_local(store_path)
            logger.info(f"벡터 스토어 저장 완료: {store_path}")

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


@app.route("/health", methods=["GET"])
def health_check():
    """헬스 체크 엔드포인트"""
    return jsonify({"status": "healthy"})


@app.route("/create_vector_store", methods=["POST"])
def create_vector_store():
    """CSV 파일을 업로드하여 벡터 스토어 생성"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "CSV 파일이 필요합니다"}), 400

        file = request.files["file"]
        if not file.filename.endswith(".csv"):
            return jsonify({"error": "CSV 파일만 지원됩니다"}), 400

        model_type = request.form.get("model_type", "huggingface")
        model_name = request.form.get("model_name", DEFAULT_CONFIG["EMBEDDING_MODEL"])

        # 임시 파일로 CSV 저장
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
            file.save(tmp_file.name)
            csv_path = tmp_file.name

        try:
            # 벡터 스토어 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name_short = model_name.split("/")[-1] if model_name else "default"
            output_dir = os.path.join(
                "vector_stores", f"store_{model_type}_{model_name_short}_{timestamp}"
            )

            creator = VectorStoreCreator(csv_path, output_dir, model_type, model_name)
            creator.process()

            return jsonify(
                {
                    "status": "success",
                    "output_dir": output_dir,
                    "model_type": model_type,
                    "model_name": model_name,
                }
            )

        finally:
            # 임시 파일 삭제
            os.unlink(csv_path)

    except Exception as e:
        logger.error(f"벡터 스토어 생성 중 오류 발생: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/list_vector_stores", methods=["GET"])
def list_vector_stores():
    """생성된 벡터 스토어 목록 조회"""
    try:
        vector_stores_path = Path("vector_stores")
        if not vector_stores_path.exists():
            return jsonify({"vector_stores": []})

        stores = []
        for store_dir in vector_stores_path.iterdir():
            if store_dir.is_dir() and store_dir.name.startswith("store_"):
                metadata_path = store_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                        stores.append(
                            {"store_id": store_dir.name, "metadata": metadata}
                        )

        return jsonify({"vector_stores": stores})

    except Exception as e:
        logger.error(f"벡터 스토어 목록 조회 중 오류 발생: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # 기본 포트와 다르게 설정 (메인 API와 충돌 방지)
    app.run(
        host=API_CONFIG.get("HOST", "0.0.0.0"),
        port=API_CONFIG.get("VECTOR_STORE_API_PORT", 5001),
        debug=API_CONFIG.get("DEBUG", False),
    )


"""
API 사용 예시:
벡터 스토어 생성:

curl -X POST \
  -F "file=@data/poc.csv" \
  -F "model_type=openai" \
  -F "model_name=text-embedding-3-large" \
  http://localhost:5001/create_vector_store
  
 
 curl -X POST \
  -F "file=@data/poc.csv" \
  -F "model_type=huggingface" \
  -F "model_name=BAAI/bge-m3" \
  http://localhost:5001/create_vector_store 
  
  
  
  벡터 스토어 목록 조회:
  curl http://localhost:5001/list_vector_stores
  
  헬스 체크:
  curl http://localhost:5001/health
  
   별도의 포트(5001)에서 벡터 스토어 생성 API가 실행되며, 기존 메인 API(5000)와 독립적으로 동작.
실행 방법:
  # 벡터 스토어 API 실행
python api/vector_store_api.py
"""
