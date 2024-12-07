from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import logging
import os, sys
from pathlib import Path
import re
import json

# 프로젝트 루트 경로를 Python 경로에 추가
project_root = str(Path(__file__).parents[1])
sys.path.append(project_root)

from src import (
    DocumentProcessor,
    Embedder,
    VectorStore,
    KoreanSentenceSplitter,
    RAGChain,
    CacheManager,
)

from config import API_CONFIG, DEFAULT_CONFIG
import tempfile

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.json.ensure_ascii = False
CORS(app)  # 모든 도메인에서의 요청 허용


class ContractAnalyzer:
    def __init__(self, vector_stores_path: str):
        """
        계약서 분석기 초기화
        Args:
            vector_stores_path: FAISS 벡터 저장소들이 있는 디렉토리 경로
        """
        self.document_processor = DocumentProcessor()
        self.text_splitter = KoreanSentenceSplitter()
        self.rag_chain = RAGChain()
        self.cache_manager = CacheManager()

        # 벡터 스토어 초기화
        self.vector_stores = {}
        self.load_vector_stores(vector_stores_path)

        logger.info("계약서 분석기 초기화 완료")

    def load_vector_stores(self, vector_stores_path: str):
        """가장 최근 벡터 스토어만 로드"""
        try:
            logger.info(f"벡터 스토어 로드 시작: {vector_stores_path}")

            # store_ 로 시작하는 모든 디렉토리 찾기
            store_dirs = []
            for d in Path(vector_stores_path).iterdir():
                if d.is_dir() and d.name.startswith("store_"):
                    try:
                        # store_modeltype_modelname_YYYYMMDD_HHMMSS 형식 파싱
                        timestamp_str = (
                            d.name.split("_")[-2] + "_" + d.name.split("_")[-1]
                        )
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        store_dirs.append((d, timestamp))
                    except (IndexError, ValueError):
                        logger.warning(f"잘못된 방식의 벡터 스토어 디렉토리: {d.name}")
                        continue

            if not store_dirs:
                raise ValueError(
                    f"사용 가능한 벡터 스토어가 없습니다: {vector_stores_path}"
                )

            # 타임스탬프 기준으로 정렬하고 가장 최근 것만 선택
            latest_store_dir, latest_timestamp = max(store_dirs, key=lambda x: x[1])

            # 메타데이터 로드
            metadata_path = latest_store_dir / "metadata.json"
            if not metadata_path.exists():
                raise ValueError(f"메타데이터 파일이 없습니다: {metadata_path}")

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # 임베더 및 벡터 스토어 초기화
            embedder = Embedder(
                model_type=metadata["model_type"],
                model_name=metadata["embedding_model"],
            )
            vector_store = VectorStore(embedder.embeddings)

            # 벡터 스토어 로드
            store_path = latest_store_dir / "faiss_store"
            if not store_path.exists():
                raise ValueError(f"벡터 스토어 파일이 없습니다: {store_path}")

            vector_store.load_local(str(store_path))
            self.vector_stores[latest_store_dir.name] = {
                "store": vector_store,
                "metadata": metadata,
                "retriever": vector_store.get_retriever(
                    search_kwargs={"k": DEFAULT_CONFIG.get("RETRIEVER_K", 4)}
                ),
                "created_at": latest_timestamp.isoformat(),
            }

            logger.info(
                f"가장 최근 벡터 스토어 로드 완료: {latest_store_dir.name} (생성일시: {latest_timestamp})"
            )

        except Exception as e:
            logger.error(f"벡터 스토어 로드 중 오류: {str(e)}")
            raise

    def analyze_contract(self, pdf_file, vector_store_id: str = None) -> dict:
        """
        계약서 PDF 파일 분석
        Args:
            pdf_file: PDF 파일 객체
            vector_store_id: 사용할 벡터 스토어 ID (없으면 가장 최근 것 사용)
        Returns:
            분석 결과 딕셔너리
        """
        try:
            # 벡터 스토어 선택
            if not vector_store_id:
                # 생성 시간 기준으로 가장 최근 벡터 스토어 선택
                try:
                    vector_store_id = max(
                        self.vector_stores.keys(),
                        key=lambda k: datetime.fromisoformat(
                            self.vector_stores[k]["created_at"]
                        ),
                    )
                    logger.info(f"가장 최근 벡터 스토어 선택: {vector_store_id}")
                    logger.info(
                        f"생성 시간: {self.vector_stores[vector_store_id]['created_at']}"
                    )
                except ValueError as e:
                    logger.error(f"벡터 스토어 선택 중 오류: {str(e)}")
                    raise ValueError("사용 가능한 벡터 스토어가 없습니다.")
            else:
                logger.info(f"지정된 벡터 스토어 사용: {vector_store_id}")

            if vector_store_id not in self.vector_stores:
                available_stores = list(self.vector_stores.keys())
                raise ValueError(
                    f"벡터 스토어를 찾을 수 없습니다: {vector_store_id}\n"
                    f"사용 가능한 벡터 스토어: {available_stores}"
                )

            selected_store = self.vector_stores[vector_store_id]
            retriever = selected_store["retriever"]

            logger.info(f"선택된 벡터 스토어 정보:")
            logger.info(f"- ID: {vector_store_id}")
            logger.info(f"- 모델: {selected_store['metadata']['embedding_model']}")
            logger.info(f"- 생성일시: {selected_store['created_at']}")

            # PDF를 임시 파일로 저장
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                pdf_file.save(tmp_file.name)

                # 1. PDF 문서 로드
                logger.info("PDF 문서 로드 시작")
                docs = self.document_processor.load_pdf(tmp_file.name)

            # 임시 파일 삭제
            os.unlink(tmp_file.name)

            # 2. 문장 분할
            # logger.info("문서 문장 분할 시작")
            # split_docs = self.text_splitter.split_documents(docs)

            logger.info("문서 넘버링으로 분할 시작")
            split_docs = self.text_splitter.split_by_numbering(docs)

            # 문장 내의 불필요한 따옴표 제거
            for doc in split_docs:
                doc["content"] = doc["content"].replace('"', "").replace("'", "")

            logger.info(f"분할된 문장 수: {len(split_docs)}")

            # 문서 분석 실행
            analysis_results = {}
            for doc in split_docs:
                try:
                    # 문서 분석 실행
                    result = self.rag_chain.analyze_documents(doc.page_content, retriever)
                    logger.info(f"########분석 결과: {result}")
                    
                    # response 데이터 추출
                    response_data = result.get('response', {})
                    logger.info(f"########응답: {response_data}")
                    
                    # 위반 여부 확인
                    violation_status = response_data.get('detection_flag','N')
                    logger.info(f"########응답 데이터1: {violation_status}")
                    
                    if violation_status == "Y":
                        try:
                            # 위반여부가 Y인 경우에만 결과 저장
                            result_data = {
                                "section_number": result.get('section_number'),  #분석 결과에서 section_number 가져오기
                                "page_number": doc.metadata.get("page_number",1),
                                "content": doc.page_content,
                                "analysis": response_data,
                                "timestamp": datetime.now().isoformat(),
                            }
                            
                            # section_number가 있는 경우에만 저장
                            if result_data["section_number"] is not None:
                                analysis_results[result_data["section_number"]] = result_data
                                logger.info(f"위반사항 발견: 섹션 {result_data['section_number']}")
                            else:
                                logger.warning("섹션 번호가 없는위반사항 발견")
                                
                        except Exception as e:
                            logger.error(f"결과 저장 중 오류: {str(e)}")
                            logger.error(f"문서 데이터: {doc.metadata}")
                            continue

                except Exception as e:
                    logger.error(f"문서 분석 중 오류:{str(e)}")
                    continue

            return {
                "total_sections": len(split_docs),
                "violation_count": len(analysis_results),
                "violations": analysis_results
            }

        except Exception as e:
            logger.error(f"계약서 분석 중 오류 발생: {str(e)}")
            raise


# 글로벌 분석기 인스턴스
analyzer = None


def initialize_analyzer():
    """서버 시작 시 분석기 초기화"""
    global analyzer
    vector_stores_path = os.getenv("VECTOR_STORES_PATH", "vector_stores")
    analyzer = ContractAnalyzer(vector_stores_path)


# Flask 2.3.0+ 방식으로 초기화
with app.app_context():
    initialize_analyzer()


@app.route("/health", methods=["GET"])
def health_check():
    """헬스 체크 엔드포인트"""
    return jsonify({"status": "healthy"})


@app.route("/vector_stores", methods=["GET"])
def list_vector_stores():
    """사용 가능한 벡터 스토어 목록 조회"""
    stores = {
        store_id: {
            "model_type": info["metadata"]["model_type"],
            "embedding_model": info["metadata"]["embedding_model"],
            "document_count": info["metadata"]["document_count"],
            "created_at": info["metadata"]["created_at"],
        }
        for store_id, info in analyzer.vector_stores.items()
    }
    return jsonify(stores)


@app.route("/analyze_contract", methods=["POST"])
def analyze_contract():
    """계약서 분석 엔드포인트"""
    try:
        if "file" not in request.files:
            return jsonify({"error": "PDF 파일이 필요합니다"}), 400

        file = request.files["file"]
        if not file.filename.endswith(".pdf"):
            return jsonify({"error": "PDF 파일만 지원됩니다"}), 400

        # 벡터 스토어 ID 가져오기 (선택사항)
        vector_store_id = request.form.get("vector_store_id")

        # 계약서 분석 실행
        results = analyzer.analyze_contract(file, vector_store_id)

        return jsonify(results)

    except Exception as e:
        logger.error(f"API 요청 처리 중 오류 발생: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/get_cached_result/<int:section_number>", methods=["GET"])
def get_cached_result(section_number: int):
    """캐시된 분석 결과 조회"""
    result = analyzer.cache_manager.get_result(section_number)
    if result:
        return jsonify(result)
    return jsonify({"error": "결과를 찾을 수 없습니다"}), 404


if __name__ == "__main__":
    app.run(host=API_CONFIG["HOST"], port=API_CONFIG["PORT"], debug=API_CONFIG["DEBUG"])

"""
API 사용 예시:

# 벡터 스토어 목록 조회
curl http://localhost:5002/vector_stores

# 특정 벡터 스토어로 분석
curl -X POST \
  -F "file=@contract.pdf" \
  -F "vector_store_id=store_huggingface_bge-m3_20240301_123456" \
  http://localhost:5002/analyze_contract
  
  
 curl -X POST \
  -F "file=@data/contract_test.pdf" \
  http://localhost:5002/analyze_contract
"""
