import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime


#streamlit run frontend/app.py
# API 엔드포인트 설정
API_URL = "http://localhost:5000"

st.title("계약서 분석 시스템")

# 파일 업로드 위젯
uploaded_file = st.file_uploader("계약서 PDF 파일을 업로드하세요", type=['pdf'])

if uploaded_file is not None:
    # 분석 시작 버튼
    if st.button("분석 시작"):
        with st.spinner("계약서 분석 중..."):
            try:
                # API 호출
                response = requests.post(
                    f"{API_URL}/analyze_contract",
                    files={"file": uploaded_file}
                )
                
                if response.status_code == 200:
                    results = response.json()
                    
                    # 메타데이터 표시
                    st.subheader("분석 정보")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("총 문장 수", results["metadata"]["total_sentences"])
                    with col2:
                        st.metric("임베딩 모델", results["metadata"]["model_info"]["embedding_model"])
                    with col3:
                        st.metric("LLM 모델", results["metadata"]["model_info"]["llm_model"])
                    
                    # 분석 결과를 데이터프레임으로 변환
                    analysis_data = []
                    for sentence_num, data in results["results"].items():
                        if "error" not in data:
                            analysis_dict = json.loads(data["analysis"].strip("```json\n").strip("\n```"))
                            analysis_data.append({
                                "문장번호": data["sentence_number"],
                                "페이지": data["page_number"],
                                "원문": data["content"],
                                "위반여부": analysis_dict["위반여부"],
                                "위반사유": analysis_dict["위반사유"],
                                "대안": analysis_dict["대안 제시"]
                            })
                    
                    df = pd.DataFrame(analysis_data)
                    
                    # 위반 사항만 필터링하는 체크박스
                    show_violations = st.checkbox("위반 사항만 보기")
                    if show_violations:
                        df = df[df["위반여부"] == "위반"]
                    
                    # 결과 테이블 표시
                    st.subheader("분석 결과")
                    st.dataframe(
                        df,
                        column_config={
                            "문장번호": st.column_config.NumberColumn(width=80),
                            "페이지": st.column_config.NumberColumn(width=80),
                            "원문": st.column_config.TextColumn(width=300),
                            "위반여부": st.column_config.TextColumn(width=100),
                            "위반사유": st.column_config.TextColumn(width=300),
                            "대안": st.column_config.TextColumn(width=300)
                        }
                    )
                    
                    # 결과 다운로드 버튼
                    csv = df.to_csv(index=False).encode('utf-8-sig')
                    filename = f"계약서분석결과_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    st.download_button(
                        label="CSV 다운로드",
                        data=csv,
                        file_name=filename,
                        mime="text/csv"
                    )
                    
                else:
                    st.error(f"API 오류: {response.json().get('error', '알 수 없는 오류')}")
                    
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")

# 캐시된 결과 조회 섹션
st.subheader("이전 분석 결과 조회")
sentence_number = st.number_input("문장 번호", min_value=1, value=1)
if st.button("결과 조회"):
    try:
        response = requests.get(f"{API_URL}/get_cached_result/{sentence_number}")
        if response.status_code == 200:
            cached_data = response.json()
            st.json(cached_data)
        else:
            st.warning("해당 문장의 분석 결과를 찾을 수 없습니다.")
    except Exception as e:
        st.error(f"조회 중 오류 발생: {str(e)}") 