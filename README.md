1. 업로드한 계약 문서 위반 문장 검출 서버 실행 방법
  python api/main.py

2. 검출 API 호출 방법
- POST 방식 호출
- API URL : http://localhost:5003/analyze_contract
- parameter : file
              ( test file path : data/contract_test.pdf)
- 응답값 예시 : 
{"metadata":
	{"processed_at":"2024-12-05T23:06:37.010133","total_sentences":48,"vector_store":{"embedding_model":"text-embedding-3-large","id":"store_openai_text-embedding-3-large_20241204_001024","model_type":"openai"}
,"violation_count":15}
,"results":
{"3":{"analysis":
      {"asis_sentence": 원문 문장, "detection_flag": Y/N, "comments": 위반문장일때 위반인 이유/검토의견, 'tobe_sentence': 대안 제시 },
      "content":"6.7유관기관 인허가 및 현장 여건에 따라 필요시 야간공사를 할 수 있으며 해당 비용을 감안하여 입찰한다.",
      "page_number":1,
      "sentence_number":3,
      "timestamp":"2024-12-05T23:02:44.297976"
     }
}
,"status":"success"}

