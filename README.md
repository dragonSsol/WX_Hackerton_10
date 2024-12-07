## 변경된 응답값 (12/7 22:30)
```json
{"metadata":
	{"processed_at":"2024-12-07T16:36:41.200928",
	 "total_sentences":304, //총 문장수
	 "vector_store":{"embedding_model":"BAAI/bge-m3","id":"store_huggingface_bge-m3_20241207_145550","model_type":"huggingface"},
	 "violation_count":70} //위반 건수

,"results":
{"3":  //section_number와 같은 값 - 문장 번호 대신 쪼개진 구역?별 번호
  {"analysis":
			{
			 "asis_sentence":"현장 내 주차공간이 없으므로 카풀 및 대중교통을 적극 이용하며, 현장에 출입하는 모든 차량은 출입허가를득한후출입할수있다.(업체필수차량만승인하며,현장주변은상시주차단속지역)", //원문문장 -앞에 번호제거
			 "comments":"법에 따라 인정되거나 법에서 보호하는 수급사업자의 권리·이익을 부당하게 제한하는 약정", //검토의견
			 "detection_flag":"Y", //검출여부(위반여부)
			 "tobe_sentence":"현장 내 주차공간이 없으므로 카풀 및 대중교통을 적극 이용하며, 현장에 출입하는 모든 차량은 출입허가를 득한 후 출입할 수 있다. 지정된 주차 공간에 주차하여야 한다."// 대안문장(추천문장)
			}
	 "content":"3) 현장 내 주차공간이 없으므로 카풀 및 대중교통을 적극 이용하며, 현장에 출입하는 모든 차량은 출입허가를득한후출입할수있다.(업체필수차량만승인하며,현장주변은상시주차단속지역)", //실제 원문 문장
	 "page_number":31,
	"section_number":3,
    "timestamp":"2024-12-07T16:17:21.007589"
},
...
"status":"success"}
```




1. 업로드한 계약 문서 위반 문장 검출 서버 실행 방법
  python api/main.py

2. 검출 API 호출 방법
- POST 방식 호출
- API URL : http://localhost:5003/analyze_contract
- parameter : file
              ( test file path : data/contract_test.pdf)
- 응답값 예시 :
```json
{
  "metadata": {
    "processed_at": "2024-12-05T23:06:37.010133",
    "total_sentences": 48,
    "vector_store": {
      "embedding_model": "text-embedding-3-large",
      "id": "store_openai_text-embedding-3-large_20241204_001024",
      "model_type": "openai"
    },
    "violation_count": 15   // 전체 검출위반 건수
  },
  "results": {
    "3": {
      "analysis": {
        "asis_sentence": "원문 문장",
        "detection_flag": "Y/N",
        "comments": "위반문장일때 위반인 이유/검토의견",
        "tobe_sentence": "대안 제시"
      },
      "content": "(원문문장과 같음) 6.7유관기관 인허가 및 현장 여건에 따라 필요시 야간공사를 할 수 있으며 해당 비용을 감안하여 입찰한다.",
      "page_number": 1,
      "sentence_number": 3,
      "timestamp": "2024-12-05T23:02:44.297976"
    }
  },
  "status": "success"
}
