# LangGraph Earnings Call PoC (MVP)

LLM 기반 어닝콜 분석 PoC 예제입니다.

## MVP 기능
1. 영어 어닝콜 텍스트 입력
2. 한국어 번역
3. 구조화 요약(JSON)
4. 긍/부정/리스크 신호 추출
5. 전분기 대비 변화 분석
6. 포트폴리오 영향도 계산
7. 개인화 리포트 생성

## 1) 스크립트 실행
```bash
pip install langgraph langchain-openai python-dotenv
export OPENAI_API_KEY=...
python earnings_langgraph_poc.py
```

기본 입력 파일:
- `earnings.txt`
- `previous.txt`
- `portfolio.json`

`OPENAI_MODEL` 환경변수로 모델을 바꿀 수 있습니다. (기본: `gpt-4o-mini`)

## 2) API + UI 실행 (테스트용)
```bash
pip install fastapi uvicorn jinja2 langgraph langchain-openai python-dotenv
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

브라우저에서 `http://localhost:8000` 접속 후 테스트할 수 있습니다.

### 엔드포인트
- `GET /health`
- `POST /analyze`

요청 예시:
```json
{
  "transcript": "Company reported quarterly revenue...",
  "previous_summary": "Previous quarter...",
  "portfolio": [{"name":"AAPL","weight":0.4}],
  "use_mock": true
}
```

- `use_mock=true`: OpenAI 키 없이도 UI/API 동작 검증 가능
- `use_mock=false`: `OPENAI_API_KEY` 필요
