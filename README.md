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

## 환경변수 설정
민감정보 보호를 위해 키는 코드에 넣지 말고 환경변수로 설정하세요.

```bash
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 입력
```

또는 쉘에서 직접:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini
```


## 최신 버전 동기화(로컬이 구버전일 때)
```bash
git fetch origin
git checkout main
git pull origin main
git reset --hard origin/main  # 로컬 변경사항을 버리고 최신으로 맞출 때만
```

로컬 커밋 확인:
```bash
git rev-parse HEAD
git log --oneline -n 5
```

## 1) 스크립트 실행
```bash
pip install langgraph langchain-openai python-dotenv
python earnings_langgraph_poc.py
```

기본 입력 파일:
- `earnings.txt`
- `previous.txt`
- `portfolio.json`

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
  "portfolio": [{"name":"AAPL","weight":0.4}],
  "use_mock": false
}
```

- `previous_summary`는 UI 입력이 아닌 서버의 에이전트 결과(`previous.txt`)를 사용
- `use_mock=true`: OpenAI 키 없이도 UI/API 동작 검증 가능
- 실모드(`use_mock=false`)는 서버 환경변수 `OPENAI_API_KEY` 필요
