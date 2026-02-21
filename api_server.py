import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from earnings_langgraph_poc import run_mock_pipeline, run_pipeline


class AnalyzeRequest(BaseModel):
    transcript: str = Field(min_length=1)
    portfolio: List[Dict[str, Any]] = Field(default_factory=list)
    use_mock: bool = False
    api_key: Optional[str] = None


app = FastAPI(title="Earnings Call PoC API", version="0.3.0")
templates = Jinja2Templates(directory="templates")


def load_previous_summary_agent_output() -> str:
    """UI 입력이 아닌 서버측 요약 에이전트 결과값(예: previous.txt)을 사용한다."""
    path = Path("previous.txt")
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return "이전 분기 요약 데이터가 제공되지 않았습니다. 이번 분기 내용 중심으로 변화 가능성을 해석하세요."


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not req.use_mock and not (req.api_key or os.getenv("OPENAI_API_KEY")):
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY가 없어 실모드 실행이 불가합니다. 서버 환경변수를 설정하거나 use_mock=true로 테스트하세요.",
        )

    portfolio = req.portfolio or [{"name": "SAMPLE", "weight": 1.0}]
    previous_summary = load_previous_summary_agent_output()

    try:
        if req.use_mock:
            return run_mock_pipeline(req.transcript, previous_summary, portfolio)
        return run_pipeline(req.transcript, previous_summary, portfolio, api_key=req.api_key)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"분석 실패: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
