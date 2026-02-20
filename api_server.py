import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from earnings_langgraph_poc import run_mock_pipeline, run_pipeline


class AnalyzeRequest(BaseModel):
    transcript: str = Field(min_length=1)
    previous_summary: str = Field(default="")
    portfolio: List[Dict[str, Any]] = Field(default_factory=list)
    use_mock: bool = False
    api_key: Optional[str] = None


app = FastAPI(title="Earnings Call PoC API", version="0.2.0")
templates = Jinja2Templates(directory="templates")


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
            detail="OPENAI_API_KEY가 없어 실모드 실행이 불가합니다. api_key를 보내거나 use_mock=true로 테스트하세요.",
        )

    portfolio = req.portfolio or [{"name": "SAMPLE", "weight": 1.0}]

    try:
        if req.use_mock:
            return run_mock_pipeline(req.transcript, req.previous_summary, portfolio)
        return run_pipeline(req.transcript, req.previous_summary, portfolio, api_key=req.api_key)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"분석 실패: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
