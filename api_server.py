import os
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from earnings_langgraph_poc import build_llm, build_translation_app, run_mock_pipeline, run_pipeline




class TranslateRequest(BaseModel):
    transcript: str = Field(min_length=1)
    use_mock: bool = False
    api_key: Optional[str] = None

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




@app.post("/translate-stream")
async def translate_stream(req: TranslateRequest):
    if not req.use_mock and not (req.api_key or os.getenv("OPENAI_API_KEY")):
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY가 없어 실모드 실행이 불가합니다. 서버 환경변수를 설정하거나 use_mock=true로 테스트하세요.",
        )

    if req.use_mock:
        translated = (
            "회사는 분기 매출이 전년 대비 증가했고, 마진이 개선되었다고 설명했습니다. "
            "다만 유럽 수요와 환율 변동성은 리스크 요인으로 언급되었습니다."
        )
    else:
        llm = build_llm(api_key=req.api_key, streaming=True)
        app_graph = build_translation_app(llm)

        async def _gen_graph() -> AsyncIterator[str]:
            async for event in app_graph.astream_events(
                {"transcript": req.transcript}, version="v2"
            ):
                if event.get("event") != "on_chat_model_stream":
                    continue

                data = event.get("data", {})
                chunk = data.get("chunk")
                text = ""
                if chunk is not None:
                    if hasattr(chunk, "content"):
                        content = chunk.content
                        if isinstance(content, str):
                            text = content
                        elif isinstance(content, list):
                            text = "".join(
                                part.get("text", "") for part in content if isinstance(part, dict)
                            )
                    else:
                        text = str(chunk)

                if text:
                    yield text

        return StreamingResponse(_gen_graph(), media_type="text/plain; charset=utf-8")

    async def _gen_mock() -> AsyncIterator[str]:
        for token in translated.split(" "):
            yield token + " "

    return StreamingResponse(_gen_mock(), media_type="text/plain; charset=utf-8")

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
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
