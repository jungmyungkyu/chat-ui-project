import json
import os
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

load_dotenv()


class EarningsState(TypedDict, total=False):
    transcript: str
    previous_summary: str
    portfolio: List[Dict[str, Any]]

    translated_text: str
    structured_data: Dict[str, Any]
    signals: Dict[str, Any]
    delta_analysis: Dict[str, Any]
    portfolio_impact: Dict[str, Any]
    final_report: str


_llm = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
    return _llm


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if "```" in text:
        chunks = text.split("```")
        for chunk in chunks:
            candidate = chunk.replace("json", "", 1).strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    return {"raw_output": text}


def translate_node(state: EarningsState) -> Dict[str, str]:
    prompt = (
        "다음 어닝콜 영어 내용을 한국어로 번역해줘. "
        "숫자(매출/마진/가이던스)는 원문 단위를 유지해줘.\n\n"
        f"[원문]\n{state['transcript']}"
    )
    result = get_llm().invoke(prompt)
    return {"translated_text": result.content}


def structure_node(state: EarningsState) -> Dict[str, Dict[str, Any]]:
    prompt = f"""
다음 어닝콜 내용을 아래 스키마로 JSON만 출력하라.

{{
  "revenue": {{"value": "", "comment": ""}},
  "margin": {{"value": "", "comment": ""}},
  "guidance": {{"value": "", "comment": ""}},
  "risks": ["", ""],
  "qna_summary": ["", ""]
}}

[내용]
{state['translated_text']}
"""
    result = get_llm().invoke(prompt)
    return {"structured_data": _extract_json(result.content)}


def signal_node(state: EarningsState) -> Dict[str, Dict[str, Any]]:
    prompt = f"""
다음 데이터를 기반으로 점수를 계산하라.
- growth_score (-1~1)
- margin_score (-1~1)
- risk_score (-1~1, 위험 클수록 음수)

반드시 JSON만 출력:
{{
  "growth_score": 0.0,
  "margin_score": 0.0,
  "risk_score": 0.0,
  "rationale": ["근거1", "근거2"]
}}

[입력]
{json.dumps(state.get('structured_data', {}), ensure_ascii=False)}
"""
    result = get_llm().invoke(prompt)
    return {"signals": _extract_json(result.content)}


def delta_node(state: EarningsState) -> Dict[str, Dict[str, Any]]:
    prompt = f"""
이전 분기 요약과 이번 분기 내용을 비교해 핵심 변화만 정리하라.
반드시 JSON으로 출력:
{{
  "improved": [""],
  "deteriorated": [""],
  "unchanged": [""],
  "one_line_takeaway": ""
}}

[이전 분기]
{state['previous_summary']}

[이번 분기]
{state['translated_text']}
"""
    result = get_llm().invoke(prompt)
    return {"delta_analysis": _extract_json(result.content)}


def impact_node(state: EarningsState) -> Dict[str, Dict[str, Any]]:
    signals = state.get("signals", {})

    growth = float(signals.get("growth_score", 0))
    margin = float(signals.get("margin_score", 0))
    risk = float(signals.get("risk_score", 0))

    per_asset: List[Dict[str, Any]] = []
    total_score = 0.0
    base_score = growth * 0.5 + margin * 0.3 + risk * 0.2

    for asset in state["portfolio"]:
        name = asset.get("name", "unknown")
        weight = float(asset.get("weight", 0))
        asset_score = round(weight * base_score, 4)
        total_score += asset_score
        per_asset.append({"name": name, "weight": weight, "impact_score": asset_score})

    return {
        "portfolio_impact": {
            "total_score": round(total_score, 4),
            "risk_level": "높음" if risk < -0.3 else "보통" if risk < 0.2 else "낮음",
            "per_asset": per_asset,
        }
    }


def report_node(state: EarningsState) -> Dict[str, str]:
    prompt = f"""
아래 입력을 바탕으로 개인 투자자용 한국어 리포트를 작성하라.
조건:
- 투자 권유/확정적 표현 금지
- '가능 시나리오' 중심
- 1) 핵심 요약 2) 긍/부정/리스크 3) 전분기 변화 4) 포트폴리오 영향 5) 체크포인트

[포트폴리오]
{json.dumps(state['portfolio'], ensure_ascii=False)}

[구조화 데이터]
{json.dumps(state.get('structured_data', {}), ensure_ascii=False)}

[신호 분석]
{json.dumps(state.get('signals', {}), ensure_ascii=False)}

[전분기 대비 변화]
{json.dumps(state.get('delta_analysis', {}), ensure_ascii=False)}

[포트폴리오 영향]
{json.dumps(state.get('portfolio_impact', {}), ensure_ascii=False)}
"""
    result = get_llm().invoke(prompt)
    return {"final_report": result.content}


def build_app():
    graph = StateGraph(EarningsState)
    graph.add_node("translate", translate_node)
    graph.add_node("structure", structure_node)
    graph.add_node("signal", signal_node)
    graph.add_node("delta", delta_node)
    graph.add_node("impact", impact_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("translate")
    graph.add_edge("translate", "structure")
    graph.add_edge("structure", "signal")
    graph.add_edge("signal", "delta")
    graph.add_edge("delta", "impact")
    graph.add_edge("impact", "report")
    return graph.compile()


def run_pipeline(transcript: str, previous_summary: str, portfolio: List[Dict[str, Any]]) -> EarningsState:
    app = build_app()
    return app.invoke(
        {
            "transcript": transcript,
            "previous_summary": previous_summary,
            "portfolio": portfolio,
        }
    )


def run_mock_pipeline(transcript: str, previous_summary: str, portfolio: List[Dict[str, Any]]) -> EarningsState:
    translated = f"[MOCK 번역]\n{transcript[:400]}"
    structured = {
        "revenue": {"value": "N/A", "comment": "샘플 파서 결과"},
        "margin": {"value": "N/A", "comment": "샘플 파서 결과"},
        "guidance": {"value": "N/A", "comment": "샘플 파서 결과"},
        "risks": ["수요 둔화 가능성", "환율 변동성"],
        "qna_summary": ["AI 수요 언급", "기업 IT 지출 보수적"],
    }
    signals = {
        "growth_score": 0.2,
        "margin_score": 0.1,
        "risk_score": -0.2,
        "rationale": ["가이던스 유지", "유럽 수요 약세"],
    }

    impact_state = impact_node({"signals": signals, "portfolio": portfolio})
    delta = {
        "improved": ["가이던스 범위 소폭 개선"],
        "deteriorated": ["거시 불확실성 언급 증가"],
        "unchanged": ["AI 성장 스토리 유지"],
        "one_line_takeaway": "성장은 유지되나 리스크 코멘트가 늘어남",
    }
    report = (
        "[MOCK 리포트]\n"
        "- 핵심 요약: 성장 기대는 유효하나 단기 변동성 리스크가 존재합니다.\n"
        "- 체크포인트: 다음 분기 가이던스, 수요 회복 신호, 비용 통제 수준을 확인하세요."
    )

    return {
        "transcript": transcript,
        "previous_summary": previous_summary,
        "portfolio": portfolio,
        "translated_text": translated,
        "structured_data": structured,
        "signals": signals,
        "delta_analysis": delta,
        "portfolio_impact": impact_state["portfolio_impact"],
        "final_report": report,
    }


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_portfolio(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    transcript = load_text_file("earnings.txt")
    previous_summary = load_text_file("previous.txt")
    portfolio = load_portfolio("portfolio.json")

    result = run_pipeline(transcript, previous_summary, portfolio)
    print("\n====== 개인화 어닝콜 리포트 ======\n")
    print(result["final_report"])
