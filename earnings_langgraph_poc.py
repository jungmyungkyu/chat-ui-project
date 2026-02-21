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


def build_llm(api_key: str | None = None) -> ChatOpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY가 필요합니다.")

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=key,
    )


def build_app(llm: ChatOpenAI):
    def translate_node(state: EarningsState) -> Dict[str, str]:
        prompt = (
            "다음 어닝콜 영어 내용을 자연스러운 한국어로 번역해줘. "
            "숫자(매출/마진/가이던스)는 원문 단위를 유지해줘.\n\n"
            f"[원문]\n{state['transcript']}"
        )
        result = llm.invoke(prompt)
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
        result = llm.invoke(prompt)
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
        result = llm.invoke(prompt)
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
        result = llm.invoke(prompt)
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
아래 입력을 바탕으로 개인 투자자용 한국어 마크다운 리포트를 작성하라.

핵심 요구사항:
- 투자 권유/확정적 표현 금지, 확률 기반 시나리오 중심
- 리포트는 충분히 자세하고 실무적으로 작성
- 분석 근거는 반드시 원문 인용으로 연결
- 숫자 및 가이던스 출처를 명시
- 인용 표시는 반드시 [^1^], [^2^] 형태로 본문에 포함

반드시 아래 섹션 순서로 작성:
## 1. 개인화 가치 (Personalization)
- 보유 종목 및 비중 반영
- 사용자별 영향도 산출
- 점수 숫자 대신 사용자 친화적 액션 레벨로 요약: "즉시 점검 필요 / 주의 관찰 / 정상 모니터링"

## 2. 변화 감지 가치 (Delta Insight)
- 전분기 대비 가이던스 변화
- 주요 리스크 키워드 증감(증가/감소 키워드 각각)
- 경영진 톤 변화

## 3. 신뢰 가치 (Transparency)
- 모든 분석에 원문 근거 연결
- 숫자 및 가이던스 출처 명시

## 4. 핵심 기능 결과
### 4-1. 포트폴리오 영향도 분석
- 사용자 친화적 결론: 지금 행동이 필요한지 여부를 명시
### 4-2. 전분기 대비 변화 감지
- 주요 리스크 키워드 증감 분석 포함
### 4-3. 행동 보조 인사이트
- 상승/하락 시나리오
- 리스크 요인 요약
- 경쟁사 대비 위치(가능하면)

## 5. 반복 키워드 (Top 5)

## 6. 위험수준 신호
- 전체 위험수준: 낮음/보통/높음 + 근거 한 줄

## 7. 직관적 신호 요약표
- 아래 형식으로 표를 작성
| 항목 | 상태 |
| --- | --- |
| 매출 성장 | 🟢/🟡/🔴 |
| 마진 | 🟢/🟡/🔴 |
| 리스크 | 🟢/🟡/🔴 |
| 가이던스 | 🟢/🟡/🔴 |

## 8. 근거 주석
- 최소 4개
- 형식 예시:
[^1^]: "transcript 원문 인용" -> 해석
[^2^]: "transcript 원문 인용" -> 해석

[원문 transcript]
{state['transcript']}

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
        result = llm.invoke(prompt)
        return {"final_report": result.content}

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


def run_pipeline(
    transcript: str,
    previous_summary: str,
    portfolio: List[Dict[str, Any]],
    api_key: str | None = None,
) -> EarningsState:
    llm = build_llm(api_key=api_key)
    app = build_app(llm)
    return app.invoke(
        {
            "transcript": transcript,
            "previous_summary": previous_summary,
            "portfolio": portfolio,
        }
    )


def run_mock_pipeline(transcript: str, previous_summary: str, portfolio: List[Dict[str, Any]]) -> EarningsState:
    translated = (
        "[MOCK 번역]\n"
        "회사는 분기 매출이 전년 대비 증가했고, 마진이 개선되었다고 설명했습니다. "
        "다만 유럽 수요와 환율 변동성은 리스크 요인으로 언급되었습니다."
    )
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

    growth = float(signals.get("growth_score", 0))
    margin = float(signals.get("margin_score", 0))
    risk = float(signals.get("risk_score", 0))
    base_score = growth * 0.5 + margin * 0.3 + risk * 0.2
    per_asset: List[Dict[str, Any]] = []
    total_score = 0.0
    for asset in portfolio:
        weight = float(asset.get("weight", 0))
        score = round(weight * base_score, 4)
        total_score += score
        per_asset.append({"name": asset.get("name", "unknown"), "weight": weight, "impact_score": score})

    delta = {
        "improved": ["가이던스 범위 소폭 개선"],
        "deteriorated": ["거시 불확실성 언급 증가"],
        "unchanged": ["AI 성장 스토리 유지"],
        "one_line_takeaway": "성장은 유지되나 리스크 코멘트가 늘어남",
    }
    report = (
        "## 1. 개인화 가치 (Personalization)\n"
        "보유 종목 비중을 반영하면 포트폴리오 영향은 완만한 긍정이며, 현재 액션 레벨은 **주의 관찰**입니다.[^1^]\n\n"
        "## 2. 변화 감지 가치 (Delta Insight)\n"
        "전분기 대비 가이던스는 소폭 개선됐고, 리스크 키워드(수요 둔화·변동성) 언급은 증가했습니다.[^2^]\n\n"
        "## 3. 신뢰 가치 (Transparency)\n"
        "핵심 판단마다 아래 근거 주석으로 원문 출처를 연결합니다.\n\n"
        "## 4. 핵심 기능 결과\n"
        "### 4-1. 포트폴리오 영향도 분석\n"
        "이번 어닝콜은 포트폴리오에 완만한 긍정 신호이나, 반도체/글로벌 수요 리스크 노출 점검이 필요합니다.[^3^]\n\n"
        "### 4-2. 전분기 대비 변화 감지\n"
        "주요 리스크 키워드 증감: 증가(유럽 수요, 변동성), 감소(공급망 정상화).\n\n"
        "### 4-3. 행동 보조 인사이트\n"
        "상승 시나리오: AI 수요 유지 시 가이던스 추가 상향 가능\n"
        "하락 시나리오: 유럽 수요 둔화와 환율 변동성 확대 시 실적 압박\n\n"
        "## 5. 반복 키워드 (Top 5)\n"
        "- AI\n- Guidance\n- Margin\n- Demand\n- Volatility\n\n"
        "## 6. 위험수준 신호\n"
        "- 전체 위험수준: **보통(🟡)** — 성장 모멘텀은 있으나 매크로/환율 리스크가 병존.[^4^]\n\n"
        "## 7. 직관적 신호 요약표\n"
        "| 항목 | 상태 |\n"
        "| --- | --- |\n"
        "| 매출 성장 | 🟢 |\n"
        "| 마진 | 🟡 |\n"
        "| 리스크 | 🔴 |\n"
        "| 가이던스 | 🟢 |\n\n"
        "## 8. 근거 주석\n"
        "[^1^]: \"AI demand increased\" -> 성장 모멘텀으로 즉시 매도/매수보다 관찰 전략이 합리적\n"
        "[^2^]: \"Guidance raised\" -> 전분기 대비 전망 개선 신호\n"
        "[^3^]: \"Risks include Europe demand\" -> 지역 수요 리스크 노출 확인 필요\n"
        "[^4^]: \"FX volatility\" -> 변동성 리스크로 위험수준 보통 판단\n"
        )

    return {
        "transcript": transcript,
        "previous_summary": previous_summary,
        "portfolio": portfolio,
        "translated_text": translated,
        "structured_data": structured,
        "signals": signals,
        "delta_analysis": delta,
        "portfolio_impact": {
            "total_score": round(total_score, 4),
            "risk_level": "높음" if risk < -0.3 else "보통" if risk < 0.2 else "낮음",
            "per_asset": per_asset,
        },
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
