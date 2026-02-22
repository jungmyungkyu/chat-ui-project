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


def _ensure_inline_footnotes(report_text: str) -> str:
    if "[^" in report_text:
        return report_text

    lines = report_text.splitlines()
    note_id = 1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines[i] = f"{line} [^{note_id}^]"
        note_id += 1
        if note_id > 4:
            break

    return "\n".join(lines)


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
    # ----------------------
    # 1️⃣ Translate Node
    # ----------------------
    def translate_node(state: EarningsState) -> Dict[str, str]:
        prompt = (
            "다음 어닝콜 영어 내용을 자연스러운 한국어로 번역해줘. "
            "숫자(매출/마진/가이던스)는 원문 단위를 유지해줘.\n\n"
            f"[원문]\n{state['transcript']}"
        )
        result = llm.invoke(prompt)
        return {"translated_text": result.content}

    # ----------------------
    # 2️⃣ Analysis Node (structure + signal + delta 통합)
    # ----------------------
    def analysis_node(state: EarningsState) -> Dict[str, Any]:
        prompt = f"""
당신은 어닝콜 분석 전문가다.

아래 작업을 모두 수행하고 반드시 JSON만 출력하라.

1. 구조화:
{{
  "revenue": {{"value": null, "unit": null, "yoy": null}},
  "margin": {{"value": null, "change": null}},
  "guidance": {{"value": null, "direction": null}},
  "risks": [],
  "qna_summary": []
}}

2. 점수 계산:
- growth_score (-1~1)
- margin_score (-1~1)
- risk_score (-1~1)

3. 전분기 대비 변화:
{{
  "guidance_change": "",
  "growth_change": "",
  "margin_change": "",
  "risk_change": "",
  "tone_change": "",
  "one_line_takeaway": ""
}}

최종 출력 스키마:
{{
  "structured_data": {{...}},
  "signals": {{
    "growth_score": 0.0,
    "margin_score": 0.0,
    "risk_score": 0.0,
    "rationale": []
  }},
  "delta_analysis": {{...}}
}}

[이전 분기 요약]
{state['previous_summary']}

[이번 분기 번역본]
{state['translated_text']}
"""
        result = llm.invoke(prompt)
        parsed = _extract_json(result.content)

        return {
            "structured_data": parsed.get("structured_data", {}),
            "signals": parsed.get("signals", {}),
            "delta_analysis": parsed.get("delta_analysis", {}),
        }

    # ----------------------
    # 3️⃣ Report Node (analysis_result + portfolio)
    # ----------------------
    def report_node(state: EarningsState) -> Dict[str, Any]:
        signals = state.get("signals", {})
        growth = float(signals.get("growth_score", 0))
        margin = float(signals.get("margin_score", 0))
        risk = float(signals.get("risk_score", 0))

        base_score = growth * 0.5 + margin * 0.3 + risk * 0.2
        total_score = 0.0
        per_asset: List[Dict[str, Any]] = []

        for asset in state["portfolio"]:
            weight = float(asset.get("weight", 0))
            score = round(weight * base_score, 4)
            total_score += score
            per_asset.append(
                {
                    "name": asset.get("name", "unknown"),
                    "weight": weight,
                    "impact_score": score,
                }
            )

        portfolio_impact = {
            "total_score": round(total_score, 4),
            "risk_level": "높음" if risk < -0.3 else "보통" if risk < 0.2 else "낮음",
            "per_asset": per_asset,
        }

        prompt = f"""
당신은 개인 투자자의 의사결정을 돕는 AI 애널리스트다.

목표:
단순 요약이 아니라, 사용자가 실제로 무엇을 점검해야 하는지 판단할 수 있도록 돕는 행동 중심 리포트를 작성하라.

중요 규칙:
- 투자 권유, 매수/매도 지시, 확정적 표현 금지
- 내부 점수(total_score)는 절대 노출하지 않는다
- transcript에 없는 리스크 생성 금지
- 하단에 footnote 목록은 출력하지 말 것
- 행동 레벨은 반드시 4단계 중 하나만 사용:
  🟢 정상 모니터링
  🟡 주의 관찰
  🟠 전략 재점검
  🔴 즉시 점검 필요

------------------------------------------------------------

# 1) 핵심 요약

- 성장 / 마진 / 가이던스 중심 요약

# 2) 긍정 / 부정 신호 (기업 관점)

- 반드시 원문 기반
- 없는 리스크 생성 금지

# 3) 전분기 대비 변화

- 가이던스
- 성장률
- 마진
- 톤 변화
- 반복 키워드 Top 5
- 한줄요약

# 4) 포트폴리오 영향

- 먼저 포트폴리오 JSON을 해석하라.
- 포트폴리오 산업 특성을 한 문장으로 요약하라.
- 분석 대상 기업이 포함되어 있는지 판단하라.

[분기 규칙]

1️⃣ 포함되어 있다면:
- 직접 영향 분석
- 행동 레벨 판단

2️⃣ 포함되어 있지 않다면:
- "현재 포트폴리오에 해당 기업은 포함되어 있지 않습니다."라고 명시
- 직접 영향은 제한적임을 설명
- 대신 내 포트폴리오 기업(AAPL, MSFT, NVDA 등)에
  산업적으로 어떤 간접 영향이 있는지 분석

- 행동 레벨을 굵게 표시
- 숫자 점수는 절대 노출하지 않는다

# 5) 체크 포인트 (가장 중요한 수정)

체크 대상은 반드시 다음 우선순위를 따른다:

1️⃣ 분석 대상 기업이 포트폴리오에 포함되어 있다면 → 해당 기업 중심
2️⃣ 포함되어 있지 않다면 → 포트폴리오에 실제로 보유 중인 종목 중심

즉,
현재 케이스에서는 AAPL, MSFT, NVDA 중심으로 작성해야 한다.

행동 레벨에 따라 강도를 조절하라:

🟢 정상 모니터링:
- 1~2개만 작성
- 내 보유 종목에 대한 간단한 관찰 항목

🟡 주의 관찰:
- 2~3개 작성
- 내 보유 종목과 산업 흐름 연결

🟠 전략 재점검:
- 3~4개 작성
- 내 보유 종목의 실적/가이던스/산업 흐름 점검

🔴 즉시 점검 필요:
- 내 보유 종목 관련 단기 리스크 점검 우선순위 제시

체크포인트는:
- 반드시 내 보유 종목 기준으로 작성
- "~여부를 확인해보세요" 형태
- 5분 내 확인 가능한 것 위주
- 분석 대상 기업만을 대상으로 작성하지 말 것

------------------------------------------------------------

[원문 transcript]
{state['transcript']}

[구조화 데이터]
{json.dumps(state.get('structured_data', {}), ensure_ascii=False)}

[신호 분석]
{json.dumps(state.get('signals', {}), ensure_ascii=False)}

[전분기 대비 변화]
{json.dumps(state.get('delta_analysis', {}), ensure_ascii=False)}

[포트폴리오 JSON]
{json.dumps(state['portfolio'], ensure_ascii=False)}
"""

        result = llm.invoke(prompt)
        final_report = _ensure_inline_footnotes(result.content)

        return {
            "portfolio_impact": portfolio_impact,
            "final_report": final_report,
        }

    graph = StateGraph(EarningsState)
    graph.add_node("translate", translate_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("translate")
    graph.add_edge("translate", "analysis")
    graph.add_edge("analysis", "report")

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
        "보유 종목 비중을 반영하면 현재 행동 레벨은 **🟡 주의 관찰**이며, 급격한 포지션 변경보다 핵심 리스크 점검이 우선입니다.[^1^]\n\n"
        "## 2. 변화 감지 가치 (Delta Insight)\n"
        "전분기 대비 가이던스는 소폭 개선됐고, 리스크 키워드(수요 둔화·변동성) 언급은 증가했습니다.[^2^]\n\n"
        "## 3. 신뢰 가치 (Transparency)\n"
        "핵심 판단마다 transcript 원문 인용을 [^n^] 형태로 본문에 표시합니다.\n\n"
        "## 4. 핵심 기능 결과\n"
        "### 4-1. 포트폴리오 영향도 분석\n"
        "이번 어닝콜은 포트폴리오에 완만한 긍정 신호이나, 반도체/글로벌 수요 리스크 노출 점검이 필요합니다. \"Risks include Europe demand\"[^3^]\n\n"
        "### 4-2. 전분기 대비 변화 감지\n"
        "주요 리스크 키워드 증감: 증가(유럽 수요, 변동성), 감소(공급망 정상화).\n\n"
        "### 4-3. 행동 보조 인사이트\n"
        "상승 시나리오: AI 수요 유지 시 가이던스 추가 상향 가능. \"AI demand increased\"[^1^]\n"
        "하락 시나리오: 유럽 수요 둔화와 환율 변동성 확대 시 실적 압박. \"FX volatility\"[^4^]\n\n"
        "## 5. 반복 키워드 (Top 5)\n"
        "- AI\n- Guidance\n- Margin\n- Demand\n- Volatility\n\n"
        "## 6. 위험수준 신호\n"
        "- 전체 위험수준: **보통(🟡)** — 성장 모멘텀은 있으나 매크로/환율 리스크가 병존. \"Guidance raised\"[^2^]\n\n"
        "## 7. 직관적 신호 요약표\n"
        "| 항목 | 상태 |\n"
        "| --- | --- |\n"
        "| 매출 성장 | 🟢 |\n"
        "| 마진 | 🟡 |\n"
        "| 리스크 | 🔴 |\n"
        "| 가이던스 | 🟢 |\n\n"
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
