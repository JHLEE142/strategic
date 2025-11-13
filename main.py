import io
import json
import math
import os
import re
from collections import Counter
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

KEYWORD_RULES = [
    {"tag": "임플란트 관심", "keywords": ["임플란트", "implant"], "weight": 25},
    {"tag": "장비 데모 요청", "keywords": ["데모", "시연", "체험"], "weight": 20},
    {"tag": "예산 검토 중", "keywords": ["예산", "견적", "비용"], "weight": 15},
    {"tag": "재방문 요청", "keywords": ["재방문", "다음 방문", "추가 방문"], "weight": 20},
    {"tag": "계약 임박", "keywords": ["계약", "확정", "진행"], "weight": 30},
]

POSITIVE_PATTERNS = ["긍정", "호의", "도입", "재방문", "추진", "관심"]
NEUTRAL_PATTERNS = ["검토", "보류", "숙고", "확인"]
NEGATIVE_PATTERNS = ["거절", "취소", "미정", "연기", "어려움"]

SENTIMENT_WEIGHTS = {"positive": 20, "neutral": 5, "negative": -20}

HOT_THRESHOLD = 70
WARM_THRESHOLD = 40
MAX_RESULTS = 300
AI_MODEL = "Qwen/Qwen3-4B-Instruct-2507:nscale"
AI_BASE_URL = "https://router.huggingface.co/v1"


load_dotenv()


@lru_cache(maxsize=1)
def get_openai_client() -> Optional[OpenAI]:
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=AI_BASE_URL)


def build_ai_prompt(user_prompt: str, df: pd.DataFrame, tokens: List[str]) -> str:
    preview_records = df.head(20).to_dict(orient="records")
    serialized_preview = json.dumps(preview_records, ensure_ascii=False, indent=2)
    prompt_tokens = ", ".join(tokens) if tokens else "키워드 없음"

    return (
        "당신은 B2B 거래처를 관리하는 고급 영업 전략가입니다. "
        "다음 데이터는 한국어로 정리된 거래처 상담 기록입니다. "
        "목표는 사용자의 요청을 바탕으로 전략적 제안을 구성하는 것입니다.\n\n"
        "응답 시 다음 요소를 포함하세요:\n"
        "1. 핵심 인사이트 요약 (3줄 이내)\n"
        "2. 우선순위가 높은 거래처 액션 플랜 (Hot/Warm/Cold 기준으로 최대 5개)\n"
        "3. 추가적으로 제안할 영업 전략 아이디어\n"
        "4. 후속 관리 체크리스트 (간단한 bullet)\n\n"
        f"- 사용자 프롬프트: {user_prompt or '입력 없음'}\n"
        f"- 추출된 키워드: {prompt_tokens}\n"
        "- 참고 데이터 미리보기 (최대 20건):\n"
        f"{serialized_preview}\n\n"
        "한국어로 간결하면서도 실행 가능한 형태로 제안하세요."
    )


def generate_ai_strategy(user_prompt: str, df: pd.DataFrame, tokens: List[str]) -> str:
    client = get_openai_client()
    if client is None:
        return "⚠️ `HUGGINGFACE_API_KEY` 환경 변수가 설정되지 않아 AI 전략을 생성할 수 없습니다."

    if df.empty:
        return "표시할 데이터가 없어 AI 전략을 생성하지 않았습니다."

    composed_prompt = build_ai_prompt(user_prompt, df, tokens)

    try:
        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 데이터 기반으로 상담 전략을 설계하는 영업 전문가입니다.",
                },
                {"role": "user", "content": composed_prompt},
            ],
            temperature=0.4,
            max_tokens=600,
        )
    except Exception as error:  # pylint: disable=broad-except
        return f"AI 전략 생성 중 오류가 발생했습니다: {error}"

    choices = getattr(response, "choices", [])
    if not choices:
        return "AI 모델로부터 유효한 응답을 받지 못했습니다."

    message = choices[0].message
    return getattr(message, "content", "AI 모델 응답이 비어 있습니다.")


def tokenize_prompt(prompt: str) -> List[str]:
    return [token for token in re.split(r"\s+", prompt.strip()) if token]


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def combine_row_text(row: pd.Series) -> str:
    return " ".join(normalize_text(value) for value in row if str(value).strip())


def keyword_features(text: str) -> Tuple[List[str], int]:
    tags: List[str] = []
    score = 0
    for rule in KEYWORD_RULES:
        if any(keyword in text for keyword in rule["keywords"]):
            tags.append(rule["tag"])
            score += rule["weight"]
    return tags, score


def detect_sentiment(text: str) -> Tuple[str, int]:
    if any(token in text for token in POSITIVE_PATTERNS):
        return "긍정", SENTIMENT_WEIGHTS["positive"]
    if any(token in text for token in NEGATIVE_PATTERNS):
        return "부정", SENTIMENT_WEIGHTS["negative"]
    if any(token in text for token in NEUTRAL_PATTERNS):
        return "중립", SENTIMENT_WEIGHTS["neutral"]
    return "미확인", 0


def safe_number(value: object) -> Optional[float]:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_last_contact(row: pd.Series) -> Tuple[Optional[int], Optional[datetime]]:
    candidate_keys = ["최근상담일", "최종상담일", "마지막상담일", "상담일"]
    for key in candidate_keys:
        if key in row:
            parsed = pd.to_datetime(row[key], errors="coerce")
            if pd.notna(parsed):
                days = (datetime.now().date() - parsed.date()).days
                return days, parsed.to_pydatetime()
    return None, None


def apply_rule_based_intelligence(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    text_cache = {idx: combine_row_text(row) for idx, row in df.iterrows()}
    tag_list: List[str] = []
    score_list: List[int] = []
    sentiment_list: List[str] = []
    action_list: List[str] = []
    grade_list: List[str] = []
    keyword_hits: List[int] = []

    for idx, row in df.iterrows():
        combined_text = text_cache[idx]
        tags, keyword_score = keyword_features(combined_text)
        sentiment_label, sentiment_score = detect_sentiment(combined_text)

        score = keyword_score + sentiment_score

        consult_count = None
        for key in ["상담횟수", "상담 횟수", "상담회수"]:
            if key in row:
                consult_count = safe_number(row[key])
                break

        if consult_count is not None:
            if consult_count >= 4:
                score += 15
            elif consult_count >= 2:
                score += 8
            else:
                score += 2

        days_since_last, _ = parse_last_contact(row)
        if days_since_last is not None:
            if days_since_last <= 14:
                score += 15
            elif days_since_last <= 30:
                score += 5
            else:
                score -= 5

        score = max(0, min(100, score))

        if score >= HOT_THRESHOLD:
            grade = "Hot"
            action = "즉시 연락하여 계약 추진을 확인하세요."
        elif score >= WARM_THRESHOLD:
            grade = "Warm"
            action = "3~5일 내 후속 상담을 제안하세요."
        else:
            grade = "Cold"
            action = "관심을 끌만한 정보를 제공하고 장기 모니터링하세요."

        tag_list.append(", ".join(tags) if tags else "태그 없음")
        score_list.append(score)
        sentiment_list.append(sentiment_label)
        action_list.append(action)
        grade_list.append(grade)

        keyword_hits.append(
            sum(
                1
                for rule in KEYWORD_RULES
                if any(keyword in combined_text for keyword in rule["keywords"])
            )
        )

    enriched["추천태그"] = tag_list
    enriched["리드점수"] = score_list
    enriched["감성분석"] = sentiment_list
    enriched["추천액션"] = action_list
    enriched["리드등급"] = grade_list
    enriched["_keyword_hits"] = keyword_hits
    return enriched


def build_analysis_snapshot(df: pd.DataFrame, tokens: List[str]) -> Dict[str, object]:
    grade_counts = df["리드등급"].value_counts().to_dict() if "리드등급" in df.columns else {}
    sentiment_counts = (
        df["감성분석"].value_counts().to_dict() if "감성분석" in df.columns else {}
    )

    tag_counter: Counter[str] = Counter()
    if "추천태그" in df.columns:
        for raw_tags in df["추천태그"].dropna():
            for tag in str(raw_tags).split(","):
                cleaned = tag.strip()
                if cleaned and cleaned != "태그 없음":
                    tag_counter[cleaned] += 1

    avg_score = float(df["리드점수"].mean()) if "리드점수" in df.columns else None

    return {
        "total_results": int(len(df)),
        "grade_counts": grade_counts,
        "sentiment_counts": sentiment_counts,
        "top_tags": tag_counter.most_common(10),
        "average_score": avg_score,
        "prompt_tokens": tokens,
    }


def read_excel(file_data: bytes) -> pd.DataFrame:
    with io.BytesIO(file_data) as buffer:
        df = pd.read_excel(buffer, engine="openpyxl")
    df.columns = df.columns.map(lambda col: str(col).strip())
    return df


def store_dataframe(df: pd.DataFrame) -> None:
    st.session_state["raw_dataframe"] = df
    st.session_state["raw_preview"] = df.head(10)


def set_prompt(prompt: str) -> None:
    st.session_state["search_prompt"] = prompt


def filter_dataframe(df: pd.DataFrame, prompt: str) -> pd.DataFrame:
    tokens = tokenize_prompt(prompt)
    if not tokens:
        capped = df.head(MAX_RESULTS)
        return apply_rule_based_intelligence(capped)

    lowercase_df = df.applymap(lambda value: str(value).lower())
    token_matches_list = []

    for token in tokens:
        token_lower = token.lower()
        token_matches = lowercase_df.apply(
            lambda col: col.str.contains(token_lower, na=False, regex=False)
        ).any(axis=1)
        token_matches_list.append(token_matches)

    if not token_matches_list:
        capped = df.head(MAX_RESULTS)
        return apply_rule_based_intelligence(capped)

    match_counts = sum(token_matches_list)
    required_matches = max(1, math.ceil(len(tokens) * 0.5))
    matched_rows = match_counts >= required_matches

    filtered = df.loc[matched_rows].copy()
    if filtered.empty:
        return filtered

    match_ratio = (match_counts[matched_rows] / len(tokens)).rename("_match_ratio")
    filtered = filtered.assign(_match_ratio=match_ratio, _match_score=match_counts[matched_rows])
    filtered = filtered.sort_values(by=["_match_ratio", "_match_score"], ascending=False)
    filtered = filtered.head(MAX_RESULTS)
    filtered = apply_rule_based_intelligence(filtered)
    return filtered.drop(columns=["_match_ratio", "_match_score", "_keyword_hits"], errors="ignore")


def update_filtered_results(df: pd.DataFrame, prompt: str) -> None:
    tokens = tokenize_prompt(prompt)
    filtered = filter_dataframe(df, prompt)
    st.session_state["filtered_dataframe"] = filtered
    st.session_state["analysis_snapshot"] = build_analysis_snapshot(filtered, tokens)
    st.session_state["ai_strategy"] = generate_ai_strategy(prompt, filtered, tokens)


def main() -> None:
    st.set_page_config(
        page_title="거래처 탐색 도우미",
        page_icon="📋",
        layout="wide",
    )

    st.title("거래처 탐색 도우미")
    st.write("300개가 넘는 거래처를 규칙 기반으로 분석해 적합한 후보를 빠르게 찾을 수 있도록 도와드립니다.")

    with st.sidebar:
        st.header("규칙 기반 지능 요약")
        st.markdown(
            "- 키워드 가중치, 상담 횟수, 최근 상담일을 조합하여 리드 점수를 산출합니다.\n"
            "- 감성 패턴을 감지해 긍정/중립/부정 상황을 구분합니다.\n"
            "- 리드 등급(Hot/Warm/Cold) 별로 후속 액션을 추천합니다."
        )
        rule_df = pd.DataFrame(KEYWORD_RULES)
        sidebar_df = rule_df.rename(columns={"tag": "태그", "keywords": "키워드", "weight": "가중치"})
        st.dataframe(sidebar_df, use_container_width=True, hide_index=True)
        st.caption("필요에 따라 키워드와 가중치를 코드에서 조정할 수 있습니다.")

    uploaded_file = st.file_uploader("엑셀 파일 업로드", type=["xlsx", "xls"])
    prompt = st.text_input("검색 프롬프트", placeholder="예: 서울 지역 IT 관련 거래처")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.page_link("pages/1_results.py", label="결과 페이지로 이동", icon="➡️")

    with col2:
        if st.button("프롬프트에 맞는 거래처 찾기", use_container_width=True):
            if uploaded_file is None:
                st.warning("먼저 엑셀 파일을 업로드해주세요.")
            else:
                try:
                    dataframe = read_excel(uploaded_file.read())
                except ValueError as error:
                    st.error(f"엑셀 파일을 읽는 중 오류가 발생했습니다: {error}")
                    return

                store_dataframe(dataframe)
                set_prompt(prompt)
                update_filtered_results(dataframe, prompt)
                st.success("결과 페이지에서 거래처 목록을 확인하세요.")

    if "raw_dataframe" in st.session_state:
        st.subheader("최근 업로드 요약")
        preview_df = st.session_state["raw_preview"]
        st.dataframe(preview_df, use_container_width=True)

    if "analysis_snapshot" in st.session_state:
        snapshot = st.session_state["analysis_snapshot"]
        if snapshot:
            st.subheader("최근 규칙 기반 분석 하이라이트")
            col_total, col_avg, col_hot = st.columns(3)
            col_total.metric("총 결과", snapshot.get("total_results", 0))
            avg_score = snapshot.get("average_score")
            col_avg.metric(
                "평균 리드 점수",
                f"{avg_score:.1f}" if avg_score is not None else "-",
            )
            col_hot.metric("Hot 리드", snapshot.get("grade_counts", {}).get("Hot", 0))

            top_tags = snapshot.get("top_tags", [])
            if top_tags:
                st.markdown(
                    "· 상위 태그: "
                    + ", ".join(
                        f"{tag} ({count})" for tag, count in top_tags[:5]
                    )
                )

    st.markdown("---")
    st.markdown(
        """
        ### 사용 가이드
        - `엑셀 파일 업로드`: 거래처 정보가 담긴 엑셀 파일을 업로드하세요.
        - `검색 프롬프트`: 찾고 싶은 거래처의 조건을 입력하세요. 여러 키워드는 공백으로 구분합니다.
        - 규칙 기반 엔진이 자동으로 리드 등급과 추천 액션을 제공합니다.
        - `결과 페이지`: 필터링된 거래처 목록을 확인하고 다운로드할 수 있습니다.
        """
    )


if __name__ == "__main__":
    main()

