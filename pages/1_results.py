import io
from typing import Optional

import pandas as pd
import streamlit as st


def export_to_excel(dataframe: pd.DataFrame) -> bytes:
    with io.BytesIO() as buffer:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            dataframe.to_excel(writer, index=False)
        return buffer.getvalue()


def get_filtered_dataframe() -> Optional[pd.DataFrame]:
    return st.session_state.get("filtered_dataframe")


def get_prompt() -> str:
    return st.session_state.get("search_prompt", "")


def get_analysis_snapshot() -> Optional[dict]:
    return st.session_state.get("analysis_snapshot")


def get_ai_strategy() -> Optional[str]:
    return st.session_state.get("ai_strategy")


def display_insights(snapshot: Optional[dict]) -> None:
    if not snapshot:
        return

    total = snapshot.get("total_results", 0)
    average_score = snapshot.get("average_score")
    prompt_tokens = snapshot.get("prompt_tokens", [])

    col_total, col_avg, col_tokens = st.columns(3)
    col_total.metric("총 결과 수", total)
    col_avg.metric(
        "평균 리드 점수",
        f"{average_score:.1f}" if average_score is not None else "-",
    )
    col_tokens.metric("프롬프트 키워드 수", len(prompt_tokens))

    grade_counts = snapshot.get("grade_counts", {})
    sentiment_counts = snapshot.get("sentiment_counts", {})
    col_hot, col_warm, col_cold = st.columns(3)
    col_hot.metric("Hot", grade_counts.get("Hot", 0))
    col_warm.metric("Warm", grade_counts.get("Warm", 0))
    col_cold.metric("Cold", grade_counts.get("Cold", 0))

    st.markdown("#### 감성 분포")
    sentiment_df = (
        pd.DataFrame(
            [
                {"감성": label, "건수": count}
                for label, count in sentiment_counts.items()
            ]
        )
        if sentiment_counts
        else pd.DataFrame(columns=["감성", "건수"])
    )
    st.dataframe(sentiment_df, use_container_width=True, hide_index=True)

    top_tags = snapshot.get("top_tags", [])
    if top_tags:
        st.markdown("#### 상위 추천 태그")
        tags_per_column = 3
        columns = st.columns(tags_per_column)
        for idx, (tag, count) in enumerate(top_tags):
            column = columns[idx % tags_per_column]
            column.markdown(f"- **{tag}** · {count}건")


def display_summary(df: pd.DataFrame, snapshot: Optional[dict]) -> None:
    st.caption(f"총 {len(df)}개의 거래처가 검색되었습니다. (최대 300개까지 표시)")
    display_insights(snapshot)

    st.markdown("#### 리드 등급별 상세 보기")
    tab_all, tab_hot, tab_warm, tab_cold = st.tabs(["전체", "Hot", "Warm", "Cold"])

    with tab_all:
        st.dataframe(df, use_container_width=True)

    if "리드등급" in df.columns:
        for tab, grade in [(tab_hot, "Hot"), (tab_warm, "Warm"), (tab_cold, "Cold")]:
            subset = df[df["리드등급"] == grade]
            with tab:
                if subset.empty:
                    st.info(f"{grade} 등급 거래처가 없습니다.")
                else:
                    st.dataframe(subset, use_container_width=True)

    if "추천액션" in df.columns:
        st.markdown("#### 추천 액션 요약")
        action_counts = (
            df["추천액션"]
            .value_counts()
            .rename_axis("추천 액션")
            .reset_index(name="건수")
        )
        st.dataframe(action_counts, use_container_width=True, hide_index=True)


def display_empty_state() -> None:
    st.info(
        "아직 검색 결과가 없습니다. 메인 페이지에서 엑셀 파일을 업로드하고 프롬프트를 입력한 뒤 다시 시도해주세요."
    )
    st.page_link("main.py", label="메인 페이지로 이동", icon="⬅️")


def main() -> None:
    st.set_page_config(page_title="거래처 결과", page_icon="📄", layout="wide")
    st.title("검색 결과")

    dataframe = get_filtered_dataframe()
    if dataframe is None or dataframe.empty:
        display_empty_state()
        return

    prompt = get_prompt()
    if prompt:
        st.subheader(f"프롬프트: {prompt}")

    snapshot = get_analysis_snapshot()
    display_summary(dataframe, snapshot)

    st.markdown("### AI 영업 전략 제안")
    ai_strategy = get_ai_strategy()
    if ai_strategy:
        st.markdown(ai_strategy)
    else:
        st.info("AI 전략 정보를 불러오지 못했습니다. 메인 페이지에서 다시 검색해 주세요.")

    excel_bytes = export_to_excel(dataframe)
    st.download_button(
        label="결과 다운로드 (Excel)",
        data=excel_bytes,
        file_name="filtered_accounts.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()

