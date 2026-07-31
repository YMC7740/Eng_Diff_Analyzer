import re
from io import BytesIO
from io import StringIO

import joblib
import pandas as pd
import streamlit as st

try:
    import spacy
except ImportError:
    spacy = None


SUBJECTS = ["國語", "數學", "自然", "社會", "英文", "生活", "綜合"]
EXAMPLE_QUESTIONS = {
    "自然": "如果植物長期缺少陽光，可能會因為無法順利行光合作用而逐漸枯萎。",
    "社會": "雖然都市交通便利，但是人口過度集中也可能導致居住品質下降。",
    "國語": "請閱讀下列文章，並說明作者如何透過景物描寫表達心情。",
}


st.set_page_config(
    page_title="測驗卷難度分析",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_nlp_model():
    if spacy is None:
        return None
    try:
        return spacy.load("zh_core_web_sm")
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_ml_model():
    try:
        return joblib.load("mdd_baseline_model.pkl")
    except Exception:
        return None


nlp = load_nlp_model()
ml_model = load_ml_model()


class QuestionPreprocessor:
    def __init__(self):
        self.semantic_markers = {
            "因果": r"(因為|所以|由於|因此|導致|以致於)",
            "假設": r"(如果|假使|要是|若|則|假如|一旦)",
            "轉折": r"(但是|可是|卻|雖然|然而|不過|只是)",
            "條件": r"(只要|只有|除非|無論|不管|任憑)",
            "並列": r"(一邊|同時|以及|和|跟|與|既)",
            "目的": r"(為了|以便|以免|用以)",
            "遞進": r"(不但|而且|甚至|更|何況)",
            "選擇": r"(或者|還是|與其|不如|寧可)",
        }

    def _calculate_mdd(self, doc):
        if not doc:
            return 0.0

        total_distance = 0
        valid_tokens = 0
        for token in doc:
            if token.dep_ != "ROOT" and not token.is_punct and token.pos_ != "PUNCT":
                total_distance += abs(token.i - token.head.i)
                valid_tokens += 1

        return total_distance / valid_tokens if valid_tokens > 0 else 0.0

    def _classify_semantic_marker(self, text):
        for marker, pattern in self.semantic_markers.items():
            if re.search(pattern, text):
                return marker
        return "連貫／承接"

    def analyze(self, text: str, subject: str) -> dict:
        clean_text = text.strip()
        doc = nlp(clean_text) if nlp and clean_text else None
        return {
            "MDD(平均依存距離)": round(self._calculate_mdd(doc), 3),
            "字數": len(clean_text),
            "學科": subject,
            "東方語意標記": self._classify_semantic_marker(clean_text),
        }


def predict_grade(analysis_result: dict) -> str:
    if ml_model is not None:
        features_df = pd.DataFrame([analysis_result])
        prediction = ml_model.predict(features_df)[0]
        return str(prediction)

    mdd = analysis_result["MDD(平均依存距離)"]
    subject = analysis_result["學科"]
    if mdd > 3.6 and subject in ["自然", "社會"]:
        return "5年級或6年級 (Rule-based)"
    return "3年級或4年級 (Rule-based)"


def build_result_row(text: str, subject: str) -> dict:
    analysis = QuestionPreprocessor().analyze(text, subject)
    prediction = predict_grade(analysis)
    return {
        "題目": text.strip(),
        **analysis,
        "預估年級": prediction,
        "建議": make_recommendation(analysis, prediction),
    }


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    for encoding in ["utf-8-sig", "utf-8", "cp950", "big5"]:
        try:
            return pd.read_csv(BytesIO(raw), encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(BytesIO(raw))


def make_recommendation(analysis: dict, prediction: str) -> str:
    mdd = analysis["MDD(平均依存距離)"]
    word_count = analysis["字數"]
    semantic_type = analysis["東方語意標記"]

    notes = []
    if mdd >= 3.6:
        notes.append("句法距離偏高，可檢查是否需要拆句")
    elif mdd == 0:
        notes.append("尚未取得依存分析，請確認 spaCy 中文模型")
    else:
        notes.append("句法負荷適中")

    if word_count >= 65:
        notes.append("字數偏長，可縮短題幹或分段")
    elif word_count <= 15:
        notes.append("字數精簡，適合低年級或單一概念檢核")

    if semantic_type in ["因果", "假設", "條件", "轉折"]:
        notes.append(f"含{semantic_type}關係，需確認學生是否已學過相關連接詞")

    if "5年級" in prediction or "6年級" in prediction:
        notes.append("可作為中高年級題目候選")

    return "；".join(notes)


def render_health_status():
    st.sidebar.subheader("系統狀態")
    if nlp is None:
        st.sidebar.error("spaCy 中文模型未就緒，MDD 將顯示 0。")
        st.sidebar.caption("可執行：python -m spacy download zh_core_web_sm")
    else:
        st.sidebar.success("spaCy 中文模型已載入")

    if ml_model is None:
        st.sidebar.warning("未找到 mdd_baseline_model.pkl，使用規則引擎。")
    else:
        st.sidebar.success("ML 模型已載入")


def render_metric_cards(result: dict):
    metric_cols = st.columns(4)
    metric_cols[0].metric("MDD", result["MDD(平均依存距離)"])
    metric_cols[1].metric("字數", result["字數"])
    metric_cols[2].metric("語意標記", result["東方語意標記"])
    metric_cols[3].metric("預估年級", result["預估年級"])


def render_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background: #f7f8f5;
            color: #1f2933;
        }
        [data-testid="stSidebar"] {
            background: #eef2ed;
            border-right: 1px solid #d9dfd8;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dde3df;
            border-radius: 8px;
            padding: 1rem;
        }
        .status-strip {
            border-left: 4px solid #2f6f62;
            background: #ffffff;
            padding: 1rem 1.2rem;
            border-radius: 8px;
            margin: 1rem 0 1.4rem;
        }
        .muted {
            color: #607064;
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    render_styles()
    render_health_status()

    st.sidebar.subheader("分析設定")
    subject = st.sidebar.selectbox("學科", SUBJECTS, index=2)
    mode = st.sidebar.radio("輸入模式", ["單題分析", "批次分析"], horizontal=False)
    show_features = st.sidebar.toggle("顯示模型特徵表", value=True)

    st.title("測驗卷難度分析")
    st.markdown(
        "<div class='muted'>輸入題目文字後，系統會計算字數、語意標記、MDD 平均依存距離，並預估適用年級。</div>",
        unsafe_allow_html=True,
    )

    if mode == "單題分析":
        sample = EXAMPLE_QUESTIONS.get(subject, "")
        question_text = st.text_area(
            "題目文字",
            value=sample,
            height=170,
            placeholder="請貼上一題測驗題幹、閱讀題或問答題文字...",
        )

        analyze_clicked = st.button("分析這一題", type="primary", use_container_width=True)
        if analyze_clicked:
            if not question_text.strip():
                st.error("請先輸入題目文字。")
                return

            result = build_result_row(question_text, subject)
            st.markdown("<div class='status-strip'>分析完成。以下是這題目前的難度訊號與調整建議。</div>", unsafe_allow_html=True)
            render_metric_cards(result)
            st.subheader("建議")
            st.write(result["建議"])

            if show_features:
                st.subheader("模型特徵")
                st.dataframe(pd.DataFrame([result]), use_container_width=True, hide_index=True)

    else:
        batch_text = st.text_area(
            "批次題目",
            height=240,
            placeholder="每一行貼上一題，系統會逐題分析。",
        )
        uploaded_file = st.file_uploader("或上傳 CSV 檔，需包含「題目」欄位，可選填「學科」欄位", type=["csv"])
        analyze_clicked = st.button("開始批次分析", type="primary", use_container_width=True)

        if analyze_clicked:
            rows = []
            if uploaded_file is not None:
                uploaded_df = read_uploaded_csv(uploaded_file)
                if "題目" not in uploaded_df.columns:
                    st.error("CSV 需要包含「題目」欄位。")
                    return
                for _, row in uploaded_df.iterrows():
                    row_subject = row.get("學科", subject)
                    rows.append(build_result_row(str(row["題目"]), str(row_subject)))
            else:
                questions = [line.strip() for line in batch_text.splitlines() if line.strip()]
                rows = [build_result_row(question, subject) for question in questions]

            if not rows:
                st.error("請先輸入至少一題，或上傳可讀取的 CSV。")
                return

            result_df = pd.DataFrame(rows)
            st.markdown("<div class='status-strip'>批次分析完成。可在表格中排序、搜尋，或下載 CSV。</div>", unsafe_allow_html=True)
            st.dataframe(result_df, use_container_width=True, hide_index=True)

            csv_buffer = StringIO()
            result_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "下載分析結果 CSV",
                data=csv_buffer.getvalue().encode("utf-8-sig"),
                file_name="question_difficulty_analysis.csv",
                mime="text/csv",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
