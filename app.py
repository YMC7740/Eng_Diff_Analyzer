import os
import joblib
import pandas as pd
import spacy
import streamlit as st

# ==========================================
# 1. 頁面基本設定
# ==========================================
st.set_page_config(
    page_title="台灣中小學試題句子難度檢測系統",
    page_icon="📚",
    layout="wide"
)

# ==========================================
# 2. 核心資源快取載入區
# ==========================================
@st.cache_resource
def load_nlp():
    """載入 spaCy 中文模型並寫入快取"""
    try:
        return spacy.load("zh_core_web_sm")
    except OSError:
        st.error("❌ 找不到 spaCy 'zh_core_web_sm' 模型！請確保 requirements.txt 有正確配置下載。")
        st.stop()

@st.cache_resource
def load_difficulty_model():
    """載入 baseline pkl 模型並寫入快取，若無檔案則傳回 None"""
    model_path = "mdd_baseline_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# ==========================================
# 3. 難度特徵運算與複句分析邏輯
# ==========================================
def analyze_clause_types(doc):
    """分析複句結構與句式類型"""
    # 台灣國小常見複句連接詞特徵庫
    connectors = {
        "因果複句": ["因為", "所以", "因此", "由於", "導致", "以致於"],
        "轉折複句": ["雖然", "但是", "不過", "然而", "卻", "可是"],
        "假設複句": ["如果", "要是", "假如", "假使", "若是", "的話"],
        "條件複句": ["只要", "只有", "除非", "無論", "不管", "都"],
        "並列複句": ["同時", "一方面", "以及", "既...又", "也"]
    }
    
    text = doc.text
    detected_types = []
    
    for clause_type, keywords in connectors.items():
        if any(kw in text for kw in keywords):
            detected_types.append(clause_type)
            
    # 如果找不到連接詞，利用依存句法分析是否有複雜的狀語從句
    if not detected_types:
        dep_labels = [token.dep_ for token in doc]
        if "advcl" in dep_labels or "conj" in dep_labels:
            detected_types.append("複雜修飾句")
        else:
            detected_types.append("簡單句")
            
    return ", ".join(detected_types)

def calculate_features(text, nlp_model):
    """計算題目的文本與語法特徵 (含 MDD 與 複句分析)"""
    doc = nlp_model(text)
    
    # 基本特徵
    char_count = len(text)
    word_count = len(doc)
    
    # 詞性比率統計
    nouns = [token for token in doc if token.pos_ in ("NOUN", "PROPN")]
    verbs = [token for token in doc if token.pos_ == "VERB"]
    noun_ratio = len(nouns) / word_count if word_count > 0 else 0.0
    verb_ratio = len(verbs) / word_count if word_count > 0 else 0.0
    
    # MDD (Mean Dependency Distance - 平均依存距離)
    dep_distances = [
        abs(token.i - token.head.i) 
        for token in doc 
        if token.head != token  # 排除 ROOT 本身
    ]
    mdd = sum(dep_distances) / len(dep_distances) if dep_distances else 0.0
    
    # 複句結構分析
    clause_types = analyze_clause_types(doc)
    
    return {
        "char_count": char_count,
        "word_count": word_count,
        "noun_ratio": noun_ratio,
        "verb_ratio": verb_ratio,
        "mdd": mdd,
        "clause_types": clause_types
    }

def predict_grade(features, ml_model):
    """根據 ML 模型預測，若無模型則使用規則引擎"""
    if ml_model is not None:
        df_features = pd.DataFrame([{
            "char_count": features["char_count"],
            "word_count": features["word_count"],
            "noun_ratio": features["noun_ratio"],
            "verb_ratio": features["verb_ratio"],
            "mdd": features["mdd"]
        }])
        try:
            return ml_model.predict(df_features)[0]
        except Exception:
            pass

    # --- 備用規則引擎 (增強版：結合 MDD 與複句類型) ---
    if features["char_count"] <= 25 and features["mdd"] < 1.8 and features["clause_types"] == "簡單句":
        return "1-2 年級 (低年級)"
    elif features["char_count"] >= 55 or features["mdd"] >= 2.4 or any(c in features["clause_types"] for c in ["假設", "條件", "因果"]):
        return "5-6 年級 (高年級)"
    else:
        return "3-4 年級 (中年級)"

# ==========================================
# 4. 前端介面與互動
# ==========================================
# 側邊欄狀態
with st.sidebar:
    st.header("⚙️ 系統狀態")
    nlp = load_nlp()
    st.success("✅ spaCy 中文模型已載入")
    
    model = load_difficulty_model()
    if model is not None:
        st.success("✅ 已成功載入 mdd_baseline_model.pkl")
    else:
        st.warning("⚠️ 未找到 mdd_baseline_model.pkl，使用規則引擎。")
        
    st.divider()
    st.markdown("### 📊 分析設定")
    subject = st.selectbox("學科", ["國語文", "數學", "社會", "自然"])
    show_table = st.checkbox("顯示特徵明細表", value=True)

# 主畫面主標題
st.title("📚 台灣中小學試題句子難度檢測系統")
st.caption("支援單題檢測、句式特徵解析，以及 CSV/Excel 題庫批次上傳分析。")

# 使用分頁區隔「單題檢測」與「批次查詢」
tab1, tab2 = st.tabs(["✍️ 單題檢測與複句分析", "📂 批次題庫上傳檢測"])

# --- TAB 1: 單題檢測 ---
with tab1:
    question_text = st.text_area(
        "題目文字",
        height=130,
        placeholder="請將試題文字貼在這裡...（例如：因為果園裡的蘋果成熟了，所以小明去摘了15顆。）"
    )

    if st.button("🚀 開始檢測單題", type="primary"):
        if not question_text.strip():
            st.error("請先輸入或貼上題目文字喔！")
        else:
            with st.spinner("系統快速運算中..."):
                features = calculate_features(question_text, nlp)
                predicted_grade = predict_grade(features, model)
                
                st.divider()
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎯 預估年級", str(predicted_grade))
                with col2:
                    st.metric("📏 總字數", f"{features['char_count']} 字")
                with col3:
                    st.metric("🧠 依存距離 (MDD)", f"{features['mdd']:.2f}")
                with col4:
                    st.metric("🔗 複句結構", features["clause_types"])
                    
                if show_table:
                    st.subheader("📋 試題特徵明細")
                    detail_df = pd.DataFrame({
                        "特徵名稱": ["總字數", "總詞數", "名詞出現比例", "動詞出現比例", "MDD 依存距離", "句式與複句分類"],
                        "數值": [
                            f"{features['char_count']}",
                            f"{features['word_count']}",
                            f"{features['noun_ratio']:.1%}",
                            f"{features['verb_ratio']:.1%}",
                            f"{features['mdd']:.2f}",
                            f"{features['clause_types']}"
                        ]
                    })
                    st.table(detail_df)

# --- TAB 2: 批次查詢 ---
with tab2:
    st.markdown("### 批次檢測 Excel / CSV 檔案")
    st.info("請確保上傳的檔案中，包含一個標題名稱為 **「題目」** 或 **「question」** 的欄位。")
    
    uploaded_file = st.file_uploader("請選擇 CSV 或 Excel 檔案", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            # 尋找文字欄位
            text_col = None
            for col in df.columns:
                if col in ["題目", "question", "Text", "試題", "內容"]:
                    text_col = col
                    break
            
            if not text_col:
                st.error("❌ 找不到有效的題目欄位！請確認表格有「題目」或「question」的標題欄位。")
            else:
                st.write(f"成功載入 `{len(df)}` 筆試題，即將為欄位 `[{text_col}]` 進行分析：")
                
                if st.button("⚡ 開始批次分析", type="primary"):
                    results = []
                    with st.spinner("正在逐題批次檢測..."):
                        for idx, row in df.iterrows():
                            q_text = str(row[text_col])
                            feat = calculate_features(q_text, nlp)
                            grade = predict_grade(feat, model)
                            
                            results.append({
                                "原始題目": q_text,
                                "預估適用年級": grade,
                                "複句結構": feat["clause_types"],
                                "總字數": feat["char_count"],
                                "MDD數值": round(feat["mdd"], 2)
                            })
                            
                    res_df = pd.DataFrame(results)
                    st.success("🎉 批次分析完成！結果如下：")
                    st.dataframe(res_df, use_container_width=True)
                    
                    # 下載結果
                    csv_data = res_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 下載完整檢測結果 (CSV)",
                        data=csv_data,
                        file_name="試題檢測分析結果.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"檔案讀取失敗，詳細錯誤：{e}")
