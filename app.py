import os
import joblib
import pandas as pd
import spacy
import streamlit as st

# ==========================================
# 1. 頁面基本設定
# ==========================================
st.set_page_config(
    page_title="台灣試題難度檢測系統",
    page_icon="📚",
    layout="wide"
)

# ==========================================
# 2. 核心資源快取載入區 (只在初次啟動執行一次)
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
# 3. 難度特徵運算與預測邏輯
# ==========================================
def calculate_features(text, nlp_model):
    """計算題目的文本與語法特徵 (含 MDD 平均依存距離)"""
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
    
    return {
        "char_count": char_count,
        "word_count": word_count,
        "noun_ratio": noun_ratio,
        "verb_ratio": verb_ratio,
        "mdd": mdd
    }

def predict_grade(features, ml_model):
    """根據 ML 模型預測，若無模型則使用規則引擎"""
    if ml_model is not None:
        # 將特徵轉換為 DataFrame 以符合 scikit-learn/joblib 預期輸入格式
        df_features = pd.DataFrame([features])
        try:
            return ml_model.predict(df_features)[0]
        except Exception:
            # 若模型欄位不匹配，安全回退到規則引擎
            pass

    # --- 備用規則引擎 (Rule-based Engine) ---
    # 根據字數與依存複雜度粗略評估
    if features["char_count"] <= 25 and features["mdd"] < 1.8:
        return "1-2 年級 (低年級)"
    elif features["char_count"] >= 60 or features["mdd"] >= 2.5:
        return "5-6 年級 (高年級)"
    else:
        return "3-4 年級 (中年級)"

# ==========================================
# 4. 前端介面與互動
# ==========================================
# 顯示側邊欄系統狀態
with st.sidebar:
    st.header("⚙️ 系統狀態")
    
    # 測試載入
    nlp = load_nlp()
    st.success("✅ spaCy 中文模型已載入")
    
    model = load_difficulty_model()
    if model is not None:
        st.success("✅ 已成功載入 mdd_baseline_model.pkl")
        use_ml = True
    else:
        st.warning("⚠️ 未找到 mdd_baseline_model.pkl，使用規則引擎。")
        use_ml = False
        
    st.divider()
    st.markdown("### 📊 分析設定")
    subject = st.selectbox("學科", ["國語文", "數學", "社會", "自然"])
    show_table = st.checkbox("顯示模型特徵表", value=True)

# 主區域顯示
st.title("📚 台灣試題難度檢測系統")
st.caption("輸入題目文字後，系統會計算字數、語意標記、MDD 平均依存距離，並預估適用年級。")

# 輸入框
question_text = st.text_area(
    "題目文字",
    height=150,
    placeholder="請將試題文字貼在這裡...（例如：小明去果園摘了15顆蘋果，送給妹妹4顆後，還剩下幾顆蘋果？）"
)

# 執行分析
if st.button("🚀 開始檢測難度", type="primary"):
    if not question_text.strip():
        st.error("請先輸入或貼上題目文字喔！")
    else:
        with st.spinner("系統快速運算中..."):
            # 1. 取得文本特徵
            features = calculate_features(question_text, nlp)
            # 2. 預測年級
            predicted_grade = predict_grade(features, model)
            
            # --- 顯示輸出結果 ---
            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 預估適用年級", str(predicted_grade))
            with col2:
                st.metric("📏 題目總字數", f"{features['char_count']} 字")
            with col3:
                st.metric("🧠 平均依存距離 (MDD)", f"{features['mdd']:.2f}")
                
            # 特徵明細表
            if show_table:
                st.subheader("📋 試題特徵分析明細")
                detail_df = pd.DataFrame({
                    "特徵名稱": ["總字數", "總詞數", "名詞出現比例", "動詞出現比例", "MDD 依存距離 (語法複雜度)"],
                    "數值": [
                        f"{features['char_count']}",
                        f"{features['word_count']}",
                        f"{features['noun_ratio']:.1%}",
                        f"{features['verb_ratio']:.1%}",
                        f"{features['mdd']:.2f}"
                    ]
                })
                st.table(detail_df)
