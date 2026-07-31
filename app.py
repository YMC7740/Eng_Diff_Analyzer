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
# 3. 難度特徵運算與進階語句分析邏輯
# ==========================================
def analyze_clause_types(doc):
    """分析複句結構與句式類型 (台灣語境進階分級版)"""
    text = doc.text
    detected_types = []
    
    # 高級/學術論述複句 (具備較強的邏輯推導或學術語境)
    advanced_keywords = [
        "由於", "導致", "以致於", "即使", "仍", "除非", "無論", "若", 
        "除了...也", "透過", "以維持", "評估", "脈絡", "偏誤", "然而", 
        "此外", "因此", "鑑於", "唯有", "與其", "不如"
    ]
    if any(kw in text for kw in advanced_keywords):
        detected_types.append("進階論述句")
        
    # 一般基礎複句
    connectors = {
        "因果複句": ["因為", "所以"],
        "轉折複句": ["雖然", "但是", "不過", "卻", "可是"],
        "假設複句": ["如果", "要是", "假如", "的話"],
        "條件複句": ["只要", "只有", "當...時", "除了"],
        "並列複句": ["同時", "一方面", "以及", "並且", "也"]
    }
    
    for clause_type, keywords in connectors.items():
        if any(kw in text for kw in keywords):
            detected_types.append(clause_type)
            
    if not detected_types:
        dep_labels = [token.dep_ for token in doc]
        if "advcl" in dep_labels or "conj" in dep_labels:
            detected_types.append("複雜修飾句")
        else:
            detected_types.append("簡單句")
            
    return ", ".join(detected_types)

def calculate_vocab_depth(doc):
    """估算詞彙深度：判斷句子中是否包含超出基礎生活用詞的高層次學科術語/抽象詞"""
    advanced_terms = {
        "演算法", "同溫層", "合力", "敘事觀點", "社會文化脈絡", "供給", "需求",
        "蒸發", "凝結", "光合作用", "分裂", "細胞", "生物體", "公義", "政治結構",
        "經濟條件", "環境影響", "社會公平", "單一因果", "偏誤", "多元觀點", "效率"
    }
    tokens = [token.text for token in doc]
    match_count = sum(1 for t in tokens if t in advanced_terms)
    return match_count

def calculate_features(text, nlp_model):
    """計算題目的文本與語法特徵 (含 MDD、專有名詞密度與複句分析)"""
    doc = nlp_model(text)
    
    char_count = len(text)
    word_count = len(doc)
    
    # 統計名詞與動詞 (包含 PROPN 專有名詞)
    nouns = [token for token in doc if token.pos_ in ("NOUN", "PROPN")]
    verbs = [token for token in doc if token.pos_ == "VERB"]
    noun_ratio = len(nouns) / word_count if word_count > 0 else 0.0
    verb_ratio = len(verbs) / word_count if word_count > 0 else 0.0
    
    # 計算 MDD (Mean Dependency Distance - 平均依存距離)
    dep_distances = [
        abs(token.i - token.head.i) 
        for token in doc 
        if token.head != token
    ]
    mdd = sum(dep_distances) / len(dep_distances) if dep_distances else 0.0
    
    # 語式結構與高級詞彙深度
    clause_types = analyze_clause_types(doc)
    vocab_depth = calculate_vocab_depth(doc)
    
    return {
        "char_count": char_count,
        "word_count": word_count,
        "noun_ratio": noun_ratio,
        "verb_ratio": verb_ratio,
        "mdd": mdd,
        "clause_types": clause_types,
        "vocab_depth": vocab_depth
    }

def predict_grade(features, ml_model):
    """結合 ML 基準模型與規則引擎的混合校正機制"""
    base_grade = 3  # 預設為中年級
    
    # 1. 嘗試由模型取得初始年級數值
    if ml_model is not None:
        try:
            df_features = pd.DataFrame([{
                "char_count": features["char_count"],
                "word_count": features["word_count"],
                "noun_ratio": features["noun_ratio"],
                "verb_ratio": features["verb_ratio"],
                "mdd": features["mdd"]
            }])
            raw_pred = ml_model.predict(df_features)[0]
            # 若模型輸出為整數型態，保留其基準；否則以 3 為基礎
            if isinstance(raw_pred, (int, float)):
                base_grade = int(raw_pred)
        except Exception:
            pass

    # 2. 進行特徵權重動態校正 (根據台灣語境加減級數)
    adjustment = 0
    
    # 低年級下修條件：字少 + 語法簡單 + 簡單句
    if features["char_count"] <= 20 and features["mdd"] < 2.5 and features["noun_ratio"] < 0.20:
        adjustment -= 2
        
    # 高年級上修條件：高深連詞 OR 專業抽象詞匯 >= 2 OR (長字數 + 高 MDD)
    if "進階論述句" in features["clause_types"] or features["vocab_depth"] >= 2:
        adjustment += 2
    elif features["char_count"] >= 35 and features["noun_ratio"] >= 0.35:
        adjustment += 2

    # 3. 輸出最終適用年級
    final_grade = base_grade + adjustment
    if final_grade >= 5:
        return "5-6 年級 (高年級)"
    elif final_grade <= 2:
        return "1-2 年級 (低年級)"
    else:
        return "3-4 年級 (中年級)"

def run_batch_analysis(question_list, nlp_model, difficulty_model):
    """批次執行運算並輸出 DataFrame 報告"""
    results = []
    for q_text in question_list:
        feat = calculate_features(q_text, nlp_model)
        grade = predict_grade(feat, difficulty_model)
        
        results.append({
            "題目內容": q_text,
            "預估適用年級": grade,
            "複句結構與句式": feat["clause_types"],
            "總字數": feat["char_count"],
            "名詞密度": f"{feat['noun_ratio']:.1%}",
            "MDD數值": round(feat["mdd"], 2)
        })
    return pd.DataFrame(results)

# ==========================================
# 4. 前端介面與互動
# ==========================================
with st.sidebar:
    st.header("⚙️ 系統狀態")
    nlp = load_nlp()
    st.success("✅ spaCy 中文模型已載入")
    
    model = load_difficulty_model()
    if model is not None:
        st.success("✅ 已成功載入 mdd_baseline_model.pkl (已啟用語境校正)")
    else:
        st.warning("⚠️ 未找到 mdd_baseline_model.pkl，使用台灣試題進階評分引擎。")
        
    st.divider()
    st.markdown("### 📊 分析設定")
    subject = st.selectbox("學科", ["國語文", "數學", "社會", "自然"])
    show_table = st.checkbox("顯示特徵明細表", value=True)

st.title("📚 台灣中小學試題句子難度檢測系統")
st.caption("支援單題檢測、句式特徵解析，以及多題文字貼上／檔案上傳的批次檢測。")

# 使用分頁區隔功能
tab1, tab2 = st.tabs(["✍️ 單題檢測與複句分析", "📋 批次多題文字與題庫檢測"])

# --- TAB 1: 單題檢測 ---
with tab1:
    question_text = st.text_area(
        "題目文字",
        height=130,
        placeholder="請將試題文字貼在這裡...（例如：小明每天早上七點起床，吃完早餐後去上學。）"
    )

    if st.button("🚀 開始檢測單題", type="primary", key="btn_single"):
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
                        "特徵名稱": [
                            "總字數", "總詞數", "名詞出現比例", 
                            "動詞出現比例", "MDD 依存距離", 
                            "學科進階術語計數", "句式與複句分類"
                        ],
                        "數值": [
                            f"{features['char_count']}",
                            f"{features['word_count']}",
                            f"{features['noun_ratio']:.1%}",
                            f"{features['verb_ratio']:.1%}",
                            f"{features['mdd']:.2f}",
                            f"{features['vocab_depth']} 個",
                            f"{features['clause_types']}"
                        ]
                    })
                    st.table(detail_df)

# --- TAB 2: 批次查詢 (支援直接貼上文字 & 上傳檔案) ---
with tab2:
    st.markdown("### 批次多題檢測")
    st.caption("請依照習慣選擇 **「直接貼上多行文字」** 或 **「上傳 CSV / Excel 試算表」**：")
    
    batch_mode = st.radio(
        "請選擇輸入方式：", 
        ["📋 貼上多行題目文字", "📂 上傳 CSV / Excel 檔案"], 
        horizontal=True
    )
    
    if batch_mode == "📋 貼上多行題目文字":
        default_sample = (
            "小明每天早上七點起床，吃完早餐後去上學。\n"
            "下雨了，媽媽提醒我出門要帶雨傘。\n"
            "如果植物沒有足夠的陽光和水分，就可能無法健康生長。\n"
            "雖然今天很熱，但是大家還是認真完成體育課的活動。\n"
            "當我們觀察天氣變化時，可以記錄氣溫、雲量和降雨情形，再比較不同日期的差異。\n"
            "水受熱後會蒸發，遇冷又可能凝結成小水滴。\n"
            "由於人口集中在都市，交通便利的同時也可能造成空氣污染與居住壓力。\n"
            "如果一個物體受到的合力不為零，它的運動狀態就可能發生改變。\n"
            "雖然網路資訊取得方便，但若沒有查證來源，使用者可能誤信不完整或錯誤的內容。\n"
            "當政府推動公共政策時，除了考量經濟效益，也必須評估環境影響與社會公平。\n"
            "細胞會透過分裂產生新細胞，以維持生物體的生長與修復。\n"
            "若市場需求增加而供給無法同步提升，商品價格通常會出現上漲壓力。\n"
            "文學作品中的敘事觀點會影響讀者理解角色動機與事件意義的方式。\n"
            "即使科技能提升資訊傳播效率，演算法所形成的同溫層仍可能限制人們接觸多元觀點的機會。\n"
            "在分析歷史事件時，研究者必須同時考慮政治結構、經濟條件與社會文化脈絡，才能避免單一因果解釋造成的偏誤。"
        )
        batch_text = st.text_area(
            "請貼上多個題目（每行一題，空行會自動忽略）：",
            value=default_sample,
            height=250
        )
        
        if st.button("⚡ 開始批次分析 (文字)", type="primary", key="btn_batch_text"):
            q_list = [line.strip() for line in batch_text.split("\n") if line.strip()]
            
            if not q_list:
                st.error("請至少貼上一題有效的題目內容喔！")
            else:
                with st.spinner(f"正在分析共 {len(q_list)} 筆試題..."):
                    res_df = run_batch_analysis(q_list, nlp, model)
                    st.success(f"🎉 已完成 {len(q_list)} 題批次檢測！分析結果如下：")
                    st.dataframe(res_df, use_container_width=True)
                    
                    csv_data = res_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="📥 下載分析報告 (CSV)",
                        data=csv_data,
                        file_name="文字批次試題檢測報告.csv",
                        mime="text/csv"
                    )
                    
    else:
        st.info("請確保上傳的檔案中，包含一個標題名稱為 **「題目」** 或 **「question」** 的欄位。")
        uploaded_file = st.file_uploader("請選擇 CSV 或 Excel 檔案", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                    
                text_col = next((col for col in df.columns if col in ["題目", "question", "Text", "試題", "內容"]), None)
                
                if not text_col:
                    st.error("❌ 找不到有效的題目欄位！請確認表格有「題目」或「question」的標題欄位。")
                else:
                    st.write(f"成功載入 `{len(df)}` 筆試題，即將為欄位 `[{text_col}]` 進行分析：")
                    if st.button("⚡ 開始批次分析 (檔案)", type="primary", key="btn_batch_file"):
                        q_list = [str(text).strip() for text in df[text_col] if pd.notna(text) and str(text).strip()]
                        with st.spinner("正在逐題批次檢測..."):
                            res_df = run_batch_analysis(q_list, nlp, model)
                            st.success("🎉 批次分析完成！結果如下：")
                            st.dataframe(res_df, use_container_width=True)
                            
                            csv_data = res_df.to_csv(index=False).encode("utf-8-sig")
                            st.download_button(
                                label="📥 下載完整檢測結果 (CSV)",
                                data=csv_data,
                                file_name="檔案試題檢測結果.csv",
                                mime="text/csv"
                            )
            except Exception as e:
                st.error(f"檔案讀取失敗，詳細錯誤：{e}")
