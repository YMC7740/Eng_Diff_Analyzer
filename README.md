# 測驗卷難度分析互動式網站

這是一個 Streamlit 互動式網站，用來分析測驗題文字的初步難度訊號。

## 功能

- 單題分析：輸入題幹後顯示 MDD、字數、語意標記與預估年級
- 批次分析：逐行貼上多題，或上傳含有「題目」欄位的 CSV
- 模型整合：若有 `mdd_baseline_model.pkl`，會使用該模型預測
- 備用規則：若沒有模型檔，會改用 rule-based 預估

## Windows 快速啟動

在專案資料夾中執行：

```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

或直接雙擊 `start.bat`。

第一次啟動會自動建立 `.venv`、安裝套件，並下載 spaCy 中文模型。完成後瀏覽器開啟：

```text
http://localhost:8501
```

## 手動安裝

```bash
pip install -r requirements.txt
python -m spacy download zh_core_web_sm
```

## 手動執行

```bash
streamlit run app.py
```

## CSV 格式

批次上傳 CSV 至少要包含「題目」欄位，可選填「學科」欄位：

```csv
題目,學科
如果植物長期缺少陽光，可能會因為無法順利行光合作用而逐漸枯萎。,自然
```

## 模型檔

若要使用真實 ML 模型，請將 `mdd_baseline_model.pkl` 放在專案根目錄，也就是與 `app.py` 同一層。
