# Research Agent

一個基於 Streamlit 的研究代理程式，使用多個 AI Agent 進行序列式研究分析。

## 功能特色

- **Topic Refiner**: 將研究主題細化為具體的子問題
- **Researcher**: 模擬研究發現和資料來源
- **Summarizer**: 生成完整的研究報告

## 安裝與執行

### 1. 建立虛擬環境

```bash
python3 -m venv .venv
```

### 2. 啟動虛擬環境

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

### 3. 安裝相依套件

```bash
pip install -r requirements.txt
```

### 4. 設定環境變數

複製範本檔案並設定您的 API key：

```bash
cp env.example .env
```

編輯 `.env` 檔案，將 `your_openai_api_key_here` 替換為您的實際 API key：

```bash
# OpenAI API 設定
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 5. 執行程式

```bash
streamlit run app.py
```

## 使用方式

1. 確保已按照上述步驟設定 `.env` 檔案
2. 執行程式，如果沒有 `.env` 檔案會顯示錯誤訊息
3. 輸入研究主題（預設為「最新多語言 LLM fine-tuning 方法」）
4. 調整 Temperature 參數
5. 點擊「Start Research 🚀」開始研究

## 輸出格式

程式會生成：
- 細化的研究問題
- 模擬的研究發現
- 完整的研究報告（可下載為 .txt 檔案）
- JSON 格式的完整輸出

## 注意事項

- 必須設定 `.env` 檔案並包含有效的 OpenAI API Key
- 程式會自動從 `.env` 檔案載入 API Key
- 如果沒有 `.env` 檔案或找不到 API Key，程式會顯示錯誤並停止執行
- 程式會模擬研究結果，並非真實的網路搜尋
