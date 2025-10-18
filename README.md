# Research Agent

一個基於 Streamlit 的研究代理程式，使用多個 AI Agent 進行序列式研究分析，並從 arXiv 獲取真實的學術論文資料。

## 功能特色

- **Topic Refiner**: 將研究主題細化為具體的子問題
- **Researcher**: 從 arXiv 搜尋最新學術論文並分析研究發現
- **Summarizer**: 基於真實論文資料生成完整的研究報告

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

## 工作流程

1. **Topic Refiner**: AI 將您的研究主題細化為 3-5 個具體的子問題
2. **Researcher**: 
   - 為每個子問題搜尋 arXiv 最新論文（每個問題 2-3 篇）
   - AI 分析論文摘要和標題，提取研究發現
   - 評估資料來源的可信度和品質
3. **Summarizer**: AI 整合所有研究發現，生成完整的研究報告

## 輸出格式

程式會生成：
- 細化的研究問題
- 基於真實 arXiv 論文的研究發現
- 論文連結、作者、發布日期等詳細資訊
- 完整的研究報告（可下載為 .txt 檔案）
- JSON 格式的完整輸出

## 資料來源

- **arXiv API**: 免費的學術論文搜尋，無使用限制
- **OpenAI API**: 用於分析和總結研究資料

## 注意事項

- 必須設定 `.env` 檔案並包含有效的 OpenAI API Key
- 程式會自動從 `.env` 檔案載入 API Key
- 如果沒有 `.env` 檔案或找不到 API Key，程式會顯示錯誤並停止執行
- 研究結果基於真實的 arXiv 學術論文，確保資料的可信度
- 程式會自動處理 API 速率限制，確保穩定運行
