# AI Research Agent - LangChain 版本

一個基於 **LangChain Agent** 架構的 AI research agent，能夠搜尋真實的學術論文並生成完整的研究報告。

<video src="demo/research_agent_demo_v2.mp4" width="800" controls muted loop playsinline>
  Your browser does not support the video tag.
  <a href="demo/research_agent_demo_v2.mp4">Download the demo video</a>
</video>

## 🎯 功能特色

### 三個專業 AI Agent 協作

1. **🏗️ Topic Refiner Agent**
   - 將廣泛的研究主題細化為 3-5 個具體、可研究的子問題
   - 為每個問題生成清晰的說明（clarifier）
   - 解析和格式化研究問題

2. **📏 Researcher Agent**
   - **真實 arXiv 論文搜尋**：搜尋並分析最新的學術論文
   - 自動過濾重複論文，確保每個問題都有獨特的研究結果
   - 提取關鍵研究發現、評估可信度和資料品質
   - 提供論文詳情：標題、作者、摘要、分類、連結

3. **✍️ Summarizer Agent**
   - 整合所有研究結果生成完整的學術報告
   - 包含：Abstract、Introduction、Key Findings、Limitations、Future Work
   - 正確格式化的引用文獻

## 🚀 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 設定環境變數

建立 `.env` 檔案：

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 執行應用程式

```bash
streamlit run app_langchain.py
```

## 📊 系統架構

### LangChain Agent 工作流程

```
輸入研究主題
    ↓
Topic Refiner Agent
(細化為具體問題)
    ↓
Researcher Agent
(搜尋 arXiv 論文)
    ↓
Summarizer Agent
(生成完整報告)
    ↓
輸出研究報告
```

### 真實資料來源

- **arXiv API**：即時搜尋最新學術論文
- **去重**：自動過濾重複論文
- **結果驗證**：每條發現都有具體論文來源

## 🎨 介面特色

- **簡潔清晰的問題顯示**：每個問題都有清楚的格式和說明
- **詳細的論文資訊**：顯示論文標題、作者、摘要、分類和連結
- **進度指示器**：實時顯示研究進度
- **可下載報告**：支援文字報告和 JSON 格式下載

## 📋 使用範例

1. **輸入研究主題**：例如「最新 RAG 的方法」
2. **點擊「Start Research 🚀」**
3. **等待 Agent 執行**（約 3-5 分鐘）
4. **查看結果**：
   - 細化的研究問題
   - 每個問題的 arXiv 論文和研究發現
   - 完整的研究報告
5. **下載結果**：可下載文字報告或完整 JSON 資料

## 🔧 技術細節

### 核心技術

- **LangChain Agent**：使用 LangChain 架構實現 multi-agent 協作
- **OpenAI GPT-4o-mini**：作為 LLM 引擎
- **arXiv API**：真實的學術論文搜尋
- **Streamlit**：Web 介面

## 📈 研究流程

### 階段 1: Topic Refiner
- 接收研究主題
- 生成 3-5 個具體研究問題
- 為每個問題添加說明

### 階段 2: Researcher
- 針對每個問題搜尋 arXiv 論文
- 分析論文內容並提取關鍵發現
- 評估資料可信度和品質

### 階段 3: Summarizer
- 整合所有研究結果
- 生成結構化的學術報告
- 包含完整的引用格式

## 🎯 輸出內容

### 研究問題
- 格式化的問題列表
- 每個問題的詳細說明
- 可展開查看原始 JSON 輸出

### 研究結果
- 每個問題的研究發現
- 真實的 arXiv 論文詳情
- 可信度和資料品質評估
- 論文連結和分類

### 最終報告
- Abstract（摘要）
- Introduction（引言）
- Key Findings（主要發現）
- Limitations（限制）
- Future Work（未來工作）
- References（參考文獻）

## ⚙️ 設定

### 預設參數

- **模型**：gpt-4o-mini
- **Temperature**：0.1（固定）
- **每題論文數**：最多 3 篇
- **論文去重**：自動過濾重複論文

## 📝 注意事項

- ⚠️ **API Key 必須設定**：需要有效的 OpenAI API Key
- ⏱️ **執行時間**：完整研究流程約需 3-5 分鐘
- 📚 **真實資料**：使用的是真實的 arXiv 論文資料
- 🔍 **網路連線**：需要網路連線以搜尋 arXiv API

## 🐛 故障排除

### 常見問題

1. **API Key 錯誤**
   - 確保 `.env` 檔案存在且包含正確的 API Key
   - 檢查 API Key 是否有效且有足夠額度

2. **arXiv 搜尋失敗**
   - 檢查網路連線
   - 確保網路可以訪問 arXiv API

## 🔗 相關資源

- [LangChain 文件](https://python.langchain.com/)
- [arXiv API 文件](https://arxiv.org/help/api)
- [Streamlit 文件](https://docs.streamlit.io/)
