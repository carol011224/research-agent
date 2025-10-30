import streamlit as st
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
import requests
import feedparser

# 是否使用 Agent（LLM 自主選擇工具）。預設 False，保持固定順序。
USE_AGENT = False

# ----------------------
# 載入環境變數
# ----------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ 無法讀取 OPENAI_API_KEY")
    st.stop()

# ----------------------
# Streamlit 設定
# ----------------------
st.set_page_config(page_title="LangChain Research Agent", layout="wide")
st.title("🤖 LangChain Research Agent")
st.write("AI Agent 研究系統：Topic Refiner → Researcher → Summarizer")

with st.sidebar:
    topic = st.text_input("Research topic", value="最新 RAG 的方法")
    st.markdown("---")

# ----------------------
# LangChain LLM 初始化
# ----------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    streaming=False,
    api_key=api_key
)

# ----------------------
# arXiv 搜尋函數
# ----------------------
def search_arxiv(query, max_results=3, start_offset=0):
    try:
        clean_query = query.replace('?', '').replace('!', '').replace('。', '').replace('？', '').strip()
        query_words = clean_query.split()
        if len(query_words) > 5:
            clean_query = ' '.join(query_words[:5])
        url = "https://export.arxiv.org/api/query"
        params = {
            'search_query': f'ti:{clean_query} OR abs:{clean_query}',
            'start': start_offset,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/atom+xml'}
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        papers = []
        for entry in feed.entries:
            paper_id = entry.id.split('/')[-1]
            summary = entry.summary.replace('\n', ' ').strip()
            if len(summary) > 500:
                summary = summary[:500] + "..."
            papers.append({
                'title': entry.title,
                'authors': [author.name for author in entry.authors],
                'summary': summary,
                'published': entry.published,
                'arxiv_id': paper_id,
                'url': entry.id,
                'categories': [tag.term for tag in entry.tags]
            })
        return papers
    except Exception as e:
        print(f"arXiv 搜尋失敗: {e}")
        return []

# ----------------------
# 封裝成 LangChain Tools
# ----------------------
def topic_refiner_tool(input_topic: str) -> str:
    """將研究主題細化為 3-5 個具體子問題並附 clarifier，輸出 JSON 字串。"""
    prompt = (
        f"你是一名資深研究助理。針對主題「{input_topic}」，生成 3~5 個具體、可研究的子問題，"
        "每個子問題需包含 question 與 clarifier。只輸出 JSON，不要加多餘文字。\n"
        "JSON 範例：\n"
        "[\n  {\n    \"question\": \"{子問題一}\",\n    \"clarifier\": \"{補充說明}\"\n  },\n  {\n    \"question\": \"{子問題二}\",\n    \"clarifier\": \"{補充說明}\"\n  }\n]"
    )
    resp = llm.invoke([
        SystemMessage(content="You are a helpful research assistant."),
        HumanMessage(content=prompt)
    ])
    return getattr(resp, "content", str(resp))

def researcher_tool(questions_json: str) -> str:
    """根據子問題 JSON，查詢真實 arXiv 論文並產出分析結果，輸出 JSON 字串。"""
    questions = json.loads(questions_json)
    results = []
    seen_ids = set()
    for i, q in enumerate(questions):
        question = q.get("question", "")
        # 對齊 app_langchain：每題使用不同 offset 提高去重效果
        start_offset = i * 5
        papers = search_arxiv(question, max_results=5, start_offset=start_offset)
        # 過濾重複
        unique_papers = [p for p in papers if p['arxiv_id'] not in seen_ids]
        for p in unique_papers:
            seen_ids.add(p['arxiv_id'])
        # 只取前三篇
        unique_papers = unique_papers[:3]
        # 使用 LLM 分析論文
        if unique_papers:
            prompt = (
                "你是一名研究員。針對子問題：" + question + "\n"
                "以下為真實的 arXiv 論文資料（title/authors/summary/published/url/categories）：\n" +
                json.dumps(unique_papers, ensure_ascii=False, indent=2) + "\n\n"
                "請只輸出 JSON（不要多餘文字），結構如下：\n"
                "{\n  \"findings\": [\"基於論文的發現1\", \"基於論文的發現2\"],\n"
                "  \"sources\": [\"arXiv 論文標題或連結\"],\n"
                "  \"confidence\": \"low|medium|high\",\n"
                "  \"data_quality\": \"good|limited\",\n"
                "  \"notes\": \"補充說明（如限制/不足）\"\n}"
            )
        else:
            prompt = (
                "你是一名研究員。針對子問題：" + question + " 未找到相關 arXiv 論文。\n"
                "請只輸出 JSON（不要多餘文字），結構如下：\n"
                "{\n  \"findings\": [\"一般知識的發現1\", \"一般知識的發現2\"],\n"
                "  \"sources\": [\"一般知識\"],\n"
                "  \"confidence\": \"low\",\n"
                "  \"data_quality\": \"limited\",\n"
                "  \"notes\": \"未找到學術論文，僅提供一般知識\"\n}"
            )
        resp = llm.invoke([
            SystemMessage(content="You are a helpful research assistant."),
            HumanMessage(content=prompt)
        ])
        raw = getattr(resp, "content", str(resp))
        results.append({"question": question, "analysis": raw, "arxiv_papers": unique_papers})
    return json.dumps(results, ensure_ascii=False)

def summarizer_tool(research_results_json: str) -> str:
    """根據研究結果 JSON 生成完整研究報告（≤600字），返回純文字。"""
    prompt = (
        "你是一名科學作家。請根據以下研究結果（含每個子問題的 findings/sources/confidence/data_quality/notes）撰寫最終研究報告。\n"
        + research_results_json + "\n\n"
        "請以 Markdown 輸出，並使用中文小節標題（作為副標題）如下：\n"
        "### 摘要\n(約 3 行)\n\n"
        "### 引言\n(1 段)\n\n"
        "### 主要發現\n(整合每個子問題的重點條列)\n\n"
        "### 限制\n(1 段)\n\n"
        "### 建議的下一步\n(3 條條列)\n\n"
        "字數 ≤ 600；只輸出上述 Markdown 內容，不要多餘說明或 JSON。"
    )
    resp = llm.invoke([
        SystemMessage(content="You are a helpful research assistant."),
        HumanMessage(content=prompt)
    ])
    return getattr(resp, "content", str(resp))

# ----------------------
# 定義 Tools
# ----------------------
tools = [
    topic_refiner_tool,
    researcher_tool,
    summarizer_tool
]

# ----------------------
# 初始化 session state（比照 app_langchain）
# ----------------------
if 'research_completed' not in st.session_state:
    st.session_state.research_completed = False
if 'research_data' not in st.session_state:
    st.session_state.research_data = {}
if 'current_topic' not in st.session_state:
    st.session_state.current_topic = ""

# ----------------------
# Main UI（比照 app_langchain）
# ----------------------
if st.button("Start Research 🚀"):
    # 清空之前的結果
    st.session_state.research_completed = False
    st.session_state.research_data = {}
    st.session_state.current_topic = topic

    st.subheader("🤖 LangChain Agent 執行")

    # 進度顯示
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        if USE_AGENT:
            # 使用 LangChain Agent 讓 LLM 自主挑選工具
            status_text.text("🤖 正在以 Agent 模式執行...")
            progress_bar.progress(30)
            agent = create_agent(model=llm, tools=tools, system_prompt="You are a helpful research assistant.")
            result = agent.invoke({"messages": [{"role": "user", "content": topic}]})
            # 簡單抽取回覆文字
            result_report = ""
            if isinstance(result, dict):
                msgs = result.get("messages")
                if isinstance(msgs, list):
                    for m in reversed(msgs):
                        if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str):
                            result_report = m["content"]
                            break
            if not result_report:
                result_report = str(result)

            # Agent 模式下，維持 UI：僅展示最終報告
            questions = []
            research_results = []
            raw_build = ""
            final_report = result_report
            status_text.text("✅ Agent 執行完成！")
            progress_bar.progress(100)
        else:
            # 固定順序：Topic → Researcher → Summarizer
            status_text.text("🏗️ 正在細化主題 (Topic Refiner)...")
            progress_bar.progress(20)

            topic_json = topic_refiner_tool(topic)
            # 解析問題列表，參考 app_langchain 的邏輯
            questions: list = []
            raw_build = topic_json
            try:
                parsed = json.loads(topic_json.strip())
                if isinstance(parsed, dict) and 'sub_questions' in parsed:
                    questions_list = parsed['sub_questions']
                elif isinstance(parsed, list):
                    questions_list = parsed
                else:
                    questions_list = []
                for i, q in enumerate(questions_list, 1):
                    if isinstance(q, dict):
                        questions.append({
                            "id": f"Q{i}",
                            "question": q.get("question", ""),
                            "clarifier": q.get("clarifier", "")
                        })
            except Exception:
                questions = []

            if not questions:
                # 後備：簡單生成三個樣板問題
                questions = [
                    {"id": "Q1", "question": f"{topic}的基本原理是什麼？", "clarifier": f"關於 {topic} 的基本概念"},
                    {"id": "Q2", "question": f"{topic}的最新發展趨勢如何？", "clarifier": f"關於 {topic} 的發展現況"},
                    {"id": "Q3", "question": f"{topic}的應用領域有哪些？", "clarifier": f"關於 {topic} 的實際應用"},
                ]

            status_text.text("📏 正在搜尋並分析論文 (Researcher)...")
            progress_bar.progress(55)

            researcher_json = researcher_tool(json.dumps(questions, ensure_ascii=False))
            research_results = []
            try:
                tmp_list = json.loads(researcher_json)
                for item in tmp_list:
                    analysis_raw = item.get("analysis", "")
                    try:
                        analysis_obj = json.loads(analysis_raw)
                    except Exception:
                        analysis_obj = {
                            "findings": [],
                            "sources": [],
                            "confidence": "unknown",
                            "data_quality": "unknown",
                            "notes": "raw text",
                            "__raw": analysis_raw
                        }
                    analysis_obj['arxiv_papers'] = item.get('arxiv_papers', [])
                    research_results.append({
                        "id": None,
                        "question": item.get("question", ""),
                        "analysis": analysis_obj
                    })
            except Exception:
                research_results = []

            status_text.text("✍️ 正在整合報告 (Summarizer)...")
            progress_bar.progress(80)

            final_report = summarizer_tool(json.dumps(research_results, ensure_ascii=False))

            status_text.text("✅ Agent 執行完成！")
            progress_bar.progress(100)

    except Exception as e:
        status_text.text(f"❌ Agent 執行失敗: {str(e)}")
        progress_bar.progress(0)
        st.error(f"Agent 執行錯誤: {str(e)}")
        st.stop()

    # 展示結果（比照 app_langchain）
    st.subheader("🏗️ Topic Refiner")
    st.markdown("### 📋 細化的研究問題")
    if questions:
        for i, q in enumerate(questions, 1):
            qid = q.get('id', f'Q{i}')
            qtext = q.get('question', '')
            clar = q.get('clarifier', '')
            st.write(f"**📌 {qid}:** {qtext}")
            if clar:
                st.write(f"💡 {clar}")
            st.write("")

    with st.expander("🔍 查看原始輸出 (Raw Output)", expanded=False):
        st.code(raw_build[:3000], language="json")

    st.subheader("📏 Researcher")
    st.write("**Research results:**")
    for i, result in enumerate(research_results):
        with st.expander(f"研究問題 {i+1}: {result['question']}"):
            analysis = result['analysis']
            st.write("**研究發現:**")
            for finding in analysis.get('findings', []):
                st.write(f"• {finding}")
            st.write("**資料來源:**")
            for source in analysis.get('sources', []):
                st.write(f"• {source}")
            st.write(f"**可信度:** {analysis.get('confidence', 'unknown')}")
            st.write(f"**資料品質:** {analysis.get('data_quality', 'unknown')}")
            arxiv_papers = analysis.get('arxiv_papers', [])
            if arxiv_papers:
                st.write("**📚 arXiv 論文:**")
                for paper in arxiv_papers:
                    st.write(f"📄 **[{paper['title']}]({paper['url']})**")
                    st.write(f"   👥 作者: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
                    st.write(f"   📅 發布: {paper['published'][:10]}")
                    st.write(f"   📝 摘要: {paper['summary'][:500]}...")
                    st.write(f"   🏷️ 分類: {', '.join(paper['categories'][:3])}")
                    st.write("---")
            else:
                st.write("**⚠️ 未找到相關 arXiv 論文**")

    with st.expander("📊 完整研究結果 (JSON)", expanded=False):
        st.write(research_results)

    st.subheader("✍️ Summarizer")
    st.markdown(final_report)

    st.session_state.research_data = {
        'questions': questions,
        'raw_build': raw_build,
        'research_results': research_results,
        'final_report': final_report,
        'topic': topic,
        'agent_output': {
            'questions': questions,
            'research_results': research_results,
            'final_report': final_report
        }
    }

    st.download_button("Download report (.txt)", data=final_report, file_name="research_report.txt")
    st.download_button("Download JSON export", data=json.dumps(st.session_state.research_data, ensure_ascii=False, indent=2), file_name="research_export.json")
    st.session_state.research_completed = True

elif st.session_state.research_completed and st.session_state.research_data:
    data = st.session_state.research_data

    st.subheader("🏗️ Topic Refiner")
    if data['questions']:
        for i, q in enumerate(data['questions'], 1):
            qid = q.get('id', f'Q{i}')
            qtext = q.get('question', '')
            clar = q.get('clarifier', '')
            st.write(f"**📌 {qid}:** {qtext}")
            if clar:
                st.write(f"💡 {clar}")
            st.write("")

    with st.expander("🔍 查看原始輸出 (Raw Output)", expanded=False):
        st.code(data['raw_build'][:3000], language="json")

    st.subheader("📏 Researcher")
    with st.expander("📊 完整研究結果 (JSON)", expanded=False):
        st.write(data['research_results'])

    st.subheader("✍️ Summarizer")
    st.markdown(data['final_report'])

    st.download_button("Download report (.txt)", data=data['final_report'], file_name="research_report.txt")
    st.download_button("Download JSON export", data=json.dumps(data, ensure_ascii=False, indent=2), file_name="research_export.json")
