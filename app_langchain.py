import streamlit as st
from openai import OpenAI
import json
import os
from dotenv import load_dotenv

# LangChain 導入
from langchain_openai import ChatOpenAI
from typing import Dict, Any, List

# 載入環境變數
load_dotenv()

st.set_page_config(page_title="LangChain Research Agent", layout="wide")
st.title("🤖 LangChain Research Agent")
st.write("AI Agent 研究系統：Topic Refiner → Researcher → Summarizer")

# ----------------------
# 檢查 API Key
# ----------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ 無法讀取 .env 檔案或找不到 OPENAI_API_KEY")
    st.stop()

# ----------------------
# Sidebar: 主題設定
# ----------------------
with st.sidebar:
    topic = st.text_input("Research topic", value="最新 RAG 的方法")
    st.markdown("---")

# ----------------------
# LangChain 設定
# ----------------------
llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-4o-mini",
    temperature=0.1,
    streaming=False
)

# ----------------------
# LLM wrapper and tools
# ----------------------
def llm_chat(prompt, api_key, model="gpt-4o-mini", temperature=0.1, max_tokens=800):
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM call failed] {e}"

def topic_refiner(topic, api_key):
    prompt = f"你是一名資深研究助理。針對主題「{topic}」，生成 3~5 個具體、可研究的子問題，並為每個子問題附上簡短 clarifier。輸出 JSON 格式。"
    raw = llm_chat(prompt, api_key)
    
    # 清理 raw 輸出，移除可能的 markdown 標記
    clean_raw = raw.strip()
    if clean_raw.startswith("```json"):
        clean_raw = clean_raw[7:]  # 移除 ```json
    if clean_raw.startswith("```"):
        clean_raw = clean_raw[3:]   # 移除 ```
    if clean_raw.endswith("```"):
        clean_raw = clean_raw[:-3]  # 移除結尾的 ```
    clean_raw = clean_raw.strip()
    
    try:
        parsed = json.loads(clean_raw)
        
        # 處理可能的 sub_questions 結構
        if isinstance(parsed, dict) and 'sub_questions' in parsed:
            questions_list = parsed['sub_questions']
        elif isinstance(parsed, list):
            questions_list = parsed
        else:
            questions_list = []
        
        # 轉換為標準格式
        formatted_questions = []
        for i, q in enumerate(questions_list, 1):
            if isinstance(q, dict):
                formatted_questions.append({
                    "id": f"Q{i}",
                    "question": q.get("question", ""),
                    "clarifier": q.get("clarifier", "")
                })
        
        if not formatted_questions:
            raise ValueError("No valid questions found")
            
        return formatted_questions, raw
        
    except Exception as e:
        # 如果 JSON 解析失敗，嘗試提取問題
        questions = []
        for line in raw.splitlines():
            line = line.strip()
            # 尋找包含 "question" 的行
            if '"question"' in line:
                try:
                    # 提取問題文字
                    question_match = line.split('"question"')[1].split('"')[1]
                    if question_match and len(question_match) > 10:
                        questions.append({
                            "id": f"Q{len(questions)+1}",
                            "question": question_match,
                            "clarifier": ""
                        })
                except:
                    pass
        
        if questions:
            return questions, raw
        else:
            # 最後的備用方案
            fallback_questions = [
                {"id": "Q1", "question": f"{topic}的基本原理是什麼？", "clarifier": f"關於 {topic} 的基本概念"},
                {"id": "Q2", "question": f"{topic}的最新發展趨勢如何？", "clarifier": f"關於 {topic} 的發展現況"},
                {"id": "Q3", "question": f"{topic}的應用領域有哪些？", "clarifier": f"關於 {topic} 的實際應用"}
            ]
            return fallback_questions, raw

def search_arxiv(query, max_results=3, start_offset=0):
    """搜尋 arXiv 最新論文"""
    try:
        import requests
        import feedparser
        
        # 簡化搜尋查詢，移除特殊字符
        clean_query = query.replace('?', '').replace('!', '').replace('。', '').replace('？', '')
        clean_query = clean_query.strip()
        
        # 提取關鍵字（如果查詢太長，只取前幾個關鍵字）
        query_words = clean_query.split()
        if len(query_words) > 5:
            # 如果查詢太長，只使用前 5 個字作為關鍵字
            clean_query = ' '.join(query_words[:5])
        
        # 使用 HTTPS
        url = "https://export.arxiv.org/api/query"
        # 改用更精確的搜尋：優先標題，然後摘要
        params = {
            'search_query': f'ti:{clean_query} OR abs:{clean_query}',
            'start': start_offset,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        # 添加 headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; Research Agent)',
            'Accept': 'application/atom+xml'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        
        feed = feedparser.parse(response.content)
        
        papers = []
        for entry in feed.entries:
            # 提取論文 ID
            paper_id = entry.id.split('/')[-1]
            
            # 清理摘要
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

def researcher(questions, api_key):
    """真正的研究員 - 搜尋真實的 arXiv 論文"""
    results = []
    seen_paper_ids = set()  # 記錄已看過的論文 ID，避免重複
    
    for i, q in enumerate(questions):
        question = q['question']
        
        # 1. 搜尋 arXiv 真實論文
        # 為每個問題使用不同的 start_offset，避免返回相同論文
        start_offset = i * 5  # 增加間距，確保每個問題獲取不同的論文
        arxiv_papers = search_arxiv(question, max_results=5, start_offset=start_offset)
        
        # 過濾掉已經看過的論文
        unique_papers = []
        for paper in arxiv_papers:
            paper_id = paper.get('arxiv_id', '')
            if paper_id and paper_id not in seen_paper_ids:
                unique_papers.append(paper)
                seen_paper_ids.add(paper_id)
        
        # 如果過濾後沒有論文，再搜尋一次（不使用 start_offset）
        if not unique_papers:
            fallback_papers = search_arxiv(question, max_results=5, start_offset=0)
            for paper in fallback_papers:
                paper_id = paper.get('arxiv_id', '')
                if paper_id and paper_id not in seen_paper_ids:
                    unique_papers.append(paper)
                    seen_paper_ids.add(paper_id)
        
        # 只取前 3 篇論文
        arxiv_papers = unique_papers[:3]
        
        # 2. 使用 LLM 分析真實論文資料
        if arxiv_papers:
            prompt = f"""你是一名研究員。針對子問題「{question}」，我已經搜尋到以下真實的 arXiv 論文：

{json.dumps(arxiv_papers, ensure_ascii=False, indent=2)}

請根據這些真實論文進行分析：
1) 提取 2-3 條具體的研究發現（基於論文摘要和標題）
2) 評估資料來源的可信度（arXiv 學術論文）
3) 提供可信度評估 (low/medium/high)
4) 如果資料不足，請註明

輸出 JSON 格式：
{{
    "findings": ["基於論文的發現1", "基於論文的發現2"],
    "sources": ["arXiv論文來源"],
    "confidence": "medium",
    "data_quality": "good/limited",
    "notes": "額外說明或資料限制"
}}"""
        else:
            prompt = f"""你是一名研究員。針對子問題「{question}」，我沒有找到相關的 arXiv 論文。

請提供：
1) 基於一般知識的 2-3 條相關發現
2) 註明資料來源限制
3) 提供可信度評估 (low/medium/high)

輸出 JSON 格式：
{{
    "findings": ["一般知識發現1", "一般知識發現2"],
    "sources": ["一般知識"],
    "confidence": "low",
    "data_quality": "limited",
    "notes": "未找到相關學術論文，基於一般知識"
}}"""
        
        raw = llm_chat(prompt, api_key)
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {
                "findings": [f"arXiv論文: {paper['title'][:50]}..." for paper in arxiv_papers[:2]] if arxiv_papers else ["未找到相關論文"],
                "sources": ["arXiv"] if arxiv_papers else ["一般知識"],
                "confidence": "medium" if arxiv_papers else "low",
                "data_quality": "good" if arxiv_papers else "limited",
                "notes": "JSON parsing failed, showing raw data",
                "__raw": raw
            }
        
        # 添加原始論文資料
        parsed['arxiv_papers'] = arxiv_papers
        
        results.append({
            "id": q.get("id"), 
            "question": q.get("question"), 
            "analysis": parsed
        })
    
    return results

def summarizer(research_results, api_key, topic):
    prompt = f"你是一名科學作家。根據下列 Researcher 輸出結果生成完整研究報告:\nTopic: {topic}\nResults: {json.dumps(research_results, ensure_ascii=False, indent=2)}\n- Abstract 3行\n- Introduction 1段\n- Key findings (整合每個子問題)\n- Limitations 1段\n- Recommended next steps 3 bullets\n字數 ≤600。"
    raw = llm_chat(prompt, api_key, temperature=0.15, max_tokens=1000)
    return raw


# ----------------------
# LangChain Agent 定義（簡化版）
# ----------------------
def create_research_agent():
    """建立研究代理（修復版）"""
    def agent_run(topic: str) -> dict:
        """執行研究流程，返回原始資料結構"""
        # 步驟 1: 主題細化
        questions, raw_build = topic_refiner(topic, api_key)
        
        # 步驟 2: 研究問題
        research_results = researcher(questions, api_key)
        
        # 步驟 3: 生成報告
        final_report = summarizer(research_results, api_key, topic)
        
        # 返回原始資料結構，不進行 JSON 序列化
        return {
            "questions": questions,
            "research_results": research_results,
            "final_report": final_report,
            "raw_build": raw_build,
            "topic": topic
        }
    
    return type('SimpleAgent', (), {'invoke': lambda self, inputs: agent_run(inputs['topic'])})()


# ----------------------
# 初始化 session state
# ----------------------
if 'research_completed' not in st.session_state:
    st.session_state.research_completed = False
if 'research_data' not in st.session_state:
    st.session_state.research_data = {}
if 'current_topic' not in st.session_state:
    st.session_state.current_topic = ""

# ----------------------
# Main UI
# ----------------------
if st.button("Start Research 🚀"):
    # 清空之前的結果
    st.session_state.research_completed = False
    st.session_state.research_data = {}
    st.session_state.current_topic = topic
    
    # 使用 LangChain Agent
    st.subheader("🤖 LangChain Agent 執行")
    
    # 創建進度條
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔄 正在初始化 Agent...")
        progress_bar.progress(10)
        
        agent = create_research_agent()
        
        status_text.text("🔍 正在執行研究流程...")
        progress_bar.progress(30)
        
        result = agent.invoke({"topic": topic})
        
        status_text.text("✅ Agent 執行完成！")
        progress_bar.progress(100)
        
    except Exception as e:
        status_text.text(f"❌ Agent 執行失敗: {str(e)}")
        progress_bar.progress(0)
        st.error(f"Agent 執行錯誤: {str(e)}")
        st.stop()
    
    # 解析結果（修復版）
    result_data = result  # 直接使用結果，不需要 JSON 解析
    
    questions = result_data.get("questions", [])
    raw_build = result_data.get("raw_build", "")
    research_results = result_data.get("research_results", [])
    final_report = result_data.get("final_report", "")
    
    st.subheader("🏗️ Topic Refiner")
    st.markdown("### 📋 細化的研究問題")
    
    # 簡單清楚的格式
    if questions:
        for i, q in enumerate(questions, 1):
            question_id = q.get('id', f'Q{i}')
            question_text = q.get('question', '')
            clarifier_text = q.get('clarifier', '')
            
            st.write(f"**📌 {question_id}:** {question_text}")
            if clarifier_text:
                st.write(f"💡 {clarifier_text}")
            st.write("")  # 空行分隔
    
    # 可展開的原始輸出
    with st.expander("🔍 查看原始輸出 (Raw Output)", expanded=False):
        st.code(raw_build[:3000], language="json")

    st.subheader("📏 Researcher")
    st.write("**Research results:**")
    
    # 顯示詳細的研究資料
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
            
            # 顯示真實的 arXiv 論文
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
    
    # 完整研究結果（預設收起來）
    with st.expander("📊 完整研究結果 (JSON)", expanded=False):
        st.write(research_results)

    st.subheader("✍️ Summarizer")
    st.markdown(final_report)
    
    # 儲存結果
    st.session_state.research_data = {
        'questions': questions,
        'raw_build': raw_build,
        'research_results': research_results,
        'final_report': final_report,
        'topic': topic,
        'agent_output': result
    }
    
    # 下載按鈕
    st.download_button("Download report (.txt)", data=st.session_state.research_data['final_report'], file_name="research_report.txt")
    st.download_button("Download JSON export", data=json.dumps(st.session_state.research_data, ensure_ascii=False, indent=2), file_name="research_export.json")
    
    st.session_state.research_completed = True

# 如果有研究結果且不是剛完成的，顯示結果
elif st.session_state.research_completed and st.session_state.research_data:
    data = st.session_state.research_data
    
    st.subheader("🏗️ Topic Refiner")
    
    # 簡單清楚的格式
    if data['questions']:
        for i, q in enumerate(data['questions'], 1):
            question_id = q.get('id', f'Q{i}')
            question_text = q.get('question', '')
            clarifier_text = q.get('clarifier', '')
            
            st.write(f"**📌 {question_id}:** {question_text}")
            if clarifier_text:
                st.write(f"💡 {clarifier_text}")
            st.write("")  # 空行分隔
    
    # 可展開的原始輸出
    with st.expander("🔍 查看原始輸出 (Raw Output)", expanded=False):
        st.code(data['raw_build'][:3000], language="json")

    st.subheader("📏 Researcher")
    with st.expander("📊 完整研究結果 (JSON)", expanded=False):
        st.write(data['research_results'])

    st.subheader("✍️ Summarizer")
    st.markdown(data['final_report'])

    # 下載按鈕
    st.download_button("Download report (.txt)", data=data['final_report'], file_name="research_report.txt")
    st.download_button("Download JSON export", data=json.dumps(data, ensure_ascii=False, indent=2), file_name="research_export.json")
