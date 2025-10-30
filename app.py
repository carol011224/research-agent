import streamlit as st
from openai import OpenAI
import json
import os
import requests
import feedparser
import random
from datetime import datetime, timedelta
from time import sleep
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

st.set_page_config(page_title="A Research Agent Demo", layout="wide")
st.title("A Research Agent Demo")
st.write("Three sub-agents: Topic Refiner → Researcher → Summarizer")

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
    topic = st.text_input("Research topic", value="最新多語言 LLM fine-tuning 方法")
    # temp = st.slider("Temperature", 0.0, 1.0, 0.2) # 可讓使用者選擇 temperature 參數
    st.markdown("---")

# ----------------------
# LLM wrapper
# ----------------------
def llm_chat(prompt, api_key, model="gpt-4o-mini", temperature=0.2, max_tokens=800, max_retries=3):
    client = OpenAI(api_key=api_key)
    
    for attempt in range(max_retries):
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
            error_str = str(e)
            
            # 檢查是否是速率限制錯誤
            if "rate_limit_exceeded" in error_str or "Rate limit" in error_str:
                if attempt < max_retries - 1:
                    wait_time = 25 + (attempt * 10)  # 遞增等待時間：25s, 35s, 45s
                    st.warning(f"API 速率限制，等待 {wait_time} 秒後重試... (嘗試 {attempt + 1}/{max_retries})")
                    sleep(wait_time)
                    continue
                else:
                    st.error("API 速率限制，已達到最大重試次數。請稍後再試。")
                    return f"[LLM call failed] Rate limit exceeded after {max_retries} attempts"
            
            # 其他錯誤
            elif "timeout" in error_str.lower():
                if attempt < max_retries - 1:
                    st.warning(f"API 請求超時，等待 10 秒後重試... (嘗試 {attempt + 1}/{max_retries})")
                    sleep(10)
                    continue
            
            # 其他錯誤直接返回
            return f"[LLM call failed] {e}"
    
    return f"[LLM call failed] Unknown error after {max_retries} attempts"

# ----------------------
# Data Collection Functions
# ----------------------
def search_arxiv(query, max_results=3, start_offset=0):
    """搜尋 arXiv 最新論文"""
    
    # 簡化搜尋查詢，移除特殊字符
    clean_query = query.replace('?', '').replace('!', '').replace('。', '')
    clean_query = clean_query.strip()
    
    try:
        # 使用 HTTPS
        url = "https://export.arxiv.org/api/query"
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
        
        st.info(f"正在搜尋 arXiv: {clean_query[:50]}...")
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
        
        if papers:
            st.success(f"成功找到 {len(papers)} 篇論文")
            return papers
        else:
            st.warning("沒有找到相關論文")
            return []
            
    except Exception as e:
        st.warning(f"arXiv 搜尋失敗: {e}")
        return []


# ----------------------
# Agent functions
# ----------------------
def topic_refiner(topic, api_key):
    prompt = f"""你是一名資深研究助理。針對主題「{topic}」，生成 3~5 個具體、可研究的子問題。

            請直接輸出問題清單，每行一個問題，不要包含任何格式符號或 JSON 結構。

            範例輸出格式：
            機器學習中的注意力機制如何工作？
            深度學習在自然語言處理中的應用
            Transformer 架構的優缺點分析

            請生成關於「{topic}」的具體研究問題："""
    
    raw = llm_chat(prompt, api_key)
    
    # 清理輸出，移除格式符號
    lines = raw.split('\n')
    questions = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        # 移除常見的格式符號
        line = line.replace('```json', '').replace('```', '').replace('{', '').replace('}', '').replace('[', '').replace(']', '')
        line = line.replace('"question":', '').replace('"', '').replace(',', '')
        
        # 只保留看起來像問題的行
        if line and len(line) > 10 and ('?' in line or '如何' in line or '什麼' in line or '為什麼' in line or '哪些' in line or '分析' in line or '研究' in line):
            questions.append({
                "id": f"Q{i+1}",
                "question": line,
                "clarifier": f"關於 {topic} 的研究問題"
            })
    
    # 如果沒有找到合適的問題，使用備用方案
    if not questions:
        questions = [
            {"id": "Q1", "question": f"{topic}的基本原理是什麼？", "clarifier": f"關於 {topic} 的基本概念"},
            {"id": "Q2", "question": f"{topic}的最新發展趨勢如何？", "clarifier": f"關於 {topic} 的發展現況"},
            {"id": "Q3", "question": f"{topic}的應用領域有哪些？", "clarifier": f"關於 {topic} 的實際應用"}
        ]
    
    return questions[:5], raw


def researcher(questions, api_key):
    results = []
    for i, q in enumerate(questions):
        question = q['question']
        
        # 清理問題文本，移除特殊字符
        clean_question = question.replace('```', '').replace('json', '').replace('{', '').replace('}', '')
        clean_question = clean_question.replace('[', '').replace(']', '').replace('"', '').replace("'", '')
        clean_question = clean_question.replace('?', '').replace('!', '').replace('。', '').replace('？', '')
        clean_question = clean_question.strip()
        
        # 使用完整的問題作為搜尋查詢，但限制長度
        search_query = clean_question
        if len(search_query) > 100:
            search_query = search_query[:100]
        
        if not clean_question or len(clean_question) < 5:
            st.warning(f"跳過無效問題: {question}")
            continue
        
        # 1. 搜尋 arXiv 最新論文
        # 為每個問題添加一些變化，避免重複結果
        start_offset = random.randint(0, 5)  # 隨機偏移，獲取不同的論文
        arxiv_papers = search_arxiv(search_query, max_results=3, start_offset=start_offset)
        if not arxiv_papers:
            st.info(f"arXiv 中沒有找到「{clean_question}」的相關論文")
        
        # 2. 整理資料為 JSON 格式
        research_data = {
            'arxiv_papers': arxiv_papers,
            'search_timestamp': datetime.now().isoformat()
        }
        
        # 3. 使用 LLM 分析真實資料
        # 在每個問題之間添加延遲，避免速率限制
        if i > 0:
            st.info("等待 10 秒以避免 API 速率限制...")
            sleep(10)
        
        prompt = f"""你是一名研究員。針對子問題「{clean_question}」，我已經收集了真實的研究資料：

                arXiv 論文資料：
                {json.dumps(arxiv_papers, ensure_ascii=False, indent=2)}

                請根據這些真實的 arXiv 論文資料進行分析：
                1) 整理 2-3 條具體的研究發現（基於論文摘要和標題）
                2) 評估資料來源的可信度（arXiv 學術論文）
                3) 提供簡短的可信度評估 (low/medium/high)
                4) 如果資料不足，請註明

                輸出 JSON 格式：
                {{\n                    "findings": ["基於論文的發現1", "基於論文的發現2"],\n                    "sources": ["arXiv論文來源"],\n                    "confidence": "medium",\n                    "data_quality": "good/limited",\n                    "notes": "額外說明或資料限制"\n                }}"""
        
        raw = llm_chat(prompt, api_key)
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {
                "findings": [f"arXiv論文: {paper['title'][:50]}..." for paper in arxiv_papers[:2]], 
                "sources": ["arXiv"],
                "confidence": "medium", 
                "data_quality": "good" if arxiv_papers else "limited",
                "notes": "JSON parsing failed, showing raw data",
                "__raw": raw
            }
        
        # 添加原始資料供參考
        parsed['raw_research_data'] = research_data
        
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
# 初始化 session state
# ----------------------
if 'research_completed' not in st.session_state:
    st.session_state.research_completed = False
if 'research_data' not in st.session_state:
    st.session_state.research_data = {}

# ----------------------
# Main UI
# ----------------------
if st.button("Start Research 🚀"):
    # 清空之前的結果
    st.session_state.research_completed = False
    st.session_state.research_data = {}
    
    st.subheader("🏗️ Topic Refiner")
    with st.spinner("Refining topic..."):
        questions, raw_build = topic_refiner(topic, api_key)
    st.write("**Refined questions:**")
    st.write(questions)
    st.write("**Raw output:**")
    st.code(raw_build[:3000])

    st.subheader("📏 Researcher")
    with st.spinner("Searching arXiv for latest papers..."):
        research_results = researcher(questions, api_key)
    
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
            
            # 顯示原始資料
            raw_data = analysis.get('raw_research_data', {})
            if raw_data.get('arxiv_papers'):
                st.write("**arXiv 論文:**")
                for paper in raw_data['arxiv_papers']:
                    st.write(f"📄 [{paper['title']}]({paper['url']})")
                    st.write(f"   {paper['summary'][:200]}...")
                    st.write(f"   👥 作者: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
                    st.write(f"   📅 發布: {paper['published'][:10]}")
                    st.write("---")
    
    st.write("**完整研究結果:**")
    st.write(research_results)

    st.subheader("✍️ Summarizer")
    with st.spinner("Generating final report..."):
        # 在生成報告前添加延遲
        sleep(10)
        final_report = summarizer(research_results, api_key, topic)
    st.markdown(final_report)

    # 下載按鈕
    st.download_button("Download report (.txt)", data=final_report, file_name="research_report.txt")
    st.download_button("Download JSON export", data=json.dumps({"topic": topic, "questions": questions, "research": research_results, "report": final_report}, ensure_ascii=False, indent=2), file_name="research_export.json")
    
    # 儲存結果到 session state
    st.session_state.research_data = {
        'questions': questions,
        'raw_build': raw_build,
        'research_results': research_results,
        'final_report': final_report,
        'topic': topic
    }
    st.session_state.research_completed = True

# 如果有研究結果且不是剛完成的，顯示結果
elif st.session_state.research_completed and st.session_state.research_data:
    data = st.session_state.research_data
    
    st.subheader("🏗️ Topic Refiner")
    st.write("**Refined questions:**")
    st.write(data['questions'])
    st.write("**Raw output:**")
    st.code(data['raw_build'][:3000])

    st.subheader("📏 Researcher")
    
    # 顯示詳細的研究資料
    for i, result in enumerate(data['research_results']):
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
            
            # 顯示原始資料
            raw_data = analysis.get('raw_research_data', {})
            if raw_data.get('arxiv_papers'):
                st.write("**arXiv 論文:**")
                for paper in raw_data['arxiv_papers']:
                    st.write(f"📄 [{paper['title']}]({paper['url']})")
                    st.write(f"   {paper['summary'][:200]}...")
                    st.write(f"   👥 作者: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
                    st.write(f"   📅 發布: {paper['published'][:10]}")
                    st.write("---")
    
    st.write("**完整研究結果:**")
    st.write(data['research_results'])

    st.subheader("✍️ Summarizer")
    st.markdown(data['final_report'])

    # 下載按鈕
    st.download_button("Download report (.txt)", data=data['final_report'], file_name="research_report.txt")
    st.download_button("Download JSON export", data=json.dumps({"topic": data['topic'], "questions": data['questions'], "research": data['research_results'], "report": data['final_report']}, ensure_ascii=False, indent=2), file_name="research_export.json")
