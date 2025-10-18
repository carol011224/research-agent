import streamlit as st
from openai import OpenAI
import json
import os
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
    temp = st.slider("Temperature", 0.0, 1.0, 0.2)
    st.markdown("---")

# ----------------------
# LLM wrapper
# ----------------------
def llm_chat(prompt, api_key, model="gpt-4o-mini", temperature=0.2, max_tokens=800):
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

# ----------------------
# Agent functions
# ----------------------
def topic_refiner(topic, api_key):
    prompt = f"你是一名資深研究助理。針對主題「{topic}」，生成 3~5 個具體、可研究的子問題，並為每個子問題附上簡短 clarifier。輸出 JSON 格式。"
    raw = llm_chat(prompt, api_key)
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = [{"id": f"Q{i+1}", "question": line[:200], "clarifier": ""} for i, line in enumerate(raw.splitlines()[:5])]
    return parsed, raw


def researcher(questions, api_key):
    results = []
    for q in questions:
        prompt = f"你是一名研究員。針對子問題 {q['question']}：\n1) 模擬 2~3 條最新研究發現，每條 1 句。\n2) 模擬可能資料來源（如 arXiv, gov report）\n3) 簡短可信度評估 (low/medium/high)。輸出 JSON。"
        raw = llm_chat(prompt, api_key)
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"findings": [s.strip() for s in raw.splitlines()[:3]], "sources_hint":"unspecified", "confidence":"unknown", "__raw": raw}
        results.append({"id": q.get("id"), "question": q.get("question"), "analysis": parsed})
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
    with st.spinner("Simulating research findings..."):
        research_results = researcher(questions, api_key)
    st.write("**Research results:**")
    st.write(research_results)

    st.subheader("✍️ Summarizer")
    with st.spinner("Generating final report..."):
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
    st.write("**Research results:**")
    st.write(data['research_results'])

    st.subheader("✍️ Summarizer")
    st.markdown(data['final_report'])

    # 下載按鈕
    st.download_button("Download report (.txt)", data=data['final_report'], file_name="research_report.txt")
    st.download_button("Download JSON export", data=json.dumps({"topic": data['topic'], "questions": data['questions'], "research": data['research_results'], "report": data['final_report']}, ensure_ascii=False, indent=2), file_name="research_export.json")
