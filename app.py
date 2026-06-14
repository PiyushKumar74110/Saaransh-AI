import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain

load_dotenv()

# PAGE CONFIG
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
)

# SESSION STATE
defaults = {
    "summary": "",
    "title": "AI Video Assistant",
    "chat_history": [],
    "result": None,
}

for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# SIMPLE CSS
st.markdown("""
<style>
.main-title {
    font-size: 34px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 10px;
}
.subtitle {
    text-align:center;
    color: gray;
    margin-bottom: 20px;
}
.block {
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("<div class='main-title'>🎬 AI Video Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Summarize & chat with any YouTube video</div>", unsafe_allow_html=True)

# SIDEBAR (NEW CLEAN INPUT AREA)
with st.sidebar:
    st.header("⚙️ Controls")

    source = st.text_input("Video URL", placeholder="https://youtube.com/watch?v=...")

    language = st.selectbox("Language", ["english", "hinglish"])

    analyze_btn = st.button("🚀 Analyze Video", use_container_width=True)

# MAIN ACTION
if analyze_btn:
    if not source.strip():
        st.error("Please enter a valid URL")
    else:
        try:
            with st.spinner("Processing video..."):

                chunks = process_input(source)
                transcript = transcribe_all(chunks, language)

                title = generate_title(transcript)
                summary = summarize(transcript)

                action_items = extract_action_items(transcript)
                decisions = extract_key_decisions(transcript)
                questions = extract_questions(transcript)

                rag_chain = build_rag_chain(transcript)

                st.session_state.result = {
                    "title": title,
                    "summary": summary,
                    "transcript": transcript,
                    "action_items": action_items,
                    "decisions": decisions,
                    "questions": questions,
                    "rag_chain": rag_chain,
                }

                st.session_state.summary = summary
                st.session_state.title = title
                st.session_state.chat_history = []

            st.success("Analysis complete 🎉")

        except Exception as e:
            st.error(f"Error: {e}")

# TABS (NEW UI STRUCTURE)
tab1, tab2 = st.tabs(["📄 Summary", "💬 Chat"])

# ---------------- SUMMARY TAB ----------------
with tab1:
    st.subheader("Video Summary")

    if st.session_state.summary:
        st.markdown(st.session_state.summary)
    else:
        st.info("Summary will appear after analysis.")

    if st.session_state.result:
        st.divider()
        st.subheader("Key Insights")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("📌 Action Items")
            st.write(st.session_state.result["action_items"])

        with col2:
            st.write("✅ Decisions")
            st.write(st.session_state.result["decisions"])

        with col3:
            st.write("❓ Questions")
            st.write(st.session_state.result["questions"])

# ---------------- CHAT TAB ----------------
with tab2:
    st.subheader("Chat with Video")

    if not st.session_state.result:
        st.info("Analyze a video first to enable chat.")
    else:
        # Render chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        question = st.chat_input("Ask something about the video...")

        if question:
            rag_chain = st.session_state.result["rag_chain"]

            with st.chat_message("user"):
                st.write(question)

            with st.spinner("Thinking..."):
                answer = rag_chain.invoke({"question": question})

            with st.chat_message("assistant"):
                st.write(answer)

            st.session_state.chat_history.append(
                {"role": "user", "content": question}
            )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": str(answer)}
            )
