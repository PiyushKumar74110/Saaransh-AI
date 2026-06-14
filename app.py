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

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# CSS


st.markdown("""
<style>

.block-container{
    padding-top:1rem;
    max-width:100%;
}

.main-header{
    text-align:center;
    border:2px solid #ddd;
    border-radius:10px;
    padding:15px;
    margin-bottom:20px;
    font-size:40px;
    font-weight:bold;
}

.summary-box{
    border:2px solid #ddd;
    border-radius:10px;
    height:600px;
    overflow-y:auto;
    padding:15px;
}

.chat-box{
    border:2px solid #ddd;
    border-radius:10px;
    height:600px;
    overflow-y:auto;
    padding:15px;
}

.user-msg{
    background:#DCF8C6;
    padding:10px;
    border-radius:10px;
    margin:8px 0;
    text-align:right;
}

.bot-msg{
    background:#F1F1F1;
    padding:10px;
    border-radius:10px;
    margin:8px 0;
}

.divider{
    border-left:2px solid #ddd;
    height:720px;
    margin:auto;
}

</style>
""", unsafe_allow_html=True)


# HEADER


st.markdown(
    '<div class="main-header">🎬 AI Video Assistant</div>',
    unsafe_allow_html=True,
)


# MAIN LAYOUT


left, divider, right = st.columns([1, 0.03, 1])


# LEFT PANEL


with left:

    source = st.text_input(
        "Video URL",
        placeholder="https://youtube.com/watch?v=..."
    )

    language = st.selectbox(
        "Language",
        ["english", "hinglish"]
    )

    analyze_btn = st.button(
        "Summarize",
        use_container_width=True
    )

    if analyze_btn:

        if not source.strip():
            st.error("Please enter a video URL.")
        else:

            try:

                with st.spinner("Processing video..."):

                    # 1. Download / Extract Audio
                    chunks = process_input(source)

                    # 2. Transcribe
                    transcript = transcribe_all(
                        chunks,
                        language
                    )

                    # 3. Summaries
                    title = generate_title(transcript)
                    summary = summarize(transcript)

                    # 4. Metadata
                    action_items = extract_action_items(
                        transcript
                    )

                    decisions = extract_key_decisions(
                        transcript
                    )

                    questions = extract_questions(
                        transcript
                    )

                    # 5. RAG
                    rag_chain = build_rag_chain(
                        transcript
                    )

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

                st.success("Analysis completed.")

            except Exception as e:
                st.error(f"Error: {e}")

    st.subheader("Summary")

    summary_text = (
        st.session_state.summary
        if st.session_state.summary
        else "Summary will appear here after analysis."
    )

    st.markdown(
        f"""
        <div class="summary-box">
        {summary_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


# DIVIDER


with divider:
    st.markdown(
        '<div class="divider"></div>',
        unsafe_allow_html=True,
    )


# RIGHT PANEL

with right:

    st.subheader("Chat with Video")

    chat_html = ""

    for msg in st.session_state.chat_history:

        if msg["role"] == "user":
            chat_html += (
                f'<div class="user-msg">'
                f'{msg["content"]}'
                f'</div>'
            )
        else:
            chat_html += (
                f'<div class="bot-msg">'
                f'{msg["content"]}'
                f'</div>'
            )

    st.markdown(
        f"""
        <div class="chat-box">
        {chat_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    input_col, send_col = st.columns([5, 1])

    with input_col:
        question = st.text_input(
            "",
            placeholder="Ask me...",
            label_visibility="collapsed"
        )

    with send_col:
        send_btn = st.button(
            "Send",
            use_container_width=True
        )

    if send_btn:

        if not st.session_state.result:
            st.warning(
                "Please analyze a video first."
            )

        elif question.strip():

            try:

                rag_chain = (
                    st.session_state.result["rag_chain"]
                )

                with st.spinner("Thinking..."):

                    answer = rag_chain.invoke({
                        "question": question
                    })

                st.session_state.chat_history.append(
                    {
                        "role": "user",
                        "content": question,
                    }
                )

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": str(answer),
                    }
                )

                st.rerun()

            except Exception as e:
                st.error(f"Chat Error: {e}")


# EMPTY STATE


if not st.session_state.result:

    st.markdown(
        """
        <div style="text-align:center;
                    margin-top:20px;
                    color:gray;">
        Paste a YouTube URL and click Summarize
        </div>
        """,
        unsafe_allow_html=True,
    )