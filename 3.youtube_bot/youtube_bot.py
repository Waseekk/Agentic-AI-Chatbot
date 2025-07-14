import os
import re
import tempfile
from typing import Optional

import streamlit as st
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from dotenv import load_dotenv

from openai import OpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Attempt to import Groq client
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Load environment variables
load_dotenv()
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV     = os.getenv("PINECONE_ENVIRONMENT")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
TRAVILY_API_KEY  = os.getenv("TRAVILY_API_KEY")

# Initialize OpenAI client and embeddings
client     = OpenAI(api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings()

# Initialize Groq client if available
if GROQ_AVAILABLE and GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

# Helper: extract 11‑char YouTube ID
def get_video_id(url: str) -> str:
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    if not match:
        st.error("❌ Invalid YouTube URL. Ensure it's a full link or shareable URL.")
        st.stop()
    return match.group(1)

# Helper: get video title using yt_dlp (new to get metadata)
def get_video_title(url: str) -> Optional[str]:
    ydl_opts = {"quiet": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return info.get("title", None)
        except Exception:
            return None

# Cache caption fetch
@st.cache_data(show_spinner=True)
def fetch_captions(video_id: str, language: str = "en") -> Optional[str]:
    try:
        segments = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        return "\n".join(seg["text"] for seg in segments)
    except NoTranscriptFound:
        return None

# Cache audio download
@st.cache_data(show_spinner=True)
def download_audio(url: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": tmp.name,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
        ],
        "quiet": True
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    return tmp.name

# Cache Whisper transcription
@st.cache_data(show_spinner=True)
def transcribe_with_whisper(mp3_path: str, model: str = "whisper-1") -> str:
    with open(mp3_path, "rb") as f:
        resp = client.audio.transcriptions.create(model=model, file=f)
    return resp.text

# Get transcript: try captions, else Whisper
@st.cache_data(show_spinner=True)
def get_transcript(url: str, language: str, model: str) -> str:
    vid      = get_video_id(url)
    captions = fetch_captions(vid, language)
    if captions:
        return captions
    st.info("⚠️ No captions found. Transcribing via Whisper...")
    mp3 = download_audio(url)
    return transcribe_with_whisper(mp3, model)

# ─── Streamlit App Setup ────────────────────────────────────────────────────────
st.set_page_config(page_title="YouTube QA Bot", layout="wide")
st.title("🛠️ YouTube QA Bot")

# ─── Sidebar Settings ──────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

use_pinecone  = st.sidebar.checkbox("Use Pinecone for vectors", value=False)
language      = st.sidebar.selectbox("Transcript Language", ["en","es","fr","de"], index=0)
whisper_model = st.sidebar.selectbox("Whisper Model", ["whisper-1"], index=0)

# LLM Provider + Model choice
llm_provider = st.sidebar.selectbox("LLM Provider", ["OpenAI","GROQ"], index=0)
if llm_provider == "GROQ":
    groq_model = st.sidebar.selectbox(
        "GROQ Model",
        ["llama3-8b-8192", "llama3-70b-8192"],
        index=0
    )
else:
    openai_model = st.sidebar.selectbox(
        "OpenAI Model",
        ["gpt-3.5-turbo","gpt-4o"],
        index=0
    )

if TRAVILY_API_KEY:
    st.sidebar.success("🔗 Travily configured")

# ─── Session State Defaults ─────────────────────────────────────────────────────
if "vectordb"     not in st.session_state: st.session_state.vectordb     = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "urls"         not in st.session_state: st.session_state.urls         = []
if "video_metadata" not in st.session_state: st.session_state.video_metadata = {}  # Store metadata keyed by URL

# ─── 1. Indexing Section ───────────────────────────────────────────────────────
st.subheader("📥 1. Index a YouTube Video")
url_input = st.text_input("Video URL to index:", key="url_input")
if st.button("Index Video"):
    if not url_input:
        st.warning("❗ Please enter a valid YouTube URL.")
    else:
        with st.spinner("Indexing..."):
            # Fetch video title metadata
            title = get_video_title(url_input) or "Unknown Title"
            # Store metadata
            st.session_state.video_metadata[url_input] = {"title": title, "url": url_input}

            transcript = get_transcript(url_input, language, whisper_model)
            splitter   = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

            chunks     = splitter.split_text(transcript)
            # Add richer metadata for each chunk with video title and url
            docs       = [
                Document(
                    page_content=chunk,
                    metadata={
                        "source_url": url_input,
                        "source_title": title
                    }
                )
                for chunk in chunks
            ]
            if st.session_state.vectordb is None:
                st.session_state.vectordb = FAISS.from_documents(docs, embeddings)
            else:
                st.session_state.vectordb.add_documents(docs)
            st.session_state.urls.append(url_input)
            st.success(f"✅ Indexed {len(chunks)} chunks from video: {title}")

if st.session_state.urls:
    st.markdown("**Indexed Videos:**")
    for vid in st.session_state.urls:
        metadata = st.session_state.video_metadata.get(vid, {})
        title = metadata.get("title", "Unknown Title")
        st.markdown(f"- [{title}]({vid})")

# ─── 2. Chat Section ────────────────────────────────────────────────────────────
st.subheader("💬 2. Chat with Indexed Content")

# Display chat history with chat_message bubbles
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Real-time chat input
if question := st.chat_input("Ask a question about the indexed videos..."):
    if not st.session_state.vectordb or not st.session_state.urls:
        st.error("❌ No videos indexed. Please index a video first.")
    else:
        # log & display user
        st.session_state.chat_history.append({"role":"user","content":question})
        # Display user's message immediately in chat bubble
        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("Retrieving answer..."):
            try:
                if llm_provider == "GROQ":
                    # 1. Retrieve top documents for this question
                    retriever = st.session_state.vectordb.as_retriever()
                    docs      = retriever.get_relevant_documents(question)

                    # 2. Build a single context string from them
                    context = "\n\n".join(d.page_content for d in docs)

                    # 3. Construct messages: system prompt, context, then user question
                    messages = [
                        {"role": "system",
                        "content": "You are a helpful assistant; use the following transcript snippets to answer."},
                        {"role": "system", "content": context},
                        {"role": "user",   "content": question}
                    ]

                    # 4. Call Groq with that enriched prompt
                    response = groq_client.chat.completions.create(
                        model=groq_model,
                        messages=messages
                    )
                    answer = response.choices[0].message.content

                else:
                    qa = RetrievalQA.from_chain_type(
                        llm=ChatOpenAI(model=openai_model, temperature=0),
                        retriever=st.session_state.vectordb.as_retriever(),
                        return_source_documents=True
                    )
                    result = qa.invoke({"query":question})
                    answer = result["result"]
                    source_docs = result.get("source_documents", [])

                # Compose source attribution text with links and titles
                if llm_provider == "OpenAI" and source_docs:
                    # Extract unique sources
                    sources = {}
                    for doc in source_docs:
                        meta = doc.metadata
                        url = meta.get("source_url", "")
                        title = meta.get("source_title", "Video")
                        sources[url] = title

                    source_md = "\n\n---\n**Sources:**\n"
                    for url, title in sources.items():
                        source_md += f"- [{title}]({url})\n"

                    answer_with_sources = answer + source_md
                else:
                    # For Groq or no sources, just show answer
                    answer_with_sources = answer

                # Append and display bot answer with sources
                st.session_state.chat_history.append({"role":"assistant","content":answer_with_sources})
                with st.chat_message("assistant"):
                    st.markdown(answer_with_sources)
            except Exception as e:
                error_msg = f"Error generating response: {e}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role":"assistant","content":error_msg})

# Clear chat history
if st.session_state.chat_history:
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# ─── Integration Status ────────────────────────────────────────────────────────
if llm_provider == "GROQ":
    if GROQ_AVAILABLE and GROQ_API_KEY:
        st.sidebar.info(f"🟢 GROQ Ready — Model: {groq_model}")
    elif GROQ_API_KEY:
        st.sidebar.error("❌ GROQ API key found but client import failed.")
    else:
        st.sidebar.error("❌ GROQ is not configured.")
else:
    st.sidebar.info(f"🟢 OpenAI Ready — Model: {openai_model}")
