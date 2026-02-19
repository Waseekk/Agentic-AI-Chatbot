"""
YouTube QA Bot - Portfolio-Ready Streamlit App
Index YouTube videos and chat with their content using Groq (free) or OpenAI.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Section 0: Imports & Config Constants
# ═══════════════════════════════════════════════════════════════════════════════
import os
import re
import time
import shutil
import logging
import tempfile
from datetime import date, datetime
from typing import Optional, List, Dict

import streamlit as st
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ── Logging (visible in Streamlit Cloud "Manage app" > Logs) ─────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("youtube_qa_bot")

# ── Tunable constants ────────────────────────────────────────────────────────
MAX_QUERIES_PER_MINUTE = 10
MAX_VIDEOS_PER_SESSION = 10
MAX_PLAYLIST_VIDEOS = 15
CHUNK_SIZE = 1000
DEFAULT_RETRIEVER_K = 4
MAX_QUESTION_LENGTH = 1000

MODEL_COSTS = {  # per 1K tokens
    "llama-3.1-8b-instant": 0.00005,
    "llama-3.3-70b-versatile": 0.00059,
    "openai/gpt-oss-20b": 0.000075,
    "gpt-4o-mini": 0.00015,
    "gpt-3.5-turbo": 0.0005,
    "gpt-4o": 0.005,
    "whisper-1": 0.006,
}

SUPPORTED_LANGUAGES = ["en", "es", "fr", "de", "hi", "pt", "ja", "ko", "ar", "zh"]

VIDEO_URL_PATTERN = re.compile(
    r'(?:v=|v/|embed/|youtu\.be/|/v/|watch\?v=|watch\?.*&v=|shorts/|live/)([A-Za-z0-9_-]{11})'
)
PLAYLIST_URL_PATTERN = re.compile(r'[?&]list=([a-zA-Z0-9_-]+)')

INJECTION_PATTERNS = re.compile(
    r'(ignore\s+(previous|above|all)\s+instructions|you\s+are\s+now|system\s*prompt|'
    r'disregard\s+(all|your)|forget\s+(everything|your)|override\s+instructions)',
    re.IGNORECASE
)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: API Key Validation & Lazy Clients
# ═══════════════════════════════════════════════════════════════════════════════
def _get_api_key(name: str) -> Optional[str]:
    """User-entered key (session) -> st.secrets (Cloud) -> os.getenv (local)."""
    session_key = f"user_{name.lower()}"
    user_val = st.session_state.get(session_key)
    if user_val:
        return str(user_val)
    try:
        val = st.secrets.get(name)
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(name)


def get_openai_client():
    if "openai_client" not in st.session_state:
        key = _get_api_key("OPENAI_API_KEY")
        if key and OPENAI_AVAILABLE:
            st.session_state.openai_client = OpenAI(api_key=key)
        else:
            st.session_state.openai_client = None
    return st.session_state.openai_client


def get_groq_client():
    if "groq_client" not in st.session_state:
        key = _get_api_key("GROQ_API_KEY")
        if key and GROQ_AVAILABLE:
            st.session_state.groq_client = Groq(api_key=key)
        else:
            st.session_state.groq_client = None
    return st.session_state.groq_client


def get_embeddings():
    if "embeddings" not in st.session_state:
        key = _get_api_key("OPENAI_API_KEY")
        if key:
            st.session_state.embeddings = OpenAIEmbeddings(api_key=key)
        else:
            st.session_state.embeddings = None
    return st.session_state.embeddings


def validate_api_keys() -> Dict[str, bool]:
    return {
        "openai": bool(_get_api_key("OPENAI_API_KEY")) and OPENAI_AVAILABLE,
        "groq": bool(_get_api_key("GROQ_API_KEY")) and GROQ_AVAILABLE,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Usage Logging & Rate Limiter
# ═══════════════════════════════════════════════════════════════════════════════
def init_usage_state():
    today = date.today().isoformat()
    if "usage" not in st.session_state or st.session_state.usage.get("date") != today:
        st.session_state.usage = {
            "date": today,
            "session_tokens": 0,
            "query_count": 0,
            "videos_indexed": 0,
            "query_timestamps": [],
        }


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def check_rate_limit() -> tuple:
    now = time.time()
    usage = st.session_state.usage
    usage["query_timestamps"] = [t for t in usage["query_timestamps"] if now - t < 60]
    if len(usage["query_timestamps"]) >= MAX_QUERIES_PER_MINUTE:
        oldest = usage["query_timestamps"][0]
        wait = int(60 - (now - oldest)) + 1
        return False, f"Please wait {wait} seconds before asking another question."
    return True, ""


def log_usage(event: str, **kwargs):
    """Log usage events to Streamlit Cloud logs."""
    details = " | ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info(f"{event} | {details}")


def record_usage(input_tokens: int, output_tokens: int, model: str):
    total = input_tokens + output_tokens
    usage = st.session_state.usage
    usage["session_tokens"] += total
    usage["query_count"] += 1
    usage["query_timestamps"].append(time.time())
    log_usage("QUERY", model=model, input_tokens=input_tokens, output_tokens=output_tokens, total=total)


def record_whisper_usage(duration_minutes: float):
    log_usage("WHISPER", duration_min=f"{duration_minutes:.1f}", est_cost=f"${duration_minutes * 0.006:.4f}")


def check_video_limit() -> tuple:
    count = st.session_state.usage.get("videos_indexed", 0)
    if count >= MAX_VIDEOS_PER_SESSION:
        return False, f"You can index up to {MAX_VIDEOS_PER_SESSION} videos per session. Refresh the page to start a new session."
    return True, ""


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: YouTube Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def get_video_id(url: str) -> str:
    match = VIDEO_URL_PATTERN.search(url)
    if match:
        return match.group(1)
    raise ValueError("Invalid YouTube URL. Supported formats: watch, shorts, live, embed, youtu.be")


def get_video_metadata(url: str) -> Dict:
    ydl_opts = {"quiet": True, "skip_download": True, "no_warnings": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "title": info.get("title", "Unknown Title"),
                "duration": info.get("duration", 0),
                "thumbnail": info.get("thumbnail", ""),
                "uploader": info.get("uploader", "Unknown"),
                "url": url,
            }
    except Exception:
        return {"title": "Unknown Title", "duration": 0, "thumbnail": "", "uploader": "Unknown", "url": url}


def is_playlist_url(url: str) -> bool:
    return bool(PLAYLIST_URL_PATTERN.search(url))


def get_playlist_videos(url: str, max_videos: int = MAX_PLAYLIST_VIDEOS) -> List[Dict]:
    ydl_opts = {
        "quiet": True, "skip_download": True, "extract_flat": True,
        "playlist_items": f"1:{max_videos}", "no_warnings": True,
    }
    videos = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            for entry in info.get("entries", []):
                if entry and "id" in entry:
                    videos.append({
                        "url": f"https://www.youtube.com/watch?v={entry['id']}",
                        "title": entry.get("title", f"Video {entry['id']}"),
                        "id": entry["id"],
                    })
    except Exception:
        pass
    return videos


@st.cache_data(show_spinner=False)
def fetch_captions_segments(video_id: str, language: str = "en") -> Optional[List[Dict]]:
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=[language])
        return transcript.to_raw_data()
    except Exception:
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript = ytt_api.fetch(video_id)
            return transcript.to_raw_data()
        except Exception:
            return None


@st.cache_data(show_spinner=False)
def download_audio(url: str) -> str:
    import glob as globmod
    tmp_dir = tempfile.mkdtemp()
    video_id = get_video_id(url)
    output_template = os.path.join(tmp_dir, f"{video_id}.%(ext)s")
    opts = {
        "format": "bestaudio/best", "outtmpl": output_template,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "128"}],
        "quiet": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        mp3_files = globmod.glob(os.path.join(tmp_dir, "*.mp3"))
        if mp3_files:
            return mp3_files[0]
        raise FileNotFoundError("No MP3 file found after download")
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise e


@st.cache_data(show_spinner=False)
def transcribe_with_whisper(mp3_path: str) -> str:
    client = get_openai_client()
    if not client:
        raise RuntimeError("OpenAI API key required for Whisper transcription.")
    with open(mp3_path, "rb") as f:
        resp = client.audio.transcriptions.create(model="whisper-1", file=f)
    file_size = os.path.getsize(mp3_path)
    est_minutes = (file_size / (128 * 1024 / 8)) / 60
    record_whisper_usage(est_minutes)
    tmp_dir = os.path.dirname(mp3_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return resp.text


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def format_timestamp_url(url: str, seconds: float) -> str:
    video_id = get_video_id(url)
    return f"https://youtu.be/{video_id}?t={int(seconds)}"


def chunk_transcript_segments(segments: List[Dict], chunk_size: int = CHUNK_SIZE) -> List[Dict]:
    chunks = []
    current_text = ""
    current_start = None
    current_end = None
    for seg in segments:
        seg_text = seg["text"].strip()
        seg_start = seg["start"]
        seg_end = seg_start + seg.get("duration", 0)
        if current_start is None:
            current_start = seg_start
        if len(current_text) + len(seg_text) + 1 > chunk_size and current_text:
            time_ref = f"[{format_time(current_start)} - {format_time(current_end)}]"
            chunks.append({"text": f"{current_text.strip()} {time_ref}", "start": current_start, "end": current_end})
            current_text = seg_text + " "
            current_start = seg_start
        else:
            current_text += seg_text + " "
        current_end = seg_end
    if current_text:
        time_ref = f"[{format_time(current_start)} - {format_time(current_end)}]"
        chunks.append({"text": f"{current_text.strip()} {time_ref}", "start": current_start, "end": current_end})
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Indexing Engine
# ═══════════════════════════════════════════════════════════════════════════════
def get_transcript_segments(url: str, language: str, allow_whisper: bool) -> Optional[List[Dict]]:
    video_id = get_video_id(url)
    segments = fetch_captions_segments(video_id, language)
    if segments:
        return segments
    if not allow_whisper:
        return None
    try:
        st.info("No captions found. Transcribing with Whisper...")
        mp3_path = download_audio(url)
        transcript_text = transcribe_with_whisper(mp3_path)
    except Exception as e:
        st.error(f"Whisper transcription failed: {e}")
        return None
    words = transcript_text.split()
    segments = []
    for i in range(0, len(words), 100):
        chunk_words = words[i:i + 100]
        segments.append({"text": " ".join(chunk_words), "start": i * 0.5, "duration": len(chunk_words) * 0.5})
    return segments


def process_single_video(url: str, language: str, allow_whisper: bool, title: str = "") -> bool:
    ok, msg = check_video_limit()
    if not ok:
        st.warning(msg)
        return False
    if url in st.session_state.get("urls", []):
        st.info(f"This video is already indexed.")
        return True
    if not title:
        meta = get_video_metadata(url)
        title = meta.get("title", "Unknown Title")
    else:
        meta = get_video_metadata(url)
        meta["title"] = title
    segments = get_transcript_segments(url, language, allow_whisper)
    if not segments:
        if allow_whisper:
            st.error(f"Could not get transcript for: {title}")
        else:
            st.warning(f"No captions available for: **{title}**. Enable Whisper in the sidebar to transcribe videos without captions.")
        return False
    chunks = chunk_transcript_segments(segments)
    docs = [
        Document(
            page_content=chunk["text"],
            metadata={"source_url": url, "source_title": title, "start_time": chunk["start"], "end_time": chunk["end"]}
        )
        for chunk in chunks
    ]
    emb = get_embeddings()
    if not emb:
        st.error("OpenAI API key is required for indexing. Add it in the sidebar under 'API Keys'.")
        return False
    if st.session_state.vectordb is None:
        st.session_state.vectordb = FAISS.from_documents(docs, emb)
    else:
        st.session_state.vectordb.add_documents(docs)
    st.session_state.urls.append(url)
    st.session_state.video_metadata[url] = meta
    st.session_state.usage["videos_indexed"] += 1
    log_usage("INDEX", title=title, chunks=len(chunks), url=url)
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: QA Engine
# ═══════════════════════════════════════════════════════════════════════════════
def sanitize_question(question: str) -> tuple:
    question = question.strip()
    if not question:
        return False, "", "Please enter a question."
    if len(question) > MAX_QUESTION_LENGTH:
        return False, "", f"Question is too long ({len(question)} chars). Please keep it under {MAX_QUESTION_LENGTH} characters."
    if INJECTION_PATTERNS.search(question):
        return False, "", "Your question was flagged by our safety filter. Please rephrase it."
    return True, question, ""


def answer_question(question: str, provider: str, model: str, k: int = DEFAULT_RETRIEVER_K) -> str:
    ok, msg = check_rate_limit()
    if not ok:
        return msg
    retriever = st.session_state.vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": k, "fetch_k": k * 3},
    )
    docs = retriever.invoke(question)
    if not docs:
        return "I couldn't find relevant content for your question. Try rephrasing or index more videos."
    if provider == "Groq":
        answer = _answer_groq(question, docs, model)
    else:
        answer = _answer_openai(question, docs, model, k)
    context_texts = [d.page_content for d in docs]
    input_tokens = estimate_tokens(question + " ".join(context_texts))
    output_tokens = estimate_tokens(answer)
    record_usage(input_tokens, output_tokens, model)
    sources_md = _format_sources(docs)
    return f"{answer}\n\n{sources_md}"


def _answer_groq(question: str, docs: List[Document], model: str) -> str:
    client = get_groq_client()
    if not client:
        return "Groq API key not configured. Add it in the sidebar under 'API Keys'."
    context = "\n\n".join(d.page_content for d in docs)
    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant. Answer the user's question using ONLY the provided "
            "transcript context. If the context doesn't contain the answer, say so. "
            "Be concise and cite timestamps when relevant."
        )},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    response = client.chat.completions.create(model=model, messages=messages, temperature=0, max_tokens=1024)
    return response.choices[0].message.content


def _answer_openai(question: str, docs: List[Document], model: str, k: int) -> str:
    client = get_openai_client()
    if not client:
        return "OpenAI API key not configured. Add it in the sidebar under 'API Keys'."
    context = "\n\n".join(d.page_content for d in docs)
    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant. Answer the user's question using ONLY the provided "
            "transcript context. If the context doesn't contain the answer, say so. "
            "Be concise and cite timestamps when relevant."
        )},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    response = client.chat.completions.create(model=model, messages=messages, temperature=0, max_tokens=1024)
    return response.choices[0].message.content


def _format_sources(docs: List[Document]) -> str:
    seen = set()
    lines = ["---", "**Sources:**"]
    for doc in docs:
        meta = doc.metadata
        url = meta.get("source_url", "")
        title = meta.get("source_title", "Video")
        start = meta.get("start_time")
        end = meta.get("end_time")
        if start is not None and end is not None:
            time_ref = f"{format_time(start)} - {format_time(end)}"
            ts_url = format_timestamp_url(url, start)
            label = f"{title} [{time_ref}]"
            key = f"{url}:{int(start)}"
        else:
            ts_url = url
            label = title
            key = url
        if key not in seen:
            seen.add(key)
            lines.append(f"- [{label}]({ts_url})")
    return "\n".join(lines) if len(lines) > 2 else ""


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Streamlit UI
# ═══════════════════════════════════════════════════════════════════════════════
def render_custom_css():
    st.markdown("""
    <style>
    .video-card {
        background: #1A1A2E;
        border-radius: 12px;
        padding: 12px;
        margin: 8px 0;
        border: 1px solid #2A2A4A;
    }
    .video-card img { border-radius: 8px; width: 100%; }
    .video-card h4 { margin: 8px 0 4px 0; font-size: 0.95em; color: #FAFAFA; }
    .video-card p { margin: 0; font-size: 0.8em; color: #999; }
    .status-dot {
        display: inline-block; width: 8px; height: 8px;
        border-radius: 50%; margin-right: 6px;
    }
    .status-green { background: #00C853; }
    .status-red { background: #FF1744; }
    .how-it-works {
        background: #1A1A2E; border-radius: 12px; padding: 16px;
        border: 1px solid #2A2A4A; margin: 12px 0;
    }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


def _reset_clients():
    for k in ["openai_client", "groq_client", "embeddings"]:
        st.session_state.pop(k, None)


def render_sidebar():
    # ── Bring Your Own Key ──
    with st.sidebar.expander("API Keys", expanded=False):
        st.caption(
            "Your keys are stored only in your browser session and are never saved. "
            "They are sent directly to the API provider (OpenAI/Groq) and nowhere else."
        )
        user_openai = st.text_input(
            "OpenAI API Key", type="password",
            value=st.session_state.get("user_openai_api_key", ""),
            key="input_openai_key", placeholder="sk-...",
        )
        user_groq = st.text_input(
            "Groq API Key (free)", type="password",
            value=st.session_state.get("user_groq_api_key", ""),
            key="input_groq_key", placeholder="gsk_...",
        )
        if st.button("Save Keys", use_container_width=True):
            changed = False
            if user_openai != st.session_state.get("user_openai_api_key", ""):
                st.session_state["user_openai_api_key"] = user_openai
                changed = True
            if user_groq != st.session_state.get("user_groq_api_key", ""):
                st.session_state["user_groq_api_key"] = user_groq
                changed = True
            if changed:
                _reset_clients()
                st.success("Keys saved!")
                st.rerun()
            else:
                st.info("No changes.")
        st.markdown(
            "[Get a free Groq key](https://console.groq.com) · "
            "[Get an OpenAI key](https://platform.openai.com/api-keys)"
        )

    keys = validate_api_keys()

    # ── Model Settings ──
    with st.sidebar.expander("Model Settings", expanded=True):
        providers = []
        if keys["groq"]:
            providers.append("Groq")
        if keys["openai"]:
            providers.append("OpenAI")
        if not providers:
            st.warning("No API keys detected. Add your keys above to get started.")
            providers = ["Groq", "OpenAI"]

        provider = st.selectbox(
            "Provider",
            providers,
            index=0,
            help="Groq is free. OpenAI costs a small amount per query.",
        )
        st.session_state.provider = provider

        if provider == "Groq":
            model = st.selectbox("Model", [
                "openai/gpt-oss-20b",
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
            ], help="All Groq models are free to use with rate limits.")
        else:
            model = st.selectbox("Model", [
                "gpt-4o-mini",
                "gpt-3.5-turbo",
                "gpt-4o",
            ], help="OpenAI models charge per token. gpt-4o-mini is the cheapest.")
        st.session_state.model = model

        language = st.selectbox(
            "Caption Language", SUPPORTED_LANGUAGES,
            help="Language of the YouTube video captions to fetch.",
        )
        st.session_state.language = language

        allow_whisper = st.toggle("Enable Whisper", value=False, help="Use OpenAI Whisper to transcribe videos without captions. Requires an OpenAI key.")
        if allow_whisper:
            st.caption("Whisper costs ~$0.006/min. Only used when captions are unavailable.")
        st.session_state.allow_whisper = allow_whisper

        k = st.slider(
            "Context chunks", 2, 8, DEFAULT_RETRIEVER_K,
            help="How many transcript chunks to use when answering. More = broader context but slower.",
        )
        st.session_state.retriever_k = k

    # ── API Status ──
    with st.sidebar.expander("Connection Status"):
        for name, ok in keys.items():
            dot = "status-green" if ok else "status-red"
            label = "Connected" if ok else "Not configured"
            st.markdown(
                f'<span class="status-dot {dot}"></span> **{name.upper()}**: {label}',
                unsafe_allow_html=True,
            )
        st.caption(
            "**OpenAI** is needed for indexing videos (embeddings). "
            "**Groq** is needed if you select Groq as your chat provider."
        )


def render_video_card(meta: Dict):
    thumb = meta.get("thumbnail", "")
    title = meta.get("title", "Unknown")
    uploader = meta.get("uploader", "")
    duration = meta.get("duration", 0)
    dur_str = f"{duration // 60}:{duration % 60:02d}" if duration else ""
    thumb_html = f'<img src="{thumb}" alt="{title}">' if thumb else ""
    st.markdown(f"""
    <div class="video-card">
        {thumb_html}
        <h4>{title}</h4>
        <p>{uploader} &middot; {dur_str}</p>
    </div>
    """, unsafe_allow_html=True)


def render_hero():
    """App title and how-it-works section."""
    st.title("YouTube QA Bot")
    st.markdown("Index any YouTube video and ask questions about its content. Answers include clickable timestamp links.")

    with st.expander("How does this work?", expanded=False):
        st.markdown("""
**1. Paste a YouTube URL** and click "Index Video". The app fetches the video's captions (free) and stores them for search.

**2. Ask a question** in the chat below. The app finds the most relevant parts of the transcript and sends them to an AI model to generate an answer.

**3. Click the source links** in the answer to jump to the exact moment in the video.

**What you need:**
- An **OpenAI API key** is required for indexing (embedding the transcript). It costs ~$0.0001 per video.
- A **Groq API key** (free) is recommended for chat. [Get one here](https://console.groq.com).
- You can also use OpenAI models for chat, but they cost more per query.
- Videos **must have captions** unless you enable Whisper (paid) in the sidebar.
        """)


def render_indexing_section():
    st.subheader("Step 1: Index a Video")

    tab_video, tab_playlist = st.tabs(["Single Video", "Playlist"])

    with tab_video:
        url_input = st.text_input(
            "YouTube video URL:",
            key="url_input",
            placeholder="https://youtube.com/watch?v=...",
            label_visibility="collapsed",
        )
        if st.button("Index Video", key="btn_index_video", use_container_width=True, type="primary"):
            if not url_input:
                st.warning("Please paste a YouTube URL above.")
            else:
                try:
                    get_video_id(url_input)
                except ValueError as e:
                    st.error(str(e))
                    return
                with st.spinner("Fetching transcript and indexing..."):
                    success = process_single_video(
                        url_input, st.session_state.language, st.session_state.allow_whisper,
                    )
                    if success:
                        title = st.session_state.video_metadata.get(url_input, {}).get("title", "")
                        st.success(f"Ready! You can now ask questions about: **{title}**")

    with tab_playlist:
        playlist_input = st.text_input(
            "YouTube playlist URL:",
            key="playlist_input",
            placeholder="https://youtube.com/playlist?list=...",
            label_visibility="collapsed",
        )
        if st.button("Index Playlist", key="btn_index_playlist", use_container_width=True, type="primary"):
            if not playlist_input:
                st.warning("Please paste a YouTube playlist URL above.")
            elif not is_playlist_url(playlist_input):
                st.error("That doesn't look like a playlist URL. Make sure it contains `list=` in the URL.")
            else:
                videos = get_playlist_videos(playlist_input)
                if not videos:
                    st.error("Could not fetch videos from this playlist. Check the URL and try again.")
                else:
                    progress = st.progress(0, text="Starting...")
                    success_count = 0
                    for i, video in enumerate(videos):
                        ok, msg = check_video_limit()
                        if not ok:
                            st.warning(msg)
                            break
                        progress.progress(
                            (i + 1) / len(videos),
                            text=f"Indexing {i + 1}/{len(videos)}: {video['title'][:50]}",
                        )
                        if process_single_video(
                            video["url"], st.session_state.language,
                            st.session_state.allow_whisper, video["title"],
                        ):
                            success_count += 1
                    progress.empty()
                    st.success(f"Done! Indexed {success_count} out of {len(videos)} videos.")

    # Show indexed videos
    if st.session_state.urls:
        st.markdown(f"**{len(st.session_state.urls)} video(s) indexed:**")
        cols = st.columns(min(3, len(st.session_state.urls)))
        for i, url in enumerate(st.session_state.urls):
            meta = st.session_state.video_metadata.get(url, {"title": "Unknown", "url": url})
            with cols[i % len(cols)]:
                render_video_card(meta)


def render_chat_section():
    st.subheader("Step 2: Ask Questions")

    if not st.session_state.urls:
        st.info("Index a YouTube video above, then come back here to ask questions about it.")
        return

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("What would you like to know about the video?"):
        ok, cleaned, msg = sanitize_question(question)
        if not ok:
            st.warning(msg)
            return

        st.session_state.chat_history.append({"role": "user", "content": cleaned})
        with st.chat_message("user"):
            st.markdown(cleaned)

        with st.spinner("Finding the answer..."):
            try:
                answer = answer_question(
                    cleaned, st.session_state.provider,
                    st.session_state.model, st.session_state.retriever_k,
                )
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
            except Exception as e:
                error_msg = f"Something went wrong: {e}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.error(error_msg)

    if st.session_state.chat_history:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


def render_footer():
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#666;font-size:0.85em;'>"
        "Built with Streamlit, LangChain, FAISS, Groq & OpenAI"
        "</p>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════
def init_session_state():
    defaults = {
        "vectordb": None,
        "chat_history": [],
        "urls": [],
        "video_metadata": {},
        "provider": "Groq",
        "model": "openai/gpt-oss-20b",
        "language": "en",
        "allow_whisper": False,
        "retriever_k": DEFAULT_RETRIEVER_K,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    init_usage_state()


def main():
    st.set_page_config(
        page_title="YouTube QA Bot",
        page_icon="https://www.youtube.com/favicon.ico",
        layout="wide",
    )
    render_custom_css()
    init_session_state()
    render_hero()
    render_sidebar()
    render_indexing_section()
    render_chat_section()
    render_footer()


if __name__ == "__main__":
    main()
