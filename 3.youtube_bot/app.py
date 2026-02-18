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
import tempfile
from datetime import date
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

load_dotenv()

# ── Tunable constants ────────────────────────────────────────────────────────
SESSION_TOKEN_BUDGET = 20_000
DAILY_TOKEN_BUDGET = 50_000
MAX_QUERIES_PER_MINUTE = 5
MAX_VIDEOS_PER_SESSION = 10
MAX_PLAYLIST_VIDEOS = 15
CHUNK_SIZE = 1000
DEFAULT_RETRIEVER_K = 4
MAX_QUESTION_LENGTH = 1000

MODEL_COSTS = {  # per 1K tokens
    "llama3-8b-8192": 0.0,
    "llama3-70b-8192": 0.0,
    "llama-3.1-8b-instant": 0.0,
    "mixtral-8x7b-32768": 0.0,
    "gpt-4o-mini": 0.00015,
    "gpt-3.5-turbo": 0.0005,
    "gpt-4o": 0.005,
    "whisper-1": 0.006,  # per minute, not per token
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
    """Read from st.secrets first (Streamlit Cloud), fall back to os.getenv."""
    try:
        val = st.secrets.get(name)
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(name)


def get_openai_client():
    """Lazy-init OpenAI client, stored in session_state."""
    if "openai_client" not in st.session_state:
        key = _get_api_key("OPENAI_API_KEY")
        if key and OPENAI_AVAILABLE:
            st.session_state.openai_client = OpenAI(api_key=key)
        else:
            st.session_state.openai_client = None
    return st.session_state.openai_client


def get_groq_client():
    """Lazy-init Groq client, stored in session_state."""
    if "groq_client" not in st.session_state:
        key = _get_api_key("GROQ_API_KEY")
        if key and GROQ_AVAILABLE:
            st.session_state.groq_client = Groq(api_key=key)
        else:
            st.session_state.groq_client = None
    return st.session_state.groq_client


def get_embeddings():
    """Lazy-init OpenAI embeddings."""
    if "embeddings" not in st.session_state:
        key = _get_api_key("OPENAI_API_KEY")
        if key:
            st.session_state.embeddings = OpenAIEmbeddings(api_key=key)
        else:
            st.session_state.embeddings = None
    return st.session_state.embeddings


def validate_api_keys() -> Dict[str, bool]:
    """Return which API providers are available."""
    return {
        "openai": bool(_get_api_key("OPENAI_API_KEY")) and OPENAI_AVAILABLE,
        "groq": bool(_get_api_key("GROQ_API_KEY")) and GROQ_AVAILABLE,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Token Budget & Rate Limiter
# ═══════════════════════════════════════════════════════════════════════════════
def init_usage_state():
    """Initialize all usage-tracking counters in session_state."""
    today = date.today().isoformat()
    if "usage" not in st.session_state or st.session_state.usage.get("date") != today:
        st.session_state.usage = {
            "date": today,
            "session_tokens": 0,
            "daily_tokens": 0,
            "session_cost": 0.0,
            "daily_cost": 0.0,
            "query_count": 0,
            "videos_indexed": 0,
            "whisper_minutes": 0.0,
            "query_timestamps": [],
        }


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def estimate_query_cost(question: str, context_chunks: List[str], model: str) -> tuple:
    """Pre-flight cost estimate. Returns (estimated_tokens, estimated_cost)."""
    input_text = question + " ".join(context_chunks)
    est_input = estimate_tokens(input_text)
    est_output = 256  # conservative average
    total = est_input + est_output
    cost_per_k = MODEL_COSTS.get(model, 0.0)
    cost = (total / 1000) * cost_per_k
    return total, cost


def check_budget(estimated_tokens: int) -> tuple:
    """Returns (ok, message). Blocks if over session or daily cap."""
    usage = st.session_state.usage
    if usage["session_tokens"] + estimated_tokens > SESSION_TOKEN_BUDGET:
        remaining = max(0, SESSION_TOKEN_BUDGET - usage["session_tokens"])
        return False, (
            f"Session token budget exceeded ({usage['session_tokens']:,}/{SESSION_TOKEN_BUDGET:,}). "
            f"~{remaining:,} tokens remaining. Try a shorter question or switch to Groq (free)."
        )
    if usage["daily_tokens"] + estimated_tokens > DAILY_TOKEN_BUDGET:
        remaining = max(0, DAILY_TOKEN_BUDGET - usage["daily_tokens"])
        return False, (
            f"Daily token budget exceeded ({usage['daily_tokens']:,}/{DAILY_TOKEN_BUDGET:,}). "
            f"~{remaining:,} tokens remaining. Come back tomorrow or switch to Groq (free)."
        )
    return True, ""


def check_rate_limit() -> tuple:
    """Sliding-window rate limiter. Returns (ok, message)."""
    now = time.time()
    usage = st.session_state.usage
    # Prune timestamps older than 60s
    usage["query_timestamps"] = [t for t in usage["query_timestamps"] if now - t < 60]
    if len(usage["query_timestamps"]) >= MAX_QUERIES_PER_MINUTE:
        oldest = usage["query_timestamps"][0]
        wait = int(60 - (now - oldest)) + 1
        return False, f"Rate limit reached ({MAX_QUERIES_PER_MINUTE}/min). Please wait {wait}s."
    return True, ""


def record_usage(input_tokens: int, output_tokens: int, model: str):
    """Post-call accounting."""
    total = input_tokens + output_tokens
    cost_per_k = MODEL_COSTS.get(model, 0.0)
    cost = (total / 1000) * cost_per_k

    usage = st.session_state.usage
    usage["session_tokens"] += total
    usage["daily_tokens"] += total
    usage["session_cost"] += cost
    usage["daily_cost"] += cost
    usage["query_count"] += 1
    usage["query_timestamps"].append(time.time())


def record_whisper_usage(duration_minutes: float):
    """Track Whisper cost separately."""
    cost = duration_minutes * MODEL_COSTS.get("whisper-1", 0.006)
    usage = st.session_state.usage
    usage["whisper_minutes"] += duration_minutes
    usage["session_cost"] += cost
    usage["daily_cost"] += cost


def check_video_limit() -> tuple:
    """Blocks indexing beyond session cap. Returns (ok, message)."""
    count = st.session_state.usage.get("videos_indexed", 0)
    if count >= MAX_VIDEOS_PER_SESSION:
        return False, f"Session video limit reached ({MAX_VIDEOS_PER_SESSION}). Start a new session to index more."
    return True, ""


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: YouTube Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def get_video_id(url: str) -> str:
    """Extract 11-char YouTube video ID with robust regex."""
    match = VIDEO_URL_PATTERN.search(url)
    if match:
        return match.group(1)
    raise ValueError("Invalid YouTube URL. Supported: watch, shorts, live, embed, youtu.be")


def get_video_metadata(url: str) -> Dict:
    """Get title, duration, thumbnail, uploader via yt-dlp."""
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
    """Check if URL contains a playlist parameter."""
    return bool(PLAYLIST_URL_PATTERN.search(url))


def get_playlist_videos(url: str, max_videos: int = MAX_PLAYLIST_VIDEOS) -> List[Dict]:
    """Get video URLs and titles from a playlist, capped at max_videos."""
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,
        "playlist_items": f"1:{max_videos}",
        "no_warnings": True,
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
    """Fetch YouTube captions as timestamped segments (FREE API)."""
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=[language])
        return transcript.to_raw_data()
    except Exception:
        # Try without language filter as fallback
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript = ytt_api.fetch(video_id)
            return transcript.to_raw_data()
        except Exception:
            return None


@st.cache_data(show_spinner=False)
def download_audio(url: str) -> str:
    """Download audio as MP3 to a temp directory."""
    import glob as globmod

    tmp_dir = tempfile.mkdtemp()
    video_id = get_video_id(url)
    output_template = os.path.join(tmp_dir, f"{video_id}.%(ext)s")

    opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "128"}
        ],
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
    """Transcribe audio with OpenAI Whisper. Cleans up temp dir after."""
    client = get_openai_client()
    if not client:
        raise RuntimeError("OpenAI API key required for Whisper transcription.")
    with open(mp3_path, "rb") as f:
        resp = client.audio.transcriptions.create(model="whisper-1", file=f)

    # Track cost: estimate duration from file size (~128kbps)
    file_size = os.path.getsize(mp3_path)
    est_minutes = (file_size / (128 * 1024 / 8)) / 60
    record_whisper_usage(est_minutes)

    # Cleanup temp directory
    tmp_dir = os.path.dirname(mp3_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return resp.text


def format_time(seconds: float) -> str:
    """Format seconds as M:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def format_timestamp_url(url: str, seconds: float) -> str:
    """Create a clickable YouTube URL at the given timestamp."""
    video_id = get_video_id(url)
    return f"https://youtu.be/{video_id}?t={int(seconds)}"


def chunk_transcript_segments(segments: List[Dict], chunk_size: int = CHUNK_SIZE) -> List[Dict]:
    """Merge caption segments into ~chunk_size char chunks with time refs."""
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
            chunks.append({
                "text": f"{current_text.strip()} {time_ref}",
                "start": current_start,
                "end": current_end,
            })
            current_text = seg_text + " "
            current_start = seg_start
        else:
            current_text += seg_text + " "

        current_end = seg_end

    if current_text:
        time_ref = f"[{format_time(current_start)} - {format_time(current_end)}]"
        chunks.append({
            "text": f"{current_text.strip()} {time_ref}",
            "start": current_start,
            "end": current_end,
        })

    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Indexing Engine
# ═══════════════════════════════════════════════════════════════════════════════
def get_transcript_segments(url: str, language: str, allow_whisper: bool) -> Optional[List[Dict]]:
    """Get transcript: captions first (free), Whisper only if opted in."""
    video_id = get_video_id(url)
    segments = fetch_captions_segments(video_id, language)
    if segments:
        return segments

    if not allow_whisper:
        return None

    # Whisper fallback
    try:
        st.info("No captions found. Transcribing with Whisper (paid)...")
        mp3_path = download_audio(url)
        transcript_text = transcribe_with_whisper(mp3_path)
    except Exception as e:
        st.error(f"Whisper transcription failed: {e}")
        return None

    # Convert plain text to pseudo-segments
    words = transcript_text.split()
    segments = []
    words_per_chunk = 100
    for i in range(0, len(words), words_per_chunk):
        chunk_words = words[i:i + words_per_chunk]
        segments.append({
            "text": " ".join(chunk_words),
            "start": i * 0.5,
            "duration": len(chunk_words) * 0.5,
        })
    return segments


def process_single_video(url: str, language: str, allow_whisper: bool, title: str = "") -> bool:
    """Full pipeline: budget check -> metadata -> transcript -> chunk -> embed -> FAISS."""
    # Check video limit
    ok, msg = check_video_limit()
    if not ok:
        st.warning(msg)
        return False

    # Skip if already indexed
    if url in st.session_state.get("urls", []):
        st.info(f"Already indexed: {url}")
        return True

    # Get metadata
    if not title:
        meta = get_video_metadata(url)
        title = meta.get("title", "Unknown Title")
    else:
        meta = get_video_metadata(url)
        meta["title"] = title

    # Get transcript
    segments = get_transcript_segments(url, language, allow_whisper)
    if not segments:
        if allow_whisper:
            st.error(f"Could not get transcript for: {title}")
        else:
            st.warning(f"No captions available for: {title}. Enable Whisper to transcribe.")
        return False

    # Chunk
    chunks = chunk_transcript_segments(segments)

    # Estimate embedding cost
    all_text = " ".join(c["text"] for c in chunks)
    embed_tokens = estimate_tokens(all_text)
    embed_cost = (embed_tokens / 1000) * 0.00002  # text-embedding-ada-002 pricing

    # Create documents
    docs = [
        Document(
            page_content=chunk["text"],
            metadata={
                "source_url": url,
                "source_title": title,
                "start_time": chunk["start"],
                "end_time": chunk["end"],
            }
        )
        for chunk in chunks
    ]

    # Embed & store in FAISS
    emb = get_embeddings()
    if not emb:
        st.error("OpenAI API key required for embeddings.")
        return False

    if st.session_state.vectordb is None:
        st.session_state.vectordb = FAISS.from_documents(docs, emb)
    else:
        st.session_state.vectordb.add_documents(docs)

    # Track
    st.session_state.urls.append(url)
    st.session_state.video_metadata[url] = meta
    st.session_state.usage["videos_indexed"] += 1
    st.session_state.usage["session_cost"] += embed_cost
    st.session_state.usage["daily_cost"] += embed_cost

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: QA Engine
# ═══════════════════════════════════════════════════════════════════════════════
def sanitize_question(question: str) -> tuple:
    """Length cap + prompt injection detection. Returns (ok, cleaned, message)."""
    question = question.strip()
    if not question:
        return False, "", "Please enter a question."
    if len(question) > MAX_QUESTION_LENGTH:
        return False, "", f"Question too long ({len(question)} chars). Max {MAX_QUESTION_LENGTH}."
    if INJECTION_PATTERNS.search(question):
        return False, "", "Your question was flagged. Please rephrase."
    return True, question, ""


def answer_question(question: str, provider: str, model: str, k: int = DEFAULT_RETRIEVER_K) -> str:
    """Main QA entry point: rate limit -> retrieve -> budget check -> LLM -> format."""
    # Rate limit
    ok, msg = check_rate_limit()
    if not ok:
        return msg

    # Retrieve with MMR for diverse context
    retriever = st.session_state.vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 3},
    )
    docs = retriever.invoke(question)
    if not docs:
        return "No relevant content found. Try a different question or index more videos."

    # Pre-flight cost estimate
    context_texts = [d.page_content for d in docs]
    est_tokens, est_cost = estimate_query_cost(question, context_texts, model)

    ok, msg = check_budget(est_tokens)
    if not ok:
        return msg

    # Call LLM
    if provider == "Groq":
        answer = _answer_groq(question, docs, model)
    else:
        answer = _answer_openai(question, docs, model, k)

    # Post-call usage tracking (estimate since we don't get exact counts from Groq)
    input_tokens = estimate_tokens(question + " ".join(context_texts))
    output_tokens = estimate_tokens(answer)
    record_usage(input_tokens, output_tokens, model)

    # Format with sources
    sources_md = _format_sources(docs)
    return f"{answer}\n\n{sources_md}"


def _answer_groq(question: str, docs: List[Document], model: str) -> str:
    """Call Groq API with manual context injection."""
    client = get_groq_client()
    if not client:
        return "Groq API key not configured."

    context = "\n\n".join(d.page_content for d in docs)
    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant. Answer the user's question using ONLY the provided "
            "transcript context. If the context doesn't contain the answer, say so. "
            "Be concise and cite timestamps when relevant."
        )},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def _answer_openai(question: str, docs: List[Document], model: str, k: int) -> str:
    """Call OpenAI directly with retrieved context."""
    client = get_openai_client()
    if not client:
        return "OpenAI API key not configured."

    context = "\n\n".join(d.page_content for d in docs)
    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant. Answer the user's question using ONLY the provided "
            "transcript context. If the context doesn't contain the answer, say so. "
            "Be concise and cite timestamps when relevant."
        )},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def _format_sources(docs: List[Document]) -> str:
    """Format source documents as markdown with clickable timestamp links."""
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
    .video-card img {
        border-radius: 8px;
        width: 100%;
    }
    .video-card h4 {
        margin: 8px 0 4px 0;
        font-size: 0.95em;
        color: #FAFAFA;
    }
    .video-card p {
        margin: 0;
        font-size: 0.8em;
        color: #999;
    }
    .budget-bar {
        height: 8px;
        border-radius: 4px;
        background: #2A2A4A;
        overflow: hidden;
        margin: 4px 0;
    }
    .budget-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    .budget-green { background: #00C853; }
    .budget-yellow { background: #FFD600; }
    .budget-red { background: #FF1744; }
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-green { background: #00C853; }
    .status-red { background: #FF1744; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Sidebar with model settings, usage dashboard, and API status."""
    keys = validate_api_keys()

    # ── Model Settings ──
    with st.sidebar.expander("Model Settings", expanded=True):
        providers = []
        if keys["groq"]:
            providers.append("Groq")
        if keys["openai"]:
            providers.append("OpenAI")
        if not providers:
            st.error("No API keys configured. Add keys to .env or .streamlit/secrets.toml")
            providers = ["Groq"]  # show dropdown anyway

        provider = st.selectbox("Provider", providers, index=0)
        st.session_state.provider = provider

        if provider == "Groq":
            model = st.selectbox("Model", [
                "llama3-8b-8192",
                "llama3-70b-8192",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
            ])
        else:
            model = st.selectbox("Model", [
                "gpt-4o-mini",
                "gpt-3.5-turbo",
                "gpt-4o",
            ])
        st.session_state.model = model

        language = st.selectbox("Language", SUPPORTED_LANGUAGES)
        st.session_state.language = language

        allow_whisper = st.toggle("Enable Whisper (paid)", value=False)
        if allow_whisper:
            st.caption("Whisper costs ~$0.006/min. Only used when captions are unavailable.")
        st.session_state.allow_whisper = allow_whisper

        k = st.slider("Context chunks (k)", 2, 8, DEFAULT_RETRIEVER_K)
        st.session_state.retriever_k = k

    # ── Usage Dashboard ──
    with st.sidebar.expander("Usage Dashboard", expanded=True):
        usage = st.session_state.usage

        col1, col2 = st.columns(2)
        col1.metric("Tokens", f"{usage['session_tokens']:,}")
        col2.metric("Est. Cost", f"${usage['session_cost']:.4f}")

        col3, col4 = st.columns(2)
        col3.metric("Videos", usage["videos_indexed"])
        col4.metric("Queries", usage["query_count"])

        # Budget progress bar
        pct = min(100, int(usage["session_tokens"] / SESSION_TOKEN_BUDGET * 100))
        if pct < 60:
            color = "budget-green"
        elif pct < 85:
            color = "budget-yellow"
        else:
            color = "budget-red"

        st.markdown(f"""
        <div class="budget-bar">
            <div class="budget-fill {color}" style="width:{pct}%"></div>
        </div>
        <p style="font-size:0.75em;color:#888;text-align:center;">
            Session budget: {pct}% used ({usage['session_tokens']:,}/{SESSION_TOKEN_BUDGET:,})
        </p>
        """, unsafe_allow_html=True)

        if usage["whisper_minutes"] > 0:
            st.caption(f"Whisper: {usage['whisper_minutes']:.1f} min (~${usage['whisper_minutes'] * 0.006:.3f})")

    # ── API Status ──
    with st.sidebar.expander("API Status"):
        for name, ok in keys.items():
            dot = "status-green" if ok else "status-red"
            label = "Connected" if ok else "Not configured"
            st.markdown(
                f'<span class="status-dot {dot}"></span> **{name.upper()}**: {label}',
                unsafe_allow_html=True,
            )


def render_video_card(meta: Dict):
    """Render a video thumbnail + title card."""
    thumb = meta.get("thumbnail", "")
    title = meta.get("title", "Unknown")
    uploader = meta.get("uploader", "")
    duration = meta.get("duration", 0)
    url = meta.get("url", "")

    dur_str = f"{duration // 60}:{duration % 60:02d}" if duration else ""

    thumb_html = f'<img src="{thumb}" alt="{title}">' if thumb else ""
    st.markdown(f"""
    <div class="video-card">
        {thumb_html}
        <h4>{title}</h4>
        <p>{uploader} &middot; {dur_str}</p>
    </div>
    """, unsafe_allow_html=True)


def render_indexing_section():
    """Video/playlist indexing with tabs."""
    st.subheader("Index YouTube Content")

    tab_video, tab_playlist = st.tabs(["Single Video", "Playlist"])

    with tab_video:
        url_input = st.text_input("YouTube video URL:", key="url_input", placeholder="https://youtube.com/watch?v=...")
        if st.button("Index Video", key="btn_index_video", use_container_width=True):
            if not url_input:
                st.warning("Enter a valid YouTube URL.")
            else:
                try:
                    get_video_id(url_input)  # validate URL
                except ValueError as e:
                    st.error(str(e))
                    return

                with st.spinner("Indexing video..."):
                    success = process_single_video(
                        url_input,
                        st.session_state.language,
                        st.session_state.allow_whisper,
                    )
                    if success:
                        title = st.session_state.video_metadata.get(url_input, {}).get("title", "")
                        st.success(f"Indexed: {title}")

    with tab_playlist:
        playlist_input = st.text_input("YouTube playlist URL:", key="playlist_input", placeholder="https://youtube.com/playlist?list=...")
        if st.button("Index Playlist", key="btn_index_playlist", use_container_width=True):
            if not playlist_input:
                st.warning("Enter a valid playlist URL.")
            elif not is_playlist_url(playlist_input):
                st.error("Not a valid playlist URL.")
            else:
                videos = get_playlist_videos(playlist_input)
                if not videos:
                    st.error("Could not fetch playlist videos.")
                else:
                    progress = st.progress(0, text="Indexing playlist...")
                    success_count = 0
                    for i, video in enumerate(videos):
                        ok, msg = check_video_limit()
                        if not ok:
                            st.warning(msg)
                            break
                        progress.progress(
                            (i + 1) / len(videos),
                            text=f"Indexing {i + 1}/{len(videos)}: {video['title'][:40]}...",
                        )
                        if process_single_video(
                            video["url"],
                            st.session_state.language,
                            st.session_state.allow_whisper,
                            video["title"],
                        ):
                            success_count += 1
                    progress.empty()
                    st.success(f"Indexed {success_count}/{len(videos)} videos from playlist.")

    # Show indexed video cards
    if st.session_state.urls:
        st.markdown("**Indexed Videos:**")
        cols = st.columns(min(3, len(st.session_state.urls)))
        for i, url in enumerate(st.session_state.urls):
            meta = st.session_state.video_metadata.get(url, {"title": "Unknown", "url": url})
            with cols[i % len(cols)]:
                render_video_card(meta)


def render_chat_section():
    """Chat interface with message bubbles."""
    st.subheader("Chat with Videos")

    if not st.session_state.urls:
        st.info("Index a video above to start chatting.")
        return

    # Display history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if question := st.chat_input("Ask a question about the indexed videos..."):
        # Sanitize
        ok, cleaned, msg = sanitize_question(question)
        if not ok:
            st.warning(msg)
            return

        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": cleaned})
        with st.chat_message("user"):
            st.markdown(cleaned)

        # Get answer
        with st.spinner("Thinking..."):
            try:
                answer = answer_question(
                    cleaned,
                    st.session_state.provider,
                    st.session_state.model,
                    st.session_state.retriever_k,
                )
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
            except Exception as e:
                error_msg = f"Error: {e}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.error(error_msg)

    # Clear chat
    if st.session_state.chat_history:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


def render_footer():
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#666;font-size:0.85em;'>"
        "YouTube QA Bot &middot; Groq + OpenAI + LangChain + FAISS + Streamlit"
        "</p>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════
def init_session_state():
    """One-time initialization of all session state keys."""
    defaults = {
        "vectordb": None,
        "chat_history": [],
        "urls": [],
        "video_metadata": {},
        "provider": "Groq",
        "model": "llama3-8b-8192",
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
    st.title("YouTube QA Bot")
    render_sidebar()
    render_indexing_section()
    render_chat_section()
    render_footer()


if __name__ == "__main__":
    main()
