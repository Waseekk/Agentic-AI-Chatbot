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

# Helper: extract 11‑char YouTube ID with robust regex
def get_video_id(url: str) -> str:
    # More comprehensive YouTube URL patterns
    patterns = [
        r'(?:v=|v\/|embed\/|youtu\.be\/|\/v\/|watch\?v=|watch\?.*&v=)([A-Za-z0-9_-]{11})',
        r'(?:shorts\/)([A-Za-z0-9_-]{11})',
        r'(?:live\/)([A-Za-z0-9_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    st.error("❌ Invalid YouTube URL. Ensure it's a valid YouTube link.")
    st.stop()

# Helper: get video title using yt_dlp (new to get metadata)
def get_video_title(url: str) -> Optional[str]:
    ydl_opts = {"quiet": True, "skip_download": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            return info.get("title", None)
        except Exception:
            return None

# Helper: check if URL is a playlist
def is_playlist_url(url: str) -> bool:
    playlist_patterns = [
        r'[?&]list=([a-zA-Z0-9_-]+)',
        r'playlist\?list=([a-zA-Z0-9_-]+)'
    ]
    return any(re.search(pattern, url) for pattern in playlist_patterns)

# Helper: get playlist videos
def get_playlist_videos(playlist_url: str) -> list:
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,  # Only extract video URLs, not full info
        "playlist_items": "1:50"  # Limit to first 50 videos to avoid overwhelming
    }
    
    videos = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            playlist_info = ydl.extract_info(playlist_url, download=False)
            
            if 'entries' in playlist_info:
                for entry in playlist_info['entries']:
                    if entry and 'id' in entry:
                        video_url = f"https://www.youtube.com/watch?v={entry['id']}"
                        video_title = entry.get('title', f"Video {entry['id']}")
                        videos.append({
                            'url': video_url,
                            'title': video_title,
                            'id': entry['id']
                        })
    except Exception as e:
        st.error(f"Error extracting playlist: {str(e)}")
    
    return videos

# Cache caption fetch - returns segments with timestamps
@st.cache_data(show_spinner=True)
def fetch_captions_segments(video_id: str, language: str = "en") -> Optional[list]:
    try:
        segments = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        return segments
    except (NoTranscriptFound, Exception) as e:
        # Handle NoTranscriptFound, TranscriptsDisabled, VideoUnavailable, network errors, etc.
        return None

# Cache caption fetch - returns plain text
@st.cache_data(show_spinner=True)
def fetch_captions(video_id: str, language: str = "en") -> Optional[str]:
    try:
        segments = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        return "\n".join(seg["text"] for seg in segments)
    except (NoTranscriptFound, Exception) as e:
        # Handle NoTranscriptFound, TranscriptsDisabled, VideoUnavailable, network errors, etc.
        return None

# Cache audio download
@st.cache_data(show_spinner=True)
def download_audio(url: str) -> str:
    import glob
    
    # Create a temporary directory for downloads
    tmp_dir = tempfile.mkdtemp()
    video_id = get_video_id(url)
    
    # Use proper outtmpl template
    output_template = os.path.join(tmp_dir, f"{video_id}.%(ext)s")
    
    opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
        ],
        "quiet": True
    }
    
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        
        # Find the downloaded MP3 file
        mp3_files = glob.glob(os.path.join(tmp_dir, f"{video_id}.mp3"))
        if mp3_files:
            return mp3_files[0]
        else:
            # Fallback: look for any MP3 file in the directory
            mp3_files = glob.glob(os.path.join(tmp_dir, "*.mp3"))
            if mp3_files:
                return mp3_files[0]
            else:
                raise FileNotFoundError("No MP3 file found after download")
                
    except Exception as e:
        # Clean up temp directory on error
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise e

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

# Get transcript segments with timestamps: try captions, else Whisper (converted to segments)
@st.cache_data(show_spinner=True)
def get_transcript_segments(url: str, language: str, model: str) -> list:
    vid = get_video_id(url)
    segments = fetch_captions_segments(vid, language)
    if segments:
        return segments
    
    st.info("⚠️ No captions found. Transcribing via Whisper...")
    mp3 = download_audio(url)
    transcript_text = transcribe_with_whisper(mp3, model)
    
    # Convert plain text to segments (without timestamps since Whisper doesn't provide them easily)
    # Split into chunks of roughly 100 words each
    words = transcript_text.split()
    segments = []
    chunk_size = 100
    
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i+chunk_size]
        segments.append({
            "text": " ".join(chunk_words),
            "start": i * 2,  # Rough estimate: 2 seconds per word group
            "duration": len(chunk_words) * 0.5  # Rough estimate: 0.5 seconds per word
        })
    
    return segments

# Helper: format time in HH:MM:SS
def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"

# Helper: chunk transcript segments
def chunk_transcript_segments(segments, chunk_size=1000):
    chunks = []
    current_text = ""
    current_start = None
    current_end = None

    for seg in segments:
        seg_text = seg["text"].strip()
        seg_start = seg["start"]
        seg_end = seg_start + seg["duration"]

        if current_start is None:
            current_start = seg_start

        # If adding this segment exceeds chunk size, finalize current chunk
        if len(current_text) + len(seg_text) + 1 > chunk_size:
            time_ref = f"[{format_time(current_start)} - {format_time(current_end)}]"
            chunks.append({
                "text": current_text.strip() + " " + time_ref,
                "start": current_start,
                "end": current_end
            })
            # Start new chunk
            current_text = seg_text + " "
            current_start = seg_start
        else:
            current_text += seg_text + " "

        current_end = seg_end

    # Add last chunk
    if current_text:
        time_ref = f"[{format_time(current_start)} - {format_time(current_end)}]"
        chunks.append({
            "text": current_text.strip() + " " + time_ref,
            "start": current_start,
            "end": current_end
        })

    return chunks

# Helper function to process a single video - MOVED HERE BEFORE IT'S CALLED
def process_single_video(url: str, language: str, whisper_model: str, title: str = None) -> bool:
    try:
        # Fetch video title if not provided
        if title is None:
            title = get_video_title(url) or "Unknown Title"
        
        # Store metadata
        st.session_state.video_metadata[url] = {"title": title, "url": url}

        # Use the consistent transcript fetching logic with fallback
        segments = get_transcript_segments(url, language, whisper_model)
        
        if not segments:
            return False
            
        # Chunk the segments
        chunks = chunk_transcript_segments(segments)

        docs = [
            Document(
                page_content=chunk["text"],
                metadata={
                    "source_url": url,
                    "source_title": title,
                    "start_time": chunk["start"],
                    "end_time": chunk["end"]
                }
            )
            for chunk in chunks
        ]

        # Add to vector database
        if st.session_state.vectordb is None:
            st.session_state.vectordb = FAISS.from_documents(docs, embeddings)
        else:
            st.session_state.vectordb.add_documents(docs)
        
        # Only add to URLs list if not already there
        if url not in st.session_state.urls:
            st.session_state.urls.append(url)
        
        return True
        
    except Exception as e:
        return False

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
st.subheader("📥 1. Index YouTube Content")

# Input type selection
input_type = st.radio(
    "Select input type:",
    ["Single Video", "Playlist"],
    horizontal=True
)

if input_type == "Single Video":
    url_input = st.text_input("Video URL to index:", key="url_input")
    if st.button("Index Video"):
        if not url_input:
            st.warning("❗ Please enter a valid YouTube URL.")
        else:
            with st.spinner("Indexing video..."):
                try:
                    # Process single video
                    success = process_single_video(url_input, language, whisper_model)
                    if success:
                        st.success(f"✅ Successfully indexed video!")
                except Exception as e:
                    st.error(f"❌ Error indexing video: {str(e)}")

else:  # Playlist
    playlist_url = st.text_input("Playlist URL to index:", key="playlist_input")
    if st.button("Index Playlist"):
        if not playlist_url:
            st.warning("❗ Please enter a valid YouTube playlist URL.")
        elif not is_playlist_url(playlist_url):
            st.error("❌ This doesn't appear to be a valid playlist URL.")
        else:
            with st.spinner("Extracting playlist videos..."):
                videos = get_playlist_videos(playlist_url)
                
                if not videos:
                    st.error("❌ No videos found in playlist or playlist is private.")
                else:
                    st.info(f"Found {len(videos)} videos in playlist. Starting indexing...")
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    successful_indexes = 0
                    failed_indexes = 0
                    
                    for i, video in enumerate(videos):
                        try:
                            status_text.text(f"Processing: {video['title'][:50]}...")
                            
                            # Process each video
                            success = process_single_video(video['url'], language, whisper_model, video['title'])
                            
                            if success:
                                successful_indexes += 1
                            else:
                                failed_indexes += 1
                                
                        except Exception as e:
                            st.warning(f"⚠️ Failed to index '{video['title']}': {str(e)}")
                            failed_indexes += 1
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(videos))
                    
                    # Final status
                    status_text.text("Indexing complete!")
                    st.success(f"✅ Playlist indexing complete! Successfully indexed: {successful_indexes}, Failed: {failed_indexes}")

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
                # Initialize answer_with_sources to avoid scope issues
                answer_with_sources = ""
                
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
                    
                    # 5. Add source information for GROQ (Fix for bug #6)
                    if docs:
                        sources = {}
                        for doc in docs:
                            meta = doc.metadata
                            url = meta.get("source_url", "")
                            title = meta.get("source_title", "Video")
                            start = meta.get("start_time", None)
                            end = meta.get("end_time", None)
                            if start is not None and end is not None:
                                time_ref = f"{format_time(start)} - {format_time(end)}"
                                label = f"{title} [{time_ref}]"
                            else:
                                label = title
                            sources[url] = label

                        source_md = "\n\n---\n**Sources:**\n"
                        for url, label in sources.items():
                            source_md += f"- [{label}]({url})\n"
                        
                        answer_with_sources = answer + source_md
                    else:
                        answer_with_sources = answer

                else:
                    # OpenAI path
                    qa = RetrievalQA.from_chain_type(
                        llm=ChatOpenAI(model=openai_model, temperature=0),
                        retriever=st.session_state.vectordb.as_retriever(),
                        return_source_documents=True
                    )
                    result = qa.invoke({"query":question})
                    answer = result["result"]
                    source_docs = result.get("source_documents", [])

                    if source_docs:
                        # Extract unique sources with time refs
                        sources = {}
                        for doc in source_docs:
                            meta = doc.metadata
                            url = meta.get("source_url", "")
                            title = meta.get("source_title", "Video")
                            start = meta.get("start_time", None)
                            end = meta.get("end_time", None)
                            if start is not None and end is not None:
                                time_ref = f"{format_time(start)} - {format_time(end)}"
                                label = f"{title} [{time_ref}]"
                            else:
                                label = title
                            sources[url] = label

                        source_md = "\n\n---\n**Sources:**\n"
                        for url, label in sources.items():
                            source_md += f"- [{label}]({url})\n"

                        answer_with_sources = answer + source_md
                    else:
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