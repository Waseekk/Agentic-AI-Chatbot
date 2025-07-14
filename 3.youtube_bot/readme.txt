YouTube QA Bot
A Streamlit-powered app that lets you index YouTube videos by fetching their transcripts (captions or Whisper transcription) and interactively ask questions about the video content using vector search and large language models (OpenAI GPT or Groq LLM).

Features
Extract transcripts from YouTube videos via:

Official captions (if available)

Audio transcription using OpenAI Whisper (fallback)

Split transcripts into manageable chunks and embed them using OpenAI embeddings

Store vectors in an in-memory FAISS vector store (optionally configurable for Pinecone)

Interactive chat interface to ask questions referencing indexed videos

Source attribution with video titles and URLs for OpenAI answers

Support for multiple languages in transcript fetching

Choose between OpenAI or Groq LLM providers and models for Q&A

Streamlit sidebar for easy configuration and control

Demo

Example of the chat interface interacting with indexed YouTube videos

Getting Started
Prerequisites
Python 3.8 or newer

An OpenAI API key (required)

Optional: Groq API key for Groq LLM support

Optional: Pinecone API key & environment for persistent vector storage

Installation
bash
Copy
git clone https://github.com/yourusername/youtube-qa-bot.git
cd youtube-qa-bot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Environment Variables
Create a .env file in the project root with the following keys:

env
Copy
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key          
PINECONE_API_KEY=your_pinecone_api_key  # optional
PINECONE_ENV=your_pinecone_env          # optional
TRAVILY_API_KEY=your_travily_api_key    # optional
Running the App
bash
Copy
streamlit run app.py
Then open your browser at http://localhost:8501.

Usage
Paste a YouTube video URL into the "Index a YouTube Video" input.

Click "Index Video" to fetch and embed the transcript.

Ask questions in the chat input about the indexed videos.

View answers with sources referenced.