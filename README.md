# Saaransh-AI

An end-to-end AI system that converts YouTube videos into structured knowledge and enables conversational interaction using Retrieval-Augmented Generation (RAG). The system performs audio extraction, transcription, summarization, insight extraction, and semantic question answering over video content.

---

## Features

- YouTube video processing via URL input
- Audio extraction using yt-dlp
- Audio conversion and preprocessing using FFmpeg
- Speech-to-text using Whisper (English)
- Speech-to-text using Sarvam AI (Hinglish / multilingual support)
- AI-based summarization using Mistral AI
- Automatic title generation
- Extraction of action items, key decisions, and questions
- Vector database using ChromaDB
- Semantic search using HuggingFace embeddings
- Retrieval-Augmented Generation (RAG) for Q&A
- Interactive chat interface using Streamlit

---

## System Architecture

The system is built as a modular AI pipeline that transforms video content into structured intelligence.

### High-Level Flow

- User provides a YouTube video URL through Streamlit UI
- Audio is extracted from the video using yt-dlp
- Audio is converted into WAV format using FFmpeg
- Audio is split into smaller chunks for processing
- Speech-to-text models generate a full transcript
- LLM processes transcript into structured insights
- Transcript is stored in a vector database for retrieval
- RAG system enables conversational Q&A over video content

---

## Detailed System Workflow

### 1. User Interface Layer
- Built using Streamlit
- Accepts YouTube URL input
- Provides language selection (English / Hinglish)
- Displays summary, insights, and chat interface
- Maintains session state for chat history

---

### 2. Audio Processing Layer
- Downloads audio from YouTube using yt-dlp
- Converts audio into WAV format using FFmpeg
- Normalizes audio (stereo to mono conversion)
- Splits audio into smaller chunks for processing
- Ensures scalability for long videos

---

### 3. Transcription Layer

#### Whisper (English Mode)
- Local speech-to-text model
- Optimized for English transcription
- Stable and deterministic output

#### Sarvam AI (Hinglish Mode)
- API-based transcription system
- Supports multilingual and Hinglish content
- Processes audio in smaller segments for accuracy

---

### 4. AI Processing Layer (Mistral AI)
- Generates concise video summary
- Produces meaningful title
- Extracts:
  - Action items (task, owner, deadline)
  - Key decisions
  - Open questions
- Converts unstructured transcript into structured knowledge

---

### 5. Vector Store Layer (ChromaDB)
- Transcript is split into semantic chunks
- Each chunk is converted into embeddings using HuggingFace model (all-MiniLM-L6-v2)
- Embeddings are stored in ChromaDB
- Enables fast semantic similarity search

---

### 6. RAG (Retrieval-Augmented Generation) Layer
- User query is embedded and matched with stored chunks
- Top relevant chunks are retrieved
- Context is injected into LLM prompt
- Mistral AI generates grounded response
- Ensures answers are based only on transcript data

---

### 7. Chat Interface Layer
- Enables real-time Q&A with video content
- Uses Streamlit chat components
- Maintains conversation history
- Continuously uses RAG pipeline for responses

---

## End-to-End Workflow

1. User submits YouTube URL
2. System extracts audio using yt-dlp
3. Audio is converted and chunked
4. Speech-to-text is performed using:
   - Whisper (English)
   - Sarvam AI (Hinglish)
5. Full transcript is generated
6. Mistral AI processes transcript:
   - Summary generation
   - Title generation
   - Action items extraction
   - Key decisions extraction
   - Question extraction
7. Transcript is split and embedded
8. ChromaDB stores embeddings
9. User asks questions in chat
10. Relevant context is retrieved
11. LLM generates final response

---

## Project Structure
```text
Saaransh-AI/
│
├── app.py
│
├── core/
│   ├── extractor.py
│   ├── rag_engine.py
│   ├── summarizer.py
│   ├── transcriber.py
│   ├── vector_store.py
│
├── utils/
│   ├── audio_processor.py
│
├── downloads/
├── vector_db/
│
├── .env
├── requirements.txt
└── README.md
```
---

## Installation

### 1. Clone Repository

git clone https://github.com/PiyushKumar74110/Saaransh-AI.git

cd Saaransh-AI

---

### 2. Create Virtual Environment

python -m venv venv

Activate:

Windows:
venv\Scripts\activate

macOS/Linux:
source venv/bin/activate

---

### 3. Install Dependencies

pip install -r requirements.txt

---

### 4. Install FFmpeg

Ubuntu:
sudo apt install ffmpeg

macOS:
brew install ffmpeg

Windows:
Install FFmpeg and add it to PATH

---

## Environment Variables

Create a `.env` file:

MISTRAL_API_KEY=your_api_key

SARVAM_API_KEY=your_api_key

WHISPER_MODEL=base

SARVAM_STT_MODEL=saaras:v2.5

---

## Running the Application

streamlit run app.py

Open:

http://localhost:8501

---

## Output Modules

- Video summary
- Generated title
- Action items with ownership and deadlines
- Key decisions
- Open questions
- Chat-based Q&A system

---

## Tech Stack

### Frontend
- Streamlit (UI and interactive web application)

### Backend
- Python (core application logic)
- LangChain (LLM orchestration and RAG pipeline)

### AI / Machine Learning Models
- Mistral AI (mistral-small-latest) for summarization, reasoning, and extraction
- OpenAI Whisper (speech-to-text for English transcription)
- Sarvam AI STT API (Hinglish and multilingual transcription)

### Retrieval-Augmented Generation (RAG)
- ChromaDB (vector database for embeddings storage and retrieval)
- HuggingFace Sentence Transformers (all-MiniLM-L6-v2 for embeddings)

### Audio Processing
- yt-dlp (YouTube audio extraction)
- FFmpeg (audio conversion to WAV format)
- SoundFile (audio reading and writing)
- NumPy (audio processing and chunk manipulation)

### NLP & Text Processing
- LangChain Text Splitters (RecursiveCharacterTextSplitter)
- Prompt engineering using ChatPromptTemplate
- Output parsing using StrOutputParser

### APIs & Services
- Sarvam AI API (speech-to-text service)
- Mistral AI API (LLM inference)

### Environment & Configuration
- python-dotenv (environment variable management)

### Development Tools
- Git & GitHub (version control)
- Virtual Environment (dependency isolation)

## Future Improvements

- Local video upload support
- Timestamp-based responses
- Speaker diarization
- Export to PDF or Notion
- Parallel transcription optimization
- Cloud deployment using Docker or Hugging Face Spaces

---

## Project Highlights

- End-to-end YouTube to AI knowledge pipeline
- Multi-model architecture (Whisper + Sarvam + Mistral)
- Real Retrieval-Augmented Generation system
- Scalable audio processing pipeline
- Interactive chat with video content

---

## License



---

## Author

Built using Python, Streamlit, LangChain, Mistral AI, Whisper, Sarvam AI, ChromaDB.
