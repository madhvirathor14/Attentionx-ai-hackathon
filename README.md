# ⚡ AttentionX — AI Content Repurposing Engine

> **AttentionX AI Hackathon · UnsaidTalks Education**

Turn a 60-minute lecture or podcast into **2–3 scroll-stopping vertical clips** in minutes — fully automated, zero manual editing.

---

## 🎬 Demo Video

<!-- ✅ MANDATORY: Replace this link with your actual screen recording before submission -->
**📽️ [Watch the Live Demo on Google Drive](https://drive.google.com/your-demo-link-here)**

> *The demo shows: video upload → AI processing pipeline → vertical clip output with karaoke captions and hook headline.*

---

## 🔥 What It Does

| Feature | How |
|---------|-----|
| 🎧 **Emotional Peak Detection** | Librosa RMS energy + spectral rolloff finds where speakers are most animated |
| 🧠 **AI Clip Selection** | Groq LLaMA 3.3 70B reads the full transcript and picks the 2–3 moments most likely to go viral |
| 📱 **Smart 9:16 Crop** | MediaPipe face detection tracks the speaker frame-by-frame; they stay centred during motion |
| 💬 **Karaoke Captions** | Groq Whisper word-level timestamps power animated word-by-word captions |
| 🔥 **Hook Headlines** | AI-generated attention-grabbing headline shown at the top for the first 5 seconds |

---

## 🏗️ Architecture

```
📹 Video Upload
      │
      ▼
┌─────────────────────┐
│  Audio Extraction   │  MoviePy → WAV
└─────────┬───────────┘
          │
    ┌─────┴──────────────────────────────────────┐
    │                                            │
    ▼                                            ▼
┌──────────────────┐                  ┌─────────────────────┐
│  Audio Analyzer  │                  │  Groq Whisper STT   │
│  (Librosa)       │                  │  Word timestamps    │
│  RMS energy      │                  └──────────┬──────────┘
│  Spectral rolloff│                             │
│  Peak detection  │                             │
└────────┬─────────┘                             │
         │  Candidate windows                    │ Full transcript
         └──────────────┬────────────────────────┘
                        ▼
              ┌──────────────────┐
              │  AI Clip Picker  │
              │  Groq LLaMA 3.3  │
              │  70B Versatile   │
              │  + Hook headlines│
              └────────┬─────────┘
                       │  Selected clips (start, end, hook)
                       ▼
              ┌──────────────────┐
              │  Face Tracker    │
              │  MediaPipe       │
              │  + Smoothing     │
              └────────┬─────────┘
                       │  Face trajectory per frame
                       ▼
              ┌──────────────────┐
              │  Video Processor │
              │  OpenCV + PIL    │
              │  9:16 crop       │
              │  Karaoke overlay │
              │  MoviePy audio   │
              └────────┬─────────┘
                       │
                       ▼
              🎬 9:16 MP4 clips
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit | Beautiful Python-native UI |
| **Transcription** | Groq Whisper Large v3 | Fast, accurate STT with word timestamps |
| **AI Analysis** | Groq LLaMA 3.3 70B | Transcript analysis + hook generation |
| **Audio Analysis** | Librosa + SciPy | Energy peak detection |
| **Face Detection** | MediaPipe | Frame-accurate face tracking |
| **Video Processing** | OpenCV + MoviePy | Cropping, frame rendering, audio merge |
| **Caption Rendering** | Pillow (PIL) | Karaoke text overlays |

---

## 🚀 Setup & Run (Full Guide)

### Prerequisites

- Python 3.10 or 3.11
- A free [Groq API key](https://console.groq.com/keys) (takes 2 minutes)
- `ffmpeg` installed on your system

**Install ffmpeg:**
```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS (with Homebrew)
brew install ffmpeg

# Windows — download from https://ffmpeg.org/download.html
```

---

### Step 1 — Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/attentionx.git
cd attentionx
```

---

### Step 2 — Create a virtual environment

```bash
# Create venv
python -m venv venv

# Activate it
# Linux / macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> ⏳ This takes 2–3 minutes the first time (MediaPipe and OpenCV are large).

---

### Step 4 — Configure your API key

```bash
# Copy the template
cp .env.example .env

# Open .env in any editor and add your Groq key:
# GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Get a free Groq key at → **https://console.groq.com/keys**

> **⚠️ Never push your `.env` file to GitHub!**  
> It's already in `.gitignore` — keep it that way.

---

### Step 5 — Run the app

```bash
streamlit run app.py
```

The app will open in your browser at **http://localhost:8501**

---

## 📋 How to Use

1. **Upload** your long-form video (MP4, MOV, MKV — up to 500 MB)
2. **Adjust settings** — number of clips (1–3), minimum clip length, captions on/off
3. **Click "⚡ Generate Short Clips"**
4. **Wait** ~3–8 minutes (depends on video length)
5. **Preview & download** your vertical clips

---

## 📁 Project Structure

```
attentionx/
├── app.py                  # Streamlit frontend (main entry point)
├── requirements.txt        # Python dependencies
├── .env.example            # API key template (safe to commit)
├── .gitignore
├── README.md
│
├── core/                   # Processing pipeline
│   ├── __init__.py
│   ├── audio_analyzer.py   # Librosa energy peak detection
│   ├── transcriber.py      # Groq Whisper STT
│   ├── ai_analyzer.py      # Groq LLaMA clip selection
│   ├── face_tracker.py     # MediaPipe face tracking
│   └── video_processor.py  # OpenCV + MoviePy video export
│
├── utils/
│   ├── __init__.py
│   └── helpers.py          # File I/O, logging, audio extraction
│
└── outputs/                # Generated clips saved here (git-ignored)
```

---

## 🌐 Optional: Deploy to Streamlit Cloud (Free)

1. Push your code to a **public** GitHub repo
2. Go to → **https://share.streamlit.io**
3. Connect your GitHub repo → select `app.py`
4. In **Advanced settings → Secrets**, add:
   ```
   GROQ_API_KEY = "your_groq_key_here"
   ```
5. Deploy! You'll get a public URL like `attentionx.streamlit.app`

> Hosted projects get bonus points in evaluation 🎯

---

## 📤 GitHub Setup & Submission

```bash
# Initialize git (if not already)
git init

# Add all files (the .gitignore will exclude .env and outputs/)
git add .

# First commit
git commit -m "feat: AttentionX AI Content Repurposing Engine"

# Add your GitHub remote
git remote add origin https://github.com/YOUR_USERNAME/attentionx.git

# Push to main branch
git push -u origin main
```

**Submit your public GitHub repo URL** on the Unstop portal.

---

## 🧠 Key Design Decisions

**Why Groq instead of OpenAI?**  
Groq's Whisper Large v3 is ~10x faster than OpenAI's equivalent at the same accuracy. The LLaMA 3.3 70B model is free-tier friendly and handles long transcripts very well. One API key covers the entire pipeline.

**Why not process every frame for face detection?**  
MediaPipe on every frame would make a 60s clip take 30+ minutes to process. Sampling every 12 frames (~0.4s intervals) with Gaussian smoothing gives smooth, natural-looking tracking with a ~15x speedup.

**Why Pillow for captions instead of OpenCV putText?**  
PIL supports proper Unicode, anti-aliasing, and per-word colour control which OpenCV's putText doesn't. The quality difference is significant when text is the main engagement driver.

---

## 📊 Evaluation Criteria Addressed

| Criterion | How AttentionX addresses it |
|-----------|----------------------------|
| **Impact (20%)** | End-to-end pipeline: one upload → ready-to-post clips. Real value for creators. |
| **Innovation (20%)** | Combined audio energy + LLM content analysis for clip selection. Face-tracked vertical crop. Word-highlighted karaoke captions. |
| **Technical Execution (20%)** | Clean modular architecture, proper error handling, typed function signatures, logging throughout. |
| **User Experience (25%)** | Custom dark-theme Streamlit UI, real-time progress, in-app preview, one-click download. |
| **Presentation (15%)** | Demo video included above. |

---

## 🙏 Built With

- [Groq](https://groq.com) — Whisper + LLaMA inference
- [MediaPipe](https://mediapipe.dev) — Face detection
- [Librosa](https://librosa.org) — Audio analysis
- [MoviePy](https://zulko.github.io/moviepy/) — Video I/O
- [Streamlit](https://streamlit.io) — Frontend

---

*AttentionX — Built for the AttentionX AI Hackathon by UnsaidTalks Education*
