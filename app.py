"""
app.py — AttentionX: AI Content Repurposing Engine
===================================================
Upload a long-form video → get 2–3 ready-to-post vertical short clips
with karaoke captions and AI-generated hook headlines.
"""

import os
import sys
import logging
import tempfile
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="AttentionX — AI Content Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    .stApp { background: #080810; color: #e4e4f0; }
    section[data-testid="stSidebar"] { background: #0e0e1a; }

    /* ── Typography ── */
    h1, h2, h3 { font-family: 'Inter', 'Segoe UI', sans-serif; }

    /* ── Hero header ── */
    .hero { text-align: center; padding: 2.5rem 1rem 1rem; }
    .hero-title {
        font-size: clamp(2.2rem, 5vw, 3.8rem);
        font-weight: 900;
        background: linear-gradient(135deg, #818cf8, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
        line-height: 1.1;
    }
    .hero-sub {
        color: #6b7280;
        font-size: 1.05rem;
        margin-top: 0.6rem;
    }

    /* ── Cards ── */
    .card {
        background: #0f0f1e;
        border: 1px solid #1e1e35;
        border-radius: 14px;
        padding: 1.5rem 1.6rem;
        margin-bottom: 1rem;
    }
    .card-highlight {
        background: #0d0d22;
        border: 1px solid #4f46e5;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }

    /* ── Pill badges ── */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.8rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
    }
    .badge-purple { background: #3730a3; color: #c7d2fe; }
    .badge-green  { background: #064e3b; color: #6ee7b7; }
    .badge-pink   { background: #831843; color: #fbcfe8; }

    /* ── Step row ── */
    .step-row {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 0.75rem;
    }
    .step-dot {
        min-width: 28px; height: 28px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        font-weight: 700;
        font-size: 0.8rem;
        display: flex; align-items: center; justify-content: center;
        margin-top: 2px;
    }

    /* ── Upload zone ── */
    div[data-testid="stFileUploader"] > div {
        background: #0d0d22;
        border: 2px dashed #4f46e5 !important;
        border-radius: 14px !important;
    }

    /* ── Primary button ── */
    div.stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.8rem 1.6rem !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        width: 100% !important;
        letter-spacing: 0.3px;
        transition: opacity 0.15s !important;
    }
    div.stButton > button:hover { opacity: 0.88 !important; }
    div.stButton > button:disabled { opacity: 0.4 !important; }

    /* ── Progress text ── */
    .status-text { color: #a5b4fc; font-size: 0.92rem; margin: 0.4rem 0; }

    /* ── Clip result card ── */
    .clip-card {
        background: #0f0f1e;
        border: 1px solid #312e81;
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
    }
    .hook-text {
        font-size: 1.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #818cf8, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
    }

    /* ── Sidebar info ── */
    .sb-section {
        background: #13131f;
        border: 1px solid #1e1e35;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        font-size: 0.88rem;
        color: #9ca3af;
    }
    .sb-section b { color: #e4e4f0; }

    /* ── Misc ── */
    hr.divider { border-color: #1e1e35; }
    .warn-box {
        background: #422006; border: 1px solid #d97706;
        border-radius: 10px; padding: 0.8rem 1rem;
        color: #fcd34d; font-size: 0.88rem; margin-bottom: 0.8rem;
    }
    .success-box {
        background: #052e16; border: 1px solid #16a34a;
        border-radius: 10px; padding: 0.8rem 1rem;
        color: #86efac; font-size: 0.88rem; margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Lazy imports (only after package install) ────────────────────────────────

def _check_groq_key() -> bool:
    return bool(os.environ.get("GROQ_API_KEY", "").strip())


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ AttentionX")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-section">
        <b>What this does</b><br><br>
        Upload any long video → AttentionX automatically:<br><br>
        🎯 &nbsp;Finds high-energy "golden nugget" moments<br>
        📱 &nbsp;Crops to TikTok/Reels 9:16 with face tracking<br>
        💬 &nbsp;Adds karaoke word-by-word captions<br>
        🔥 &nbsp;Generates viral hook headlines with AI
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-section">
        <b>Setup — API Key</b><br><br>
        1. Get a free key at <code>console.groq.com</code><br>
        2. Create a <code>.env</code> file in the project root<br>
        3. Add: <code>GROQ_API_KEY=your_key</code>
    </div>
    """, unsafe_allow_html=True)

    # Live API key check
    if _check_groq_key():
        st.markdown('<div class="success-box">✅ &nbsp;Groq API key loaded</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn-box">⚠️ &nbsp;GROQ_API_KEY not found — add it to .env</div>',
                    unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-section">
        <b>Evaluation criteria</b><br><br>
        🏆 UX (25%) · Impact (20%)<br>
        🔧 Tech (20%) · Innovation (20%)<br>
        🎬 Presentation (15%)
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="color:#374151;font-size:0.75rem;text-align:center;margin-top:1rem;">AttentionX · AttentionX AI Hackathon</p>', unsafe_allow_html=True)


# ─── Hero ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="hero-title">⚡ AttentionX</div>
    <div class="hero-sub">
        Turn long-form videos into scroll-stopping short clips — automatically.<br>
        <span style="color:#4f46e5">AI-powered</span> · Face-tracked 9:16 crop · Karaoke captions · Hook headlines
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── How It Works ─────────────────────────────────────────────────────────────

with st.expander("📖  How it works", expanded=False):
    cols = st.columns(3)
    steps = [
        ("🎧", "Audio Energy Analysis",
         "Librosa scans the audio track and finds where you speak with the most "
         "passion — RMS energy + spectral rolloff combined."),
        ("🧠", "AI Clip Selection",
         "Groq Whisper transcribes the video. Then LLaMA 3.3 70B reads the full "
         "transcript and picks the 2–3 moments most likely to go viral."),
        ("🎬", "Smart Video Export",
         "MediaPipe tracks your face frame-by-frame. The crop stays centred on "
         "you as you move. Karaoke captions and the hook headline are burned in."),
    ]
    for col, (icon, title, body) in zip(cols, steps):
        col.markdown(f"""
        <div class="card" style="height:180px;">
            <div style="font-size:2rem;margin-bottom:0.5rem">{icon}</div>
            <b style="color:#e4e4f0">{title}</b>
            <p style="color:#6b7280;font-size:0.87rem;margin-top:0.4rem">{body}</p>
        </div>
        """, unsafe_allow_html=True)


# ─── Upload + Settings ───────────────────────────────────────────────────────

col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown("### 📤  Upload Your Video")
    uploaded = st.file_uploader(
        "Drop your video here (MP4, MOV, AVI, MKV — up to 500 MB)",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        label_visibility="collapsed",
    )
    if uploaded:
        st.markdown(f"""
        <div class="success-box">
            ✅ &nbsp;<b>{uploaded.name}</b> &nbsp;·&nbsp;
            {uploaded.size / 1_000_000:.1f} MB
        </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown("### ⚙️  Settings")

    num_clips = st.slider(
        "Number of clips to generate",
        min_value=1, max_value=3, value=2,
        help="More clips = longer processing time."
    )

    min_clip_len = st.slider(
        "Minimum clip length (seconds)",
        min_value=20, max_value=90, value=50, step=5,
        help="Each clip will be at least this long."
    )

    add_hook = st.toggle("Show hook headline on clips", value=True)
    show_captions = st.toggle("Add karaoke captions", value=True)

    st.markdown('<div class="sb-section" style="margin-top:0.5rem">💡 <b>Tip:</b> '
                '45–60s clips perform best on Reels & TikTok.</div>',
                unsafe_allow_html=True)


# ─── Process Button ───────────────────────────────────────────────────────────

st.markdown("---")
can_run = uploaded is not None and _check_groq_key()

if not _check_groq_key():
    st.markdown('<div class="warn-box">⚠️ Add your <b>GROQ_API_KEY</b> to .env before processing.</div>',
                unsafe_allow_html=True)

run_btn = st.button(
    "⚡ Generate Short Clips",
    disabled=not can_run,
    use_container_width=True,
)


# ─── Processing Pipeline ──────────────────────────────────────────────────────

if run_btn and can_run:
    import utils.helpers as helpers
    from core.audio_analyzer import find_emotional_peaks
    from core.transcriber import transcribe_audio
    from core.ai_analyzer import select_best_clips
    from core.face_tracker import build_face_trajectory
    from core.video_processor import process_clip

    helpers.setup_logging()
    logger = logging.getLogger("app")

    output_dir = "outputs"
    helpers.ensure_dir(output_dir)

    with tempfile.TemporaryDirectory() as tmp_dir:

        # ── Save uploaded file ───────────────────────────────────────────────
        video_tmp = os.path.join(tmp_dir, uploaded.name)
        with open(video_tmp, "wb") as f:
            f.write(uploaded.getbuffer())

        total_dur = helpers.get_video_duration(video_tmp)
        fps_val   = helpers.get_video_fps(video_tmp)

        st.markdown(f"""
        <div class="card-highlight">
            📹 &nbsp;<b>{uploaded.name}</b> &nbsp;·&nbsp;
            Duration: <b>{helpers.format_seconds(total_dur)}</b> &nbsp;·&nbsp;
            FPS: <b>{fps_val:.1f}</b>
        </div>
        """, unsafe_allow_html=True)

        # ── Step 1: Audio analysis ───────────────────────────────────────────
        with st.status("🎧  Analysing audio energy…", expanded=True) as status1:
            st.write("Extracting audio track…")
            audio_path = helpers.extract_audio(video_tmp, tmp_dir)

            st.write("Running Librosa energy analysis…")
            candidates = find_emotional_peaks(
                audio_path,
                min_duration=float(min_clip_len),
                top_n=num_clips + 3,
            )
            st.write(f"Found **{len(candidates)}** candidate windows ✓")
            status1.update(label="✅  Audio analysis complete", state="complete")

        # ── Step 2: Transcription ────────────────────────────────────────────
        with st.status("📝  Transcribing with Groq Whisper…", expanded=True) as status2:
            st.write("Sending audio to Groq Whisper Large v3…")
            transcript = transcribe_audio(audio_path)
            n_words = len(transcript["words"])
            st.write(f"Transcribed: **{len(transcript['text'])} chars**, **{n_words} words** ✓")
            status2.update(label="✅  Transcription complete", state="complete")

        # ── Step 3: AI clip selection ────────────────────────────────────────
        with st.status("🧠  Selecting best clips with LLaMA 3.3 70B…", expanded=True) as status3:
            st.write("Asking AI to identify viral moments…")
            selected_clips = select_best_clips(
                transcript,
                candidates,
                total_duration=total_dur,
                max_clips=num_clips,
            )
            for clip in selected_clips:
                st.write(f"🎯 `{helpers.format_seconds(clip['start'])}` → "
                         f"`{helpers.format_seconds(clip['end'])}` — *{clip['hook']}*")
            status3.update(label="✅  AI selection complete", state="complete")

        # ── Step 4: Video processing ─────────────────────────────────────────
        st.markdown("### 🎬  Processing clips…")
        result_paths = []

        for i, clip_info in enumerate(selected_clips):
            clip_label = f"Clip {i+1}: {clip_info['clip_title']}"

            with st.status(f"✂️  {clip_label}", expanded=True) as clip_status:
                start = clip_info["start"]
                end   = clip_info["end"]

                st.write("Building face trajectory…")
                face_traj = build_face_trajectory(
                    video_tmp, start, end, fps=fps_val
                )

                st.write("Rendering vertical video with captions…")
                prog_bar = st.progress(0.0)

                def on_progress(frac, _bar=prog_bar):
                    _bar.progress(min(frac, 1.0))

                safe_title = clip_info["clip_title"].replace("/", "_")
                out_path = os.path.join(output_dir, f"attentionx_clip{i+1}_{safe_title}.mp4")

                hook = clip_info["hook"] if add_hook else ""
                words_for_clip = transcript["words"] if show_captions else []

                process_clip(
                    video_path=video_tmp,
                    start_sec=start,
                    end_sec=end,
                    face_trajectory=face_traj,
                    words=words_for_clip,
                    hook_text=hook,
                    output_path=out_path,
                    progress_callback=on_progress,
                )

                prog_bar.progress(1.0)
                result_paths.append((clip_info, out_path))
                clip_status.update(label=f"✅  {clip_label} done!", state="complete")

        # ── Step 5: Show results ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 🎉  Your Clips Are Ready!")

        for i, (clip_info, path) in enumerate(result_paths):
            if not os.path.exists(path):
                continue

            file_size_mb = os.path.getsize(path) / 1_000_000

            st.markdown(f"""
            <div class="clip-card">
                <div class="hook-text">🔥 "{clip_info['hook']}"</div>
                <p style="color:#6b7280;font-size:0.88rem;margin:0.2rem 0 0.8rem">
                    {clip_info.get('why_viral','')}<br>
                    ⏱ {helpers.format_seconds(clip_info['start'])} →
                       {helpers.format_seconds(clip_info['end'])}
                    &nbsp;·&nbsp; 📁 {file_size_mb:.1f} MB
                </p>
            </div>
            """, unsafe_allow_html=True)

            col_v, col_d = st.columns([2, 1])
            with col_v:
                st.video(path)
            with col_d:
                with open(path, "rb") as f:
                    st.download_button(
                        label=f"⬇️ Download Clip {i+1}",
                        data=f,
                        file_name=Path(path).name,
                        mime="video/mp4",
                        key=f"dl_{i}",
                        use_container_width=True,
                    )
                st.markdown(f"""
                <div class="sb-section" style="margin-top:0.5rem;">
                    <b>Duration:</b> {helpers.format_seconds(clip_info['end'] - clip_info['start'])}<br>
                    <b>Format:</b> MP4 · 9:16<br>
                    <b>Captions:</b> {'✅' if show_captions else '❌'}<br>
                    <b>Hook:</b> {'✅' if add_hook else '❌'}
                </div>
                """, unsafe_allow_html=True)

        st.balloons()
        st.success("✅ All clips generated! Upload to TikTok, Instagram Reels, or YouTube Shorts.")
