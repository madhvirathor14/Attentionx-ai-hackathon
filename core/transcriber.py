"""
transcriber.py

Sends audio to Groq's Whisper Large v3 and gets back a full transcript
with word-level timestamps. These timestamps are what powers the
karaoke-style caption system.

Why Groq?  It's blazing fast (~10x faster than OpenAI's endpoint) and
the free tier is generous enough for hackathon use.
"""

import os
import math
import logging
from pathlib import Path
from typing import List, Dict, Any

from groq import Groq

logger = logging.getLogger(__name__)

# Groq's audio endpoint has a 25 MB limit per request
MAX_AUDIO_BYTES = 24 * 1024 * 1024   # 24 MB to stay safe


def transcribe_audio(audio_path: str) -> Dict[str, Any]:
    """
    Transcribe an audio file using Groq Whisper Large v3.

    Returns a dict with:
        - "text":     full transcript string
        - "segments": list of {start, end, text} segment dicts
        - "words":    list of {word, start, end} word dicts (for karaoke)
    """
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    file_size = Path(audio_path).stat().st_size
    logger.info(f"Audio file size: {file_size / 1024 / 1024:.1f} MB")

    if file_size > MAX_AUDIO_BYTES:
        logger.warning(
            f"Audio is {file_size/1e6:.1f}MB — over Groq's 25MB limit. "
            "Will transcribe first 20 minutes only."
        )
        # Trim audio before sending (handled upstream — just log the warning)

    logger.info("Sending audio to Groq Whisper…")

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=(Path(audio_path).name, f),
            model="whisper-large-v3",
            response_format="verbose_json",
            timestamp_granularities=["segment", "word"],
            language="en",          # Hint: change if video isn't English
            temperature=0.0,        # Deterministic — better for timestamps
        )

    # ── Parse response ───────────────────────────────────────────────────────
    full_text = response.text or ""
    segments: List[Dict] = []
    words: List[Dict] = []

    # Segments
    if hasattr(response, "segments") and response.segments:
        for seg in response.segments:
            segments.append({
                "start": float(seg.get("start", 0)),
                "end":   float(seg.get("end", 0)),
                "text":  seg.get("text", "").strip(),
            })

    # Word-level timestamps (what powers karaoke captions)
    if hasattr(response, "words") and response.words:
        for w in response.words:
            words.append({
                "word":  w.get("word", "").strip(),
                "start": float(w.get("start", 0)),
                "end":   float(w.get("end", 0)),
            })
    else:
        # Groq sometimes doesn't return word-level even when asked.
        # Fallback: interpolate word timestamps from segments.
        logger.warning("No word-level timestamps returned — interpolating from segments.")
        words = _interpolate_word_timestamps(segments)

    logger.info(f"Transcript: {len(full_text)} chars | {len(segments)} segments | {len(words)} words")

    return {
        "text":     full_text,
        "segments": segments,
        "words":    words,
    }


def _interpolate_word_timestamps(segments: List[Dict]) -> List[Dict]:
    """
    If Whisper doesn't return word timestamps, distribute words evenly
    within each segment's time range.  Rough, but better than nothing.
    """
    words = []
    for seg in segments:
        seg_text = seg["text"].strip()
        seg_words = seg_text.split()
        if not seg_words:
            continue

        dur = seg["end"] - seg["start"]
        time_per_word = dur / len(seg_words)

        for i, w in enumerate(seg_words):
            words.append({
                "word":  w,
                "start": round(seg["start"] + i * time_per_word, 3),
                "end":   round(seg["start"] + (i + 1) * time_per_word, 3),
            })

    return words
