import os
import logging
from pathlib import Path
from typing import List, Dict, Any
from groq import Groq

logger = logging.getLogger(__name__)


def transcribe_audio(audio_path: str) -> Dict[str, Any]:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    # Compress WAV to small MP3 to stay under Groq's 25MB limit
    mp3_path = audio_path.replace(".wav", ".mp3")
    try:
        import subprocess
        subprocess.run([
            "ffmpeg", "-i", audio_path,
            "-ar", "16000", "-ac", "1", "-b:a", "32k",
            "-y", mp3_path
        ], capture_output=True, check=True)
        send_path = mp3_path
        logger.info(f"Compressed: {Path(mp3_path).stat().st_size/1e6:.1f}MB")
    except Exception as e:
        logger.warning(f"Compression failed, using original: {e}")
        send_path = audio_path

    logger.info(f"Sending {Path(send_path).stat().st_size/1e6:.1f}MB to Groq Whisper")

    with open(send_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=(Path(send_path).name, f),
            model="whisper-large-v3",
            response_format="verbose_json",
            timestamp_granularities=["segment", "word"],
            temperature=0.0,
        )

    full_text = response.text or ""
    segments: List[Dict] = []
    words: List[Dict] = []

    if hasattr(response, "segments") and response.segments:
        for seg in response.segments:
            segments.append({
                "start": float(seg.get("start", 0)),
                "end":   float(seg.get("end", 0)),
                "text":  seg.get("text", "").strip(),
            })

    if hasattr(response, "words") and response.words:
        for w in response.words:
            words.append({
                "word":  w.get("word", "").strip(),
                "start": float(w.get("start", 0)),
                "end":   float(w.get("end", 0)),
            })
    else:
        logger.warning("No word timestamps — interpolating from segments.")
        words = _interpolate_word_timestamps(segments)

    logger.info(f"Transcript: {len(full_text)} chars | {len(segments)} segs | {len(words)} words")
    return {"text": full_text, "segments": segments, "words": words}


def _interpolate_word_timestamps(segments: List[Dict]) -> List[Dict]:
    words = []
    for seg in segments:
        seg_words = seg["text"].strip().split()
        if not seg_words:
            continue
        dur = seg["end"] - seg["start"]
        tpw = dur / len(seg_words)
        for i, w in enumerate(seg_words):
            words.append({
                "word":  w,
                "start": round(seg["start"] + i * tpw, 3),
                "end":   round(seg["start"] + (i + 1) * tpw, 3),
            })
    return words
