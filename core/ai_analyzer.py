"""
ai_analyzer.py

Uses Groq's LLaMA 3.3 70B to intelligently select the 2–3 best clip windows
from the transcript + energy candidate list, and generates:
  - A punchy "hook" headline for each clip (for the caption overlay)
  - A short description of why this moment is viral-worthy
  - Refined start/end timestamps aligned to sentence boundaries
"""

import os
import json
import logging
from typing import List, Dict, Tuple, Any

from groq import Groq

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert video editor and social media strategist for
the Creator Economy. You specialise in identifying "golden nugget" moments from
long-form educational/mentorship videos that will perform well on TikTok, Instagram
Reels, and YouTube Shorts.

Given a transcript excerpt and time-tagged candidate windows, you must:
1. Pick the 2–3 BEST moments that are self-contained, emotionally engaging, and
   make sense without needing context from the rest of the video.
2. For each moment, write a SHORT catchy hook headline (max 8 words) that would
   make someone stop scrolling.
3. Refine the timestamps to start/end at natural sentence boundaries (don't cut
   mid-sentence).
4. Explain briefly WHY this moment is viral-worthy.

Respond ONLY with valid JSON. No preamble, no markdown backticks. Just JSON."""

USER_TEMPLATE = """Here is the full transcript:

{full_transcript}

Here are the candidate time windows found by audio energy analysis
(format: [start_seconds, end_seconds, energy_score]):
{candidates}

Total video duration: {total_duration} seconds.

Return a JSON array with 2–3 objects, each having these exact keys:
- "start":       float (refined start time in seconds)
- "end":         float (refined end time in seconds)
- "hook":        string (catchy headline, max 8 words, ALL CAPS style)
- "why_viral":   string (1 sentence — why this will stop the scroll)
- "clip_title":  string (short descriptive title for the file name, snake_case)
"""


def select_best_clips(
    transcript: Dict[str, Any],
    candidates: List[Tuple[float, float, float]],
    total_duration: float,
    max_clips: int = 3,
) -> List[Dict]:
    """
    Ask the LLM to pick the best clips from candidates + transcript.

    Args:
        transcript:      Output from transcriber.transcribe_audio()
        candidates:      List of (start, end, score) from audio_analyzer
        total_duration:  Total video length in seconds
        max_clips:       Max number of clips to produce (2 or 3)

    Returns:
        List of clip dicts with keys: start, end, hook, why_viral, clip_title
    """
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    # Build a condensed transcript string (keep it under token limit)
    full_text = transcript.get("text", "").strip()

    # If transcript is very long, summarise per-segment instead
    if len(full_text) > 12000:
        seg_lines = []
        for seg in transcript.get("segments", []):
            seg_lines.append(f"[{seg['start']:.1f}s – {seg['end']:.1f}s]: {seg['text']}")
        full_text = "\n".join(seg_lines)

    cand_json = json.dumps(
        [[round(s, 1), round(e, 1), round(sc, 3)] for s, e, sc in candidates],
        indent=2,
    )

    user_msg = USER_TEMPLATE.format(
        full_transcript=full_text,
        candidates=cand_json,
        total_duration=round(total_duration, 1),
    )

    logger.info("Asking LLaMA 3.3 70B to select best clips…")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.4,
        max_tokens=1024,
    )

    raw = response.choices[0].message.content.strip()

    # Clean up in case model wraps in ```json
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        clips = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed: {e}\nRaw response:\n{raw}")
        # Graceful fallback: use top N candidates with generic hooks
        clips = _fallback_clips(candidates, max_clips)

    # Validate and sanitise
    valid_clips = []
    for clip in clips[:max_clips]:
        start = float(clip.get("start", 0))
        end = float(clip.get("end", start + 60))
        hook = str(clip.get("hook", "WATCH THIS MOMENT")).upper()
        title = str(clip.get("clip_title", f"clip_{len(valid_clips)+1}"))
        why = str(clip.get("why_viral", ""))

        # Clamp to video boundaries
        start = max(0.0, min(start, total_duration - 10))
        end = min(total_duration, max(end, start + 10))

        # Enforce max 90-second clips
        if end - start > 90:
            end = start + 90

        valid_clips.append({
            "start":      round(start, 2),
            "end":        round(end, 2),
            "hook":       hook,
            "why_viral":  why,
            "clip_title": title.replace(" ", "_").lower()[:40],
        })

    logger.info(f"AI selected {len(valid_clips)} clips:")
    for i, c in enumerate(valid_clips):
        logger.info(f'  [{i+1}] {c["start"]}s – {c["end"]}s | "{c["hook"]}"')

    return valid_clips


def _fallback_clips(
    candidates: List[Tuple[float, float, float]], max_clips: int
) -> List[Dict]:
    """Return a safe fallback if LLM fails to parse."""
    hooks = [
        "THIS CHANGES EVERYTHING",
        "THE TRUTH NOBODY TELLS YOU",
        "MOST PEOPLE GET THIS WRONG",
    ]
    clips = []
    for i, (s, e, _) in enumerate(candidates[:max_clips]):
        clips.append({
            "start":      s,
            "end":        e,
            "hook":       hooks[i % len(hooks)],
            "why_viral":  "High-energy audio moment.",
            "clip_title": f"clip_{i+1}",
        })
    return clips
