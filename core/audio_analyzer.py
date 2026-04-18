"""
audio_analyzer.py

Finds emotionally intense / high-energy moments in a video's audio track.
Strategy: compute RMS energy + spectral rolloff, smooth the curve, then
find peaks that are at least `min_gap` seconds apart.

These candidates are passed to the AI step which picks the best ones by
actual content quality — not just loudness.
"""

import librosa
import numpy as np
from scipy.signal import find_peaks
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def find_emotional_peaks(
    audio_path: str,
    min_duration: float = 45.0,
    top_n: int = 6,
) -> List[Tuple[float, float, float]]:
    """
    Scan audio and return candidate time windows sorted by energy score.

    Args:
        audio_path:    Path to extracted WAV/MP3 file.
        min_duration:  Minimum clip length we want (seconds).
        top_n:         How many candidates to surface.

    Returns:
        List of (start_sec, end_sec, score) sorted best → worst.
    """
    logger.info("Loading audio for energy analysis…")
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    total_dur = len(y) / sr
    logger.info(f"Duration: {total_dur:.1f}s  |  SR: {sr} Hz")

    # ── Feature extraction ──────────────────────────────────────────────────
    # 0.5s frames, 0.25s hop → 4 frames/sec resolution
    frame_len = int(sr * 0.5)
    hop = int(sr * 0.25)

    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    def _norm(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-9)

    # Combined score: RMS is king, rolloff & ZCR add "speech passion" signal
    score = 0.65 * _norm(rms) + 0.20 * _norm(rolloff) + 0.15 * _norm(zcr)

    # Smooth over ~6s window to favour sustained energy over quick bursts
    smooth_win = max(1, int(6.0 / 0.25))
    score_smooth = np.convolve(score, np.ones(smooth_win) / smooth_win, mode="same")

    # ── Peak detection ───────────────────────────────────────────────────────
    min_gap_frames = int(min_duration / 0.25)
    threshold = np.percentile(score_smooth, 35)   # ignore bottom 35%

    peaks, _ = find_peaks(score_smooth, distance=min_gap_frames, height=threshold)

    # ── Build segments around each peak ─────────────────────────────────────
    half = min_duration / 2.0
    candidates: List[Tuple[float, float, float]] = []

    for p in peaks:
        t_peak = float(times[p])
        seg_start = max(0.0, t_peak - half)
        seg_end = min(total_dur, seg_start + min_duration)

        # Edge case: near end of video
        if seg_end >= total_dur - 3:
            seg_end = max(0, total_dur - 2)
            seg_start = max(0.0, seg_end - min_duration)

        if seg_end - seg_start < 15:
            continue

        seg_score = float(score_smooth[p])
        candidates.append((round(seg_start, 2), round(seg_end, 2), seg_score))

    # ── Fallback: divide into equal chunks if no peaks found ────────────────
    if not candidates:
        logger.warning("No clear peaks found — using uniform chunk fallback.")
        chunk = min_duration
        n = max(1, int(total_dur / chunk))
        for i in range(n):
            s = i * chunk
            e = min(total_dur, s + chunk)
            fi_s = int(s / 0.25)
            fi_e = min(int(e / 0.25), len(score_smooth))
            chunk_score = float(score_smooth[fi_s:fi_e].mean()) if fi_e > fi_s else 0.0
            candidates.append((round(s, 2), round(e, 2), chunk_score))

    # ── Deduplicate overlapping segments, keep highest score ────────────────
    candidates.sort(key=lambda x: x[2], reverse=True)
    kept: List[Tuple[float, float, float]] = []

    for cand in candidates:
        too_close = False
        for ex in kept:
            overlap_start = max(cand[0], ex[0])
            overlap_end = min(cand[1], ex[1])
            if overlap_end - overlap_start > 10:  # >10s overlap → skip
                too_close = True
                break
        if not too_close:
            kept.append(cand)
        if len(kept) >= top_n:
            break

    logger.info(f"Returning {len(kept)} candidate windows.")
    for i, (s, e, sc) in enumerate(kept):
        logger.info(f"  [{i+1}] {s:.1f}s → {e:.1f}s  score={sc:.3f}")

    return kept
