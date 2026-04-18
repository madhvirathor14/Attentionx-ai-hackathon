import librosa
import numpy as np
from scipy.signal import find_peaks
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def find_emotional_peaks(audio_path, min_duration=45.0, top_n=6):
    logger.info("Loading audio...")
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    total_dur = len(y) / sr

    frame_len = int(sr * 0.5)
    hop = int(sr * 0.25)

    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    def _norm(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-9)

    score = 0.65 * _norm(rms) + 0.20 * _norm(rolloff) + 0.15 * _norm(zcr)
    smooth_win = max(1, int(6.0 / 0.25))
    score_smooth = np.convolve(score, np.ones(smooth_win) / smooth_win, mode="same")

    min_gap_frames = int(min_duration / 0.25)
    threshold = np.percentile(score_smooth, 35)
    peaks, _ = find_peaks(score_smooth, distance=min_gap_frames, height=threshold)

    half = min_duration / 2.0
    candidates = []

    for p in peaks:
        t_peak = float(times[p])
        seg_start = max(0.0, t_peak - half)
        seg_end = min(total_dur, seg_start + min_duration)
        if seg_end >= total_dur - 3:
            seg_end = max(0, total_dur - 2)
            seg_start = max(0.0, seg_end - min_duration)
        if seg_end - seg_start < 15:
            continue
        candidates.append((round(seg_start, 2), round(seg_end, 2), float(score_smooth[p])))

    if not candidates:
        chunk = min_duration
        n = max(1, int(total_dur / chunk))
        for i in range(n):
            s = i * chunk
            e = min(total_dur, s + chunk)
            fi_s = int(s / 0.25)
            fi_e = min(int(e / 0.25), len(score_smooth))
            sc = float(score_smooth[fi_s:fi_e].mean()) if fi_e > fi_s else 0.0
            candidates.append((round(s, 2), round(e, 2), sc))

    candidates.sort(key=lambda x: x[2], reverse=True)
    kept = []

    for cand in candidates:
        too_close = False
        for ex in kept:
            if min(cand[1], ex[1]) - max(cand[0], ex[0]) > 10:
                too_close = True
                break
        if not too_close:
            kept.append(cand)
        if len(kept) >= top_n:
            break

    return kept
