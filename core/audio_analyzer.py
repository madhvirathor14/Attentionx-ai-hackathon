"""
face_tracker.py

Samples frames from a video segment and returns a smoothed trajectory of
the speaker's face X-position (normalized 0–1).  Used by video_processor
to keep the speaker centred in the 9:16 crop.

We sample every SAMPLE_INTERVAL frames instead of every single frame to
keep processing fast — then interpolate between sampled positions.
"""

import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger(__name__)

SAMPLE_INTERVAL = 12   # Detect face every 12 frames (~0.4s at 30fps)


def build_face_trajectory(
    video_path: str,
    start_sec: float,
    end_sec: float,
    fps: Optional[float] = None,
) -> Dict[int, float]:
    """
    Build a frame_number → normalized_face_cx mapping for the given segment.

    - Samples every SAMPLE_INTERVAL frames with MediaPipe
    - Fills gaps with linear interpolation
    - Missing detections inherit the nearest known position

    Returns: dict mapping absolute frame numbers to face center X (0–1 range).
    """
    cap = cv2.VideoCapture(video_path)
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    start_frame = int(start_sec * fps)
    end_frame   = int(end_sec   * fps)

    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1,          # Full-range model (handles distant faces)
        min_detection_confidence=0.35,
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    sampled: Dict[int, float] = {}   # frame → cx
    last_cx = 0.5                    # default to center

    frame_idx = start_frame
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        should_sample = ((frame_idx - start_frame) % SAMPLE_INTERVAL == 0)

        if should_sample:
            cx = _detect_face_cx(frame, detector)
            if cx is not None:
                last_cx = cx
            sampled[frame_idx] = last_cx

        frame_idx += 1

    cap.release()
    detector.close()

    if not sampled:
        logger.warning("No faces detected in segment — defaulting to center crop.")
        return {f: 0.5 for f in range(start_frame, end_frame)}

    # ── Interpolate to fill every frame ─────────────────────────────────────
    full_traj = _interpolate(sampled, start_frame, end_frame)

    # ── Smooth with a Gaussian window to prevent jitter ─────────────────────
    frame_keys = sorted(full_traj.keys())
    cx_values  = np.array([full_traj[k] for k in frame_keys], dtype=float)

    from scipy.ndimage import gaussian_filter1d
    cx_smooth = gaussian_filter1d(cx_values, sigma=fps * 0.8)   # 0.8s smoothing

    smoothed = {k: float(cx_smooth[i]) for i, k in enumerate(frame_keys)}
    return smoothed


def _detect_face_cx(frame: np.ndarray, detector) -> Optional[float]:
    """
    Run MediaPipe on a single BGR frame.
    Returns normalized X of the face bounding-box center, or None.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    if not results.detections:
        return None

    # Pick the highest-confidence detection
    best = max(results.detections, key=lambda d: d.score[0])
    bbox = best.location_data.relative_bounding_box

    cx = bbox.xmin + bbox.width  / 2.0
    # clamp to valid range
    return float(np.clip(cx, 0.05, 0.95))


def _interpolate(
    sampled: Dict[int, float],
    start: int,
    end: int,
) -> Dict[int, float]:
    """Linear interpolation between sampled face positions."""
    keys   = sorted(sampled.keys())
    result: Dict[int, float] = {}

    for f in range(start, end):
        if f in sampled:
            result[f] = sampled[f]
            continue

        # Find surrounding keyframes
        prev_k = next((k for k in reversed(keys) if k <= f), None)
        next_k = next((k for k in keys if k >= f),          None)

        if prev_k is None:
            result[f] = sampled[next_k]
        elif next_k is None:
            result[f] = sampled[prev_k]
        elif prev_k == next_k:
            result[f] = sampled[prev_k]
        else:
            t = (f - prev_k) / (next_k - prev_k)
            result[f] = sampled[prev_k] + t * (sampled[next_k] - sampled[prev_k])

    return result
