"""
helpers.py

File I/O utilities, temp directory management, audio extraction,
and simple logging setup for AttentionX.
"""

import os
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Optional


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a clean console logger."""
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def extract_audio(video_path: str, output_dir: Optional[str] = None) -> str:
    """
    Extract audio from a video file and save as a WAV.
    Uses moviepy so no separate ffmpeg binary is needed.

    Args:
        video_path:  Source video file path.
        output_dir:  Where to save the WAV (defaults to system temp).

    Returns:
        Path to extracted WAV file.
    """
    from moviepy.editor import VideoFileClip

    if output_dir is None:
        output_dir = tempfile.gettempdir()

    stem = Path(video_path).stem
    out_path = os.path.join(output_dir, f"{stem}_audio.wav")

    clip = VideoFileClip(video_path)
    if clip.audio is None:
        raise ValueError("Video has no audio track.")

    clip.audio.write_audiofile(out_path, verbose=False, logger=None)
    clip.close()

    return out_path


def get_video_duration(video_path: str) -> float:
    """Return video duration in seconds using moviepy."""
    from moviepy.editor import VideoFileClip
    clip = VideoFileClip(video_path)
    dur  = clip.duration
    clip.close()
    return dur


def get_video_fps(video_path: str) -> float:
    """Return video FPS."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return fps


def format_seconds(sec: float) -> str:
    """Convert seconds to MM:SS string."""
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist, then return path."""
    os.makedirs(path, exist_ok=True)
    return path


def cleanup_files(*paths: str) -> None:
    """Delete files silently, ignoring errors."""
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
