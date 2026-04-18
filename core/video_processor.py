"""
video_processor.py

The heart of AttentionX.  Takes a video clip window + face trajectory +
word timestamps and produces a polished 9:16 short-form video with:
  - Smart face-centred vertical crop
  - Karaoke-style word-highlighted captions at the bottom
  - Hook headline overlay at the top
  - Audio preserved from the original
"""

import cv2
import numpy as np
import os
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Caption style constants
CAPTION_FONT_SCALE_RATIO = 0.042   # font height as fraction of frame height
HOOK_FONT_SCALE_RATIO    = 0.036
CAPTION_Y_RATIO          = 0.83    # vertical position of captions (83% down)
HOOK_Y_RATIO             = 0.05    # hook at top (5%)


# ── Font Loading ─────────────────────────────────────────────────────────────

def _get_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    """Try to load a clean sans-serif font; fall back to PIL default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",   # macOS
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue

    # Last resort — PIL built-in (no size control, but won't crash)
    return ImageFont.load_default()


# ── Core Frame Processing ────────────────────────────────────────────────────

def crop_to_vertical(frame: np.ndarray, face_cx: float) -> np.ndarray:
    """
    Crop a 16:9 (or any landscape) BGR frame to 9:16 (vertical).
    Centres the crop window on face_cx (normalised 0–1 X position).
    """
    h, w = frame.shape[:2]
    target_w = int(h * 9 / 16)

    if target_w >= w:
        # Already narrower than 9:16 — just return as-is
        return frame

    # Ideal crop centre = face position mapped to pixel x
    ideal_cx = int(face_cx * w)
    x_start  = ideal_cx - target_w // 2

    # Clamp so we don't go out of bounds
    x_start  = max(0, min(x_start, w - target_w))
    x_end    = x_start + target_w

    return frame[:, x_start:x_end]


def _get_active_words(
    words: List[Dict],
    current_time: float,
    window: int = 5,
) -> Tuple[List[Dict], int]:
    """
    Return the window of words around the current timestamp, plus
    the index of the word being spoken right now.

    Returns: (word_list, active_index_or_-1)
    """
    # Find currently spoken word
    active_idx_global = -1
    for i, w in enumerate(words):
        if w["start"] <= current_time <= w["end"]:
            active_idx_global = i
            break

    # If between words, highlight the last finished word briefly
    if active_idx_global == -1:
        for i in range(len(words) - 1, -1, -1):
            if words[i]["end"] <= current_time:
                active_idx_global = i
                break

    if active_idx_global == -1:
        return [], -1

    # Build window: 2 words back, rest forward
    win_start = max(0, active_idx_global - 2)
    win_end   = min(len(words), win_start + window)
    win_start = max(0, win_end - window)   # re-clamp in case near end

    window_words = words[win_start:win_end]
    local_active = active_idx_global - win_start

    return window_words, local_active


def render_caption(
    pil_img: Image.Image,
    words: List[Dict],
    current_time: float,
    hook_text: str,
    show_hook: bool = True,
) -> Image.Image:
    """
    Draw karaoke captions + hook headline onto a PIL Image.
    Returns the modified image.
    """
    w, h = pil_img.size
    draw = ImageDraw.Draw(pil_img)

    cap_size  = max(22, int(h * CAPTION_FONT_SCALE_RATIO))
    hook_size = max(18, int(h * HOOK_FONT_SCALE_RATIO))

    cap_font  = _get_font(cap_size,  bold=True)
    hook_font = _get_font(hook_size, bold=True)

    # ── Karaoke Caption ──────────────────────────────────────────────────────
    window_words, active_idx = _get_active_words(words, current_time, window=5)

    if window_words:
        cap_y = int(h * CAPTION_Y_RATIO)

        # Measure total width to centre
        full_text = " ".join(wd["word"] for wd in window_words)
        try:
            bbox_full = draw.textbbox((0, 0), full_text, font=cap_font)
            total_w   = bbox_full[2] - bbox_full[0]
        except Exception:
            total_w = len(full_text) * cap_size // 2

        x_cursor = (w - total_w) // 2
        padding  = 14

        # Semi-transparent background pill
        bg_layer = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        bg_draw  = ImageDraw.Draw(bg_layer)
        bg_rect  = [
            x_cursor - padding,
            cap_y - padding // 2,
            x_cursor + total_w + padding,
            cap_y + cap_size + padding,
        ]
        bg_draw.rounded_rectangle(bg_rect, radius=10, fill=(0, 0, 0, 185))
        pil_img = Image.alpha_composite(pil_img.convert("RGBA"), bg_layer).convert("RGB")
        draw    = ImageDraw.Draw(pil_img)

        # Draw each word individually to enable per-word highlight
        for i, wd in enumerate(window_words):
            word_str = wd["word"] + (" " if i < len(window_words) - 1 else "")
            color    = (255, 230, 0) if i == active_idx else (255, 255, 255)

            # Outline
            for ox, oy in [(-2, -2), (2, -2), (-2, 2), (2, 2), (0, -2), (0, 2)]:
                draw.text((x_cursor + ox, cap_y + oy), word_str, font=cap_font, fill=(0, 0, 0))

            draw.text((x_cursor, cap_y), word_str, font=cap_font, fill=color)

            try:
                word_bbox = draw.textbbox((0, 0), word_str, font=cap_font)
                x_cursor += word_bbox[2] - word_bbox[0]
            except Exception:
                x_cursor += len(word_str) * cap_size // 2

    # ── Hook Headline (top of frame) ─────────────────────────────────────────
    if show_hook and hook_text:
        hook_y = int(h * HOOK_Y_RATIO)
        try:
            hook_bbox = draw.textbbox((0, 0), hook_text, font=hook_font)
            hook_w    = hook_bbox[2] - hook_bbox[0]
        except Exception:
            hook_w    = len(hook_text) * hook_size // 2

        hook_x = (w - hook_w) // 2
        pad    = 10

        # Background
        hook_bg = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        hook_bd = ImageDraw.Draw(hook_bg)
        hook_bd.rounded_rectangle(
            [hook_x - pad, hook_y - pad // 2,
             hook_x + hook_w + pad, hook_y + hook_size + pad],
            radius=8,
            fill=(99, 102, 241, 210),    # Indigo brand colour
        )
        pil_img = Image.alpha_composite(pil_img.convert("RGBA"), hook_bg).convert("RGB")
        draw    = ImageDraw.Draw(pil_img)

        # Outline + text
        for ox, oy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            draw.text((hook_x + ox, hook_y + oy), hook_text, font=hook_font, fill=(0, 0, 50))
        draw.text((hook_x, hook_y), hook_text, font=hook_font, fill=(255, 255, 255))

    return pil_img


# ── Main Export Function ─────────────────────────────────────────────────────

def process_clip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    face_trajectory: Dict[int, float],
    words: List[Dict],
    hook_text: str,
    output_path: str,
    progress_callback=None,
) -> str:
    """
    Full pipeline for a single clip:
      1. Open video at start_sec
      2. For each frame: crop to 9:16 (face-tracked) + draw captions
      3. Write frames to a temp video
      4. Merge audio from original using MoviePy
      5. Save final MP4 to output_path

    Args:
        video_path:        Original source video path.
        start_sec:         Clip start time in seconds.
        end_sec:           Clip end time in seconds.
        face_trajectory:   Dict from face_tracker.build_face_trajectory().
        words:             Word timestamp list from transcriber.
        hook_text:         Headline shown at top of frame.
        output_path:       Where to save the final clip.
        progress_callback: Optional callable(fraction 0–1) for UI updates.

    Returns:
        output_path on success.
    """
    cap  = cv2.VideoCapture(video_path)
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output dims
    out_h = orig_h
    out_w = int(orig_h * 9 / 16)
    if out_w > orig_w:
        out_w = orig_w

    start_frame = int(start_sec * fps)
    end_frame   = int(end_sec   * fps)
    total_frames = max(1, end_frame - start_frame)

    # Temp file for video-only (no audio yet)
    tmp_vid = output_path.replace(".mp4", "_noaudio.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_vid, fourcc, fps, (out_w, out_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Filter word timestamps to this clip's range
    clip_words = [
        w for w in words
        if w["end"] >= start_sec and w["start"] <= end_sec
    ]
    # Make timestamps relative to clip start for rendering
    clip_words_rel = [
        {**w, "start": w["start"] - start_sec, "end": w["end"] - start_sec}
        for w in clip_words
    ]

    logger.info(f"Processing {total_frames} frames → {out_w}×{out_h} vertical crop…")

    frame_idx = start_frame
    processed = 0
    # Show hook only for the first 5 seconds
    hook_end_sec = 5.0

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Face centre for this frame
        face_cx = face_trajectory.get(frame_idx, 0.5)

        # Crop to 9:16
        cropped = crop_to_vertical(frame, face_cx)

        # Convert to PIL for caption rendering
        pil_img  = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        t_in_clip = (frame_idx - start_frame) / fps

        show_hook = t_in_clip <= hook_end_sec
        pil_img = render_caption(pil_img, clip_words_rel, t_in_clip, hook_text, show_hook)

        # Back to BGR numpy for OpenCV writer
        out_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Ensure correct size (PIL resize can sometimes shift by 1px)
        if out_frame.shape[1] != out_w or out_frame.shape[0] != out_h:
            out_frame = cv2.resize(out_frame, (out_w, out_h))

        writer.write(out_frame)
        frame_idx += 1
        processed += 1

        if progress_callback and processed % 30 == 0:
            progress_callback(processed / total_frames)

    cap.release()
    writer.release()

    # ── Merge audio using MoviePy ────────────────────────────────────────────
    logger.info("Merging audio…")
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip

        silent_clip   = VideoFileClip(tmp_vid)
        original_clip = VideoFileClip(video_path).subclip(start_sec, end_sec)

        if original_clip.audio is not None:
            final_clip = silent_clip.set_audio(original_clip.audio)
        else:
            final_clip = silent_clip

        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None,
        )
        silent_clip.close()
        original_clip.close()
        final_clip.close()

    except Exception as e:
        logger.error(f"Audio merge failed: {e}")
        # Still provide the silent version if audio fails
        import shutil
        shutil.copy(tmp_vid, output_path)
    finally:
        if os.path.exists(tmp_vid):
            os.remove(tmp_vid)

    if progress_callback:
        progress_callback(1.0)

    logger.info(f"Saved clip → {output_path}")
    return output_path
