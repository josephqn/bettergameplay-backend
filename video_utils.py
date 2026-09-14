"""video_utils.py - shared ffmpeg helpers for trimming and frame extraction.

Pulled out of trim.py / classify.py / analyze_clip.py so all three call sites
(the CLI scripts and the FastAPI endpoints) share one implementation instead
of three slightly-diverging copies.
"""

import logging
import os
import shutil
import subprocess
from typing import List, Optional

logger = logging.getLogger("bettergameplay.video_utils")

# Caps on extracted frame width (aspect ratio preserved, height rounded to even). Frames are
# never upscaled past their source resolution -- these are ceilings, not fixed targets.
# CLASSIFY_FRAME_WIDTH: CLIP resizes everything to 224x224 internally, so there's no benefit
# to feeding it full source-resolution frames.
# ANALYSIS_FRAME_WIDTH: frames sent to the Gemini vision call. Kept large enough to keep
# on-screen UI text/icons legible, but well under typical 1080p/1440p capture resolution.
CLASSIFY_FRAME_WIDTH = 256
ANALYSIS_FRAME_WIDTH = 1024


class FFmpegError(RuntimeError):
    """Raised when an ffmpeg/ffprobe subprocess call fails."""


def _scale_filter(max_width: Optional[int]) -> Optional[str]:
    """Build an ffmpeg -vf clause that caps frame width at max_width (aspect-preserving,
    even height) without ever upscaling a smaller source. Returns None if max_width is falsy."""
    if not max_width:
        return None
    return f"scale='min(iw,{max_width})':-2"


def check_ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def get_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError(f"ffprobe failed: {result.stderr.strip()}")
    try:
        return float(result.stdout.strip())
    except ValueError as e:
        raise FFmpegError(f"could not parse duration from ffprobe output: {result.stdout!r}") from e


def trim_clip(video_path: str, output_path: str, start: float, end: float) -> None:
    """Cut [start, end) out of video_path and re-encode it to output_path.

    Kept for standalone use (trim.py) and any case where a materialized trimmed file is
    actually wanted. The /api/video/analyze pipeline no longer calls this: it seeks and pulls
    frames directly out of the source upload instead (see extract_frames_fps /
    extract_frames_count below), skipping this re-encode plus the extra file write/read.
    """
    duration = end - start
    cmd = [
        "ffmpeg", "-nostdin", "-y",
        "-i", video_path,
        "-ss", str(start),
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise FFmpegError(f"ffmpeg trim failed: {result.stderr.decode(errors='ignore')}")


def extract_frames_fps(
    video_path: str,
    out_dir: str,
    fps: float,
    start: Optional[float] = None,
    duration: Optional[float] = None,
    max_width: Optional[int] = ANALYSIS_FRAME_WIDTH,
) -> List[str]:
    """Sample frames at a fixed rate (frames per second). Used for the Gemini vision
    pipeline, which wants even temporal sampling across the whole clip.

    Pass `start`/`duration` to seek directly into a window of video_path (fast, keyframe-based
    input seeking) instead of requiring a pre-trimmed file -- this lets callers skip trimming
    the source video before extracting. Frames are downscaled to max_width (aspect-preserving,
    never upscaled) before being written; omit/None to keep source resolution.
    """
    pattern = os.path.join(out_dir, "frame_%04d.jpg")
    cmd = ["ffmpeg", "-nostdin", "-y"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", video_path]
    if duration is not None:
        cmd += ["-t", str(duration)]
    vf_parts = [f"fps={fps}"]
    scale = _scale_filter(max_width)
    if scale:
        vf_parts.append(scale)
    cmd += ["-vf", ",".join(vf_parts), "-q:v", "2", pattern]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise FFmpegError(f"ffmpeg frame extraction failed: {result.stderr.decode(errors='ignore')}")
    names = sorted(f for f in os.listdir(out_dir) if f.startswith("frame_") and f.endswith(".jpg"))
    return [os.path.join(out_dir, f) for f in names]


def extract_frames_count(
    video_path: str,
    out_dir: str,
    count: int,
    duration: Optional[float] = None,
    start: float = 0.0,
    max_width: Optional[int] = CLASSIFY_FRAME_WIDTH,
) -> List[str]:
    """Sample a fixed number of frames, evenly spaced across [start, start+duration]. Used for
    the cheap CLIP classifier, which just needs a representative handful of frames.

    `start` lets callers seek directly into a window of an untrimmed source video instead of
    requiring a pre-trimmed file. `duration`, if omitted, defaults to the remaining video length
    after `start`. Frames are downscaled to max_width (aspect-preserving, never upscaled) before
    being written -- CLIP resizes everything to 224x224 internally, so there's no benefit to
    extracting at full source resolution.
    """
    if duration is None:
        duration = get_duration(video_path) - start
    frame_paths = []
    scale = _scale_filter(max_width)
    for i in range(count):
        timestamp = start + duration * (i + 0.5) / count
        frame_path = os.path.join(out_dir, f"frame_{i:02d}.jpg")
        cmd = [
            "ffmpeg", "-nostdin", "-y",
            "-ss", str(timestamp),
            "-i", video_path,
            "-frames:v", "1",
        ]
        if scale:
            cmd += ["-vf", scale]
        cmd += ["-q:v", "2", frame_path]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            logger.warning(
                "frame extraction failed at t=%.2fs: %s",
                timestamp, result.stderr.decode(errors="ignore"),
            )
            continue
        if os.path.exists(frame_path):
            frame_paths.append(frame_path)
    return frame_paths