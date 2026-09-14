"""champion_detection.py - given an uploaded video and a [start, end] window, extracts frames
at the requested fps (same ffmpeg -ss/-t seek approach as analyze.py — frames come straight
from the original upload, no intermediate trimmed file) and runs each one through the vision
model, one frame per call, using the prompt from champion_detection_prompt(fps). Each frame's
detected champions (id + normalized position) are returned alongside its timestamp; if a
frame's response wasn't valid JSON, the raw model text is returned instead so nothing is lost.
"""

import logging
import os
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from video_utils import (
    FFmpegError,
    check_ffmpeg_available,
    extract_frames_fps,
    get_duration,
)
from gemini_pipeline import DEFAULT_MODEL, build_client, call_gemini_frames, parse_json_block
from prompts import champion_detection_prompt
from champion_debug import draw_champion_debug_frame

MAX_CLIP_DURATION = 20.0
MIN_CLIP_DURATION = 1.0
# See analyze.py — trim selections arrive as floats from client-side drag math, so give the
# bounds checks a little tolerance rather than rejecting an intended-exact 20.00000000000003s.
DURATION_EPSILON = 0.05
DETECTION_FPS = 2.0
CHUNK_SIZE = 1024 * 1024

# Champion detection reads small UI elements (health bar color/position), which is far more
# resolution-sensitive than the event/coaching pipeline's extract_frames_fps calls (which just
# need enough detail for broad movement/combat patterns). Give this endpoint its own cap
# instead of inheriting video_utils.ANALYSIS_FRAME_WIDTH, so raising it doesn't also make the
# event/coaching pipeline more expensive. Like all max_width caps in video_utils, this never
# upscales past the source resolution -- it only helps if the source is actually wider than this.
CHAMPION_DETECTION_FRAME_WIDTH = 1536

# Debug overlays land here, one subfolder per request, so runs don't clobber each other.
# Unlike the raw extracted frames (temp dir, cleaned up in `finally`), this is meant to be
# inspected afterward, so it's not deleted automatically.
CHAMPION_DEBUG_ROOT = os.environ.get("CHAMPION_DEBUG_DIR", "champions_debug")

logger = logging.getLogger("gamesense.champion_detection")

router = APIRouter()

DETECTION_TIMEOUT_SECONDS = 30.0


def run_champion_detection(
    frame_paths: List[str],
    prompt: str,
    vision_model: str,
    timeout_seconds: float = DETECTION_TIMEOUT_SECONDS,
    debug_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run champion detection on each frame independently (the prompt is single-frame, so
    each call only ever sees one image — champion identity/positions aren't compared across
    frames here, that's left to whatever consumes this output, e.g. the event pipeline).

    Returns one entry per frame: {"frame_index", "timestamp_seconds", "champions",
    "valid_json", "raw_output"}. `champions` is the parsed list from the model's JSON when
    valid (the common case), and empty when it wasn't -- in which case `raw_output` holds
    the model's raw text so nothing is silently dropped.

    If debug_dir is given, a copy of each frame with the detected champion positions drawn
    on it (yellow=played, green=friendly, red=enemy) is saved there, named to match the
    source frame. Frames with no valid JSON are saved with zero circles -- that's expected
    and still useful, since seeing the frame itself tells you whether it's a hard case.
    """
    client = build_client(timeout_seconds)
    results: List[Dict[str, Any]] = []

    for frame_index, frame_path in enumerate(frame_paths):
        logger.info(
            "champion_detection: frame %d/%d...", frame_index + 1, len(frame_paths)
        )
        raw_output = call_gemini_frames(client, vision_model, prompt, [frame_path])
        parsed, valid_json = parse_json_block(raw_output)

        champions = []
        if valid_json and isinstance(parsed, dict):
            champions = parsed.get("champions", [])
        elif not valid_json:
            logger.warning(
                "champion_detection: frame %d output wasn't valid JSON; raw text kept in raw_output",
                frame_index,
            )

        # Defense against the model occasionally reporting pixel coordinates instead of the
        # requested 0.0-1.0 fraction (e.g. x: 622 on a 1536px-wide frame) -- such a value would
        # otherwise pass straight through to the API response and silently draw off-canvas in
        # the debug overlay. Drop rather than clamp: clamping would hide the mistake behind a
        # plausible-looking edge position instead of surfacing it.
        valid_champions = []
        for champ in champions:
            x, y = champ.get("x"), champ.get("y")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                valid_champions.append(champ)
            else:
                logger.warning(
                    "champion_detection: frame %d dropped '%s' -- x/y out of the 0.0-1.0 range (x=%r, y=%r), looks like a pixel coordinate",
                    frame_index, champ.get("id", "unknown"), x, y,
                )
        champions = valid_champions

        if debug_dir:
            debug_output_path = os.path.join(debug_dir, os.path.basename(frame_path))
            try:
                draw_champion_debug_frame(frame_path, champions, debug_output_path)
            except Exception:
                # Debug visualization is a diagnostic aid, not part of the actual detection
                # result -- a drawing failure shouldn't take down the request.
                logger.exception(
                    "champion_detection: failed to draw debug overlay for frame %d", frame_index
                )

        results.append(
            {
                "frame_index": frame_index,
                "champions": champions,
                "valid_json": valid_json,
                "raw_output": raw_output,
            }
        )

    return results


class FrameDetection(BaseModel):
    frame_index: int
    timestamp_seconds: float
    champions: List[Dict[str, Any]]
    valid_json: bool
    raw_output: str


class ChampionDetectionResponse(BaseModel):
    fps: float
    frame_count: int
    frames: List[FrameDetection]
    timing_seconds: dict
    debug_dir: str | None = None


@router.post("/api/video/champion_detection", response_model=ChampionDetectionResponse)
async def detect_champions(
    video: UploadFile = File(...),
    start: float = Form(...),
    end: float = Form(...),
    fps: float = Form(DETECTION_FPS),
    vision_model: str = Form(DEFAULT_MODEL),
    debug: bool = Form(True),
):
    request_start = time.perf_counter()

    if not check_ffmpeg_available():
        raise HTTPException(status_code=500, detail="ffmpeg/ffprobe not found on server")
    if not video.filename:
        raise HTTPException(status_code=400, detail="video is required")
    if start < 0:
        raise HTTPException(status_code=400, detail="start time must be >= 0")
    if end <= start:
        raise HTTPException(status_code=400, detail="end time must be greater than start time")

    clip_duration = end - start
    if clip_duration > MAX_CLIP_DURATION + DURATION_EPSILON:
        raise HTTPException(
            status_code=400,
            detail=f"requested clip is {clip_duration:.1f}s, which exceeds the {MAX_CLIP_DURATION:.0f}s limit",
        )
    if clip_duration < MIN_CLIP_DURATION - DURATION_EPSILON:
        raise HTTPException(
            status_code=400,
            detail=f"requested clip must be at least {MIN_CLIP_DURATION:.0f}s",
        )
    clip_duration = max(MIN_CLIP_DURATION, min(clip_duration, MAX_CLIP_DURATION))
    end = start + clip_duration

    input_suffix = os.path.splitext(video.filename)[1] or ".mp4"
    input_fd, input_path = tempfile.mkstemp(suffix=input_suffix)
    frames_dir = tempfile.mkdtemp(prefix="champion_frames_")

    debug_dir: str | None = None
    if debug:
        debug_dir = os.path.join(CHAMPION_DEBUG_ROOT, uuid.uuid4().hex[:12])
        os.makedirs(debug_dir, exist_ok=True)
        logger.info("champion_detection: debug overlays will be saved to %s", debug_dir)

    timing: dict = {}

    try:
        # --- save upload ---
        logger.info("champion_detection: saving upload '%s' to %s", video.filename, input_path)
        upload_start = time.perf_counter()
        try:
            with os.fdopen(input_fd, "wb") as f:
                while True:
                    chunk = await video.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception:
            logger.exception("failed to save uploaded video")
            raise HTTPException(status_code=500, detail="failed to save uploaded video")
        timing["upload"] = round(time.perf_counter() - upload_start, 3)
        logger.info("champion_detection: upload saved (%.2fs)", timing["upload"])

        # --- make sure the requested window actually fits inside the source video ---
        try:
            source_duration = await run_in_threadpool(get_duration, input_path)
        except FFmpegError as e:
            logger.error("champion_detection: failed to read video duration: %s", e)
            raise HTTPException(status_code=500, detail=f"failed to read video: {e}")
        if end > source_duration:
            raise HTTPException(
                status_code=400,
                detail=f"end time {end}s exceeds video duration {source_duration:.2f}s",
            )

        # --- extract frames at the requested fps, straight from the [start, end] window ---
        logger.info("champion_detection: extracting frames at %s fps...", fps)
        frame_extraction_start = time.perf_counter()
        try:
            frame_paths = await run_in_threadpool(
                extract_frames_fps, input_path, frames_dir, fps, start, clip_duration,
                CHAMPION_DETECTION_FRAME_WIDTH,
            )
        except FFmpegError as e:
            logger.error("champion_detection: frame extraction failed: %s", e)
            raise HTTPException(status_code=500, detail=f"frame extraction failed: {e}")
        if not frame_paths:
            raise HTTPException(status_code=500, detail="no frames were extracted")
        timing["frame_extraction"] = round(time.perf_counter() - frame_extraction_start, 3)
        logger.info(
            "champion_detection: extracted %d frames (%.2fs), running detection (model=%s)...",
            len(frame_paths), timing["frame_extraction"], vision_model,
        )

        # --- run champion detection on each extracted frame ---
        prompt = champion_detection_prompt(fps)
        detection_start = time.perf_counter()
        try:
            frames = await run_in_threadpool(
                run_champion_detection, frame_paths, prompt, vision_model,
                DETECTION_TIMEOUT_SECONDS, debug_dir,
            )
        except Exception as e:
            logger.exception("champion detection failed")
            raise HTTPException(status_code=502, detail=f"champion detection failed: {e}")
        for frame in frames:
            frame["timestamp_seconds"] = round(frame["frame_index"] / fps, 3)
        timing["detection"] = round(time.perf_counter() - detection_start, 3)
        logger.info("champion_detection: detection complete (%.2fs)", timing["detection"])

        timing["total"] = round(time.perf_counter() - request_start, 3)
        logger.info(
            "champion_detection: total=%.2fs upload=%.2fs frame_extraction=%.2fs detection=%.2fs",
            timing["total"], timing["upload"], timing["frame_extraction"], timing["detection"],
        )

        return ChampionDetectionResponse(
            fps=fps,
            frame_count=len(frame_paths),
            frames=frames,
            timing_seconds=timing,
            debug_dir=debug_dir,
        )
    finally:
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except OSError:
            logger.warning("failed to remove temp file: %s", input_path)
        try:
            for name in os.listdir(frames_dir):
                os.remove(os.path.join(frames_dir, name))
            os.rmdir(frames_dir)
        except OSError:
            logger.warning("failed to remove temp dir: %s", frames_dir)