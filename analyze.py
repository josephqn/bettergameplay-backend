"""analyze.py - combined endpoint. Upload a video plus a start/end window; the server seeks
directly into that window of the upload to pull frames, runs them through the Gemini classifier
to confirm it's League of Legends, and - only if it passes classification - runs the window
through the two-stage Gemini coaching pipeline.

This replaces calling /api/video/trim, /api/video/classify, and analyze_clip.py by hand in
sequence: one upload, one response. Skipping the Gemini calls on non-LoL clips also saves you
API cost on obviously-wrong uploads. There's no intermediate trimmed file: both frame-extraction
passes seek straight into the original upload via ffmpeg -ss/-t, so we never pay for an extra
re-encode + write/read round trip just to hand off to a second ffmpeg call.
"""

import logging
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from video_utils import (
    FFmpegError,
    check_ffmpeg_available,
    extract_frames_count,
    extract_frames_fps,
    get_duration,
)
from classify import LOL_THRESHOLD, NUM_FRAMES, classify_frames
from gemini_pipeline import DEFAULT_MODEL, DEFAULT_TIMEOUT_SECONDS, GeminiRateLimitError, run_coaching_pipeline

MAX_CLIP_DURATION = 20.0
MIN_CLIP_DURATION = 1.0
# Trim selections arrive as floats computed client-side (e.g. from a
# percentage-of-duration drag), so an intended exactly-20s or exactly-1s
# selection can arrive as 20.00000000000003 or 0.9999999999998. Give the
# bounds checks a small tolerance so those aren't rejected as invalid.
DURATION_EPSILON = 0.05
ANALYSIS_FPS = 4.0
CHUNK_SIZE = 1024 * 1024

logger = logging.getLogger("bettergameplay.analyze")

router = APIRouter()


class AnalyzeResponse(BaseModel):
    is_league_of_legends: bool
    classification_confidence: Optional[float] = None
    classification_skipped: bool = False
    coaching: Optional[Union[Dict[str, Any], List[Any], str]] = None
    coaching_valid_json: bool = False
    timing_seconds: dict


@router.post("/api/video/analyze", response_model=AnalyzeResponse)
async def analyze_video(
    video: UploadFile = File(...),
    start: float = Form(...),
    end: float = Form(...),
    fps: float = Form(ANALYSIS_FPS),
    vision_model: str = Form(DEFAULT_MODEL),
    text_model: str = Form(DEFAULT_MODEL),
    gemini_timeout_seconds: float = Form(
        DEFAULT_TIMEOUT_SECONDS,
        description="Read timeout (seconds) for each Gemini API call. Raise this if you see "
                     "'Gemini analysis failed: The read operation timed out' on longer/heavier "
                     "clips (more frames or a slower model).",
    ),
    skip_classification: bool = Form(
        False,
        description="Set true if the caller already classified this clip (e.g. via "
                     "/api/video/classify) and confirmed it's League of Legends footage, to "
                     "avoid re-running the CLIP classifier here.",
    ),
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
    # The tolerance above can let a hair-over selection (e.g. 20.03s) through;
    # clamp it back to the actual bound so ffmpeg and the source-duration
    # check downstream never see more than the limit.
    clip_duration = max(MIN_CLIP_DURATION, min(clip_duration, MAX_CLIP_DURATION))
    end = start + clip_duration

    input_suffix = os.path.splitext(video.filename)[1] or ".mp4"
    input_fd, input_path = tempfile.mkstemp(suffix=input_suffix)
    classify_frames_dir = None if skip_classification else tempfile.mkdtemp(prefix="classify_frames_")
    analysis_frames_dir = tempfile.mkdtemp(prefix="analysis_frames_")

    timing: dict = {}

    try:
        # --- save upload ---
        logger.info("analyze: saving upload '%s' to %s", video.filename, input_path)
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
        logger.info("analyze: upload saved (%.2fs)", timing["upload"])

        # --- make sure the requested window actually fits inside the source video ---
        logger.info("analyze: reading source video duration...")
        try:
            source_duration = await run_in_threadpool(get_duration, input_path)
        except FFmpegError as e:
            logger.error("analyze: failed to read video duration: %s", e)
            raise HTTPException(status_code=500, detail=f"failed to read video: {e}")
        logger.info("analyze: source duration=%.2fs, requested window=%.2fs-%.2fs", source_duration, start, end)
        if end > source_duration + DURATION_EPSILON:
            raise HTTPException(
                status_code=400,
                detail=f"end time {end:.2f}s exceeds video duration {source_duration:.2f}s",
            )
        # As with the MAX/MIN clip-duration clamp above, the epsilon can let an
        # `end` that's a hair past the true source duration (float rounding from
        # the client, or ffprobe reporting slightly short) through the check
        # above. Clamp it back so neither ffmpeg seek below ever asks for frames
        # past the actual end of the source file.
        if end > source_duration:
            end = source_duration
            clip_duration = end - start
            if clip_duration < MIN_CLIP_DURATION - DURATION_EPSILON:
                raise HTTPException(
                    status_code=400,
                    detail=f"requested clip must be at least {MIN_CLIP_DURATION:.0f}s",
                )

        # --- classify with Gemini before running the coaching pipeline ---
        # Skippable when the caller already classified this clip (e.g. via /api/video/classify)
        # to avoid classifying the same footage twice.
        # Frames are pulled directly from the [start, end] window of the original upload --
        # no intermediate trimmed file is created; ffmpeg seeks straight to each timestamp.
        confidence: Optional[float] = None
        if skip_classification:
            timing["classify"] = 0.0
            logger.info("analyze: skipping classification (skip_classification=true)")
        else:
            logger.info("analyze: extracting %d frames for classification...", NUM_FRAMES)
            classify_start = time.perf_counter()
            classify_paths = await run_in_threadpool(
                extract_frames_count,
                input_path,
                classify_frames_dir,
                NUM_FRAMES,
                clip_duration,
                start,
            )
            if not classify_paths:
                raise HTTPException(status_code=500, detail="failed to extract frames for classification")
            logger.info("analyze: running Gemini classification on %d frames...", len(classify_paths))
            try:
                confidence, _per_frame = await run_in_threadpool(classify_frames, classify_paths)
            except Exception:
                logger.exception("classification failed")
                raise HTTPException(status_code=500, detail="failed to classify video")
            timing["classify"] = round(time.perf_counter() - classify_start, 3)
            logger.info("analyze: classification complete (%.2fs, confidence=%.4f)", timing["classify"], confidence)

            if confidence < LOL_THRESHOLD:
                timing["total"] = round(time.perf_counter() - request_start, 3)
                logger.info("analyze: clip rejected, not League of Legends (confidence=%.4f)", confidence)
                return AnalyzeResponse(
                    is_league_of_legends=False,
                    classification_confidence=round(confidence, 4),
                    timing_seconds=timing,
                )

        # --- Gemini coaching pipeline (only reached for clips that passed classification) ---
        logger.info("analyze: extracting frames for Gemini analysis at %s fps...", fps)
        frame_extraction_start = time.perf_counter()
        try:
            analysis_paths = await run_in_threadpool(
                extract_frames_fps,
                input_path,
                analysis_frames_dir,
                fps,
                start,
                clip_duration,
            )
        except FFmpegError as e:
            logger.error("analyze: frame extraction for analysis failed: %s", e)
            raise HTTPException(status_code=500, detail=f"frame extraction for analysis failed: {e}")
        if not analysis_paths:
            raise HTTPException(status_code=500, detail="no frames were extracted for analysis")
        timing["frame_extraction"] = round(time.perf_counter() - frame_extraction_start, 3)
        logger.info(
            "analyze: extracted %d frames for analysis (%.2fs), starting Gemini pipeline (vision=%s, text=%s)...",
            len(analysis_paths), timing["frame_extraction"], vision_model, text_model,
        )

        try:
            result = await run_in_threadpool(
                run_coaching_pipeline, analysis_paths, fps, vision_model, text_model, gemini_timeout_seconds
            )
        except GeminiRateLimitError:
            logger.warning("analyze: Gemini rate limit exceeded")
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        except Exception as e:
            logger.exception("Gemini analysis failed")
            raise HTTPException(status_code=502, detail=f"Gemini analysis failed: {e}")
        timing["event_extraction"] = result["event_extraction_seconds"]
        timing["coaching"] = result["coaching_seconds"]
        logger.info("analyze: Gemini pipeline complete")

        if not result["events_valid_json"]:
            logger.warning("analyze: stage 1 output wasn't valid JSON; raw text was passed through to stage 2")
        logger.info("analyze: extracted events: %s", result["events"])
        if not result["coaching_valid_json"]:
            logger.warning("analyze: stage 2 output wasn't valid JSON; returning raw text in 'coaching'")

        timing["total"] = round(time.perf_counter() - request_start, 3)
        logger.info(
            "analyze: total=%.2fs upload=%.2fs classify=%.2fs "
            "frame_extraction=%.2fs event_extraction=%.2fs coaching=%.2fs confidence=%s",
            timing["total"], timing["upload"], timing["classify"],
            timing["frame_extraction"], timing["event_extraction"], timing["coaching"],
            "skipped" if confidence is None else f"{confidence:.4f}",
        )

        return AnalyzeResponse(
            is_league_of_legends=True,
            classification_confidence=None if confidence is None else round(confidence, 4),
            classification_skipped=skip_classification,
            coaching=result["coaching"],
            coaching_valid_json=result["coaching_valid_json"],
            timing_seconds=timing,
        )
    finally:
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except OSError:
            logger.warning("failed to remove temp file: %s", input_path)
        for d in (classify_frames_dir, analysis_frames_dir):
            if d is None:
                continue
            try:
                for name in os.listdir(d):
                    os.remove(os.path.join(d, name))
                os.rmdir(d)
            except OSError:
                logger.warning("failed to remove temp dir: %s", d)