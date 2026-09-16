"""analyze.py - combined endpoint. Upload a video plus a start/end window; the server seeks
directly into that window of the upload to pull frames, runs them through the Gemini classifier
to confirm it's League of Legends, and - only if it passes classification - runs the window
through the two-stage Gemini coaching pipeline.

This replaces calling /api/video/trim, /api/video/classify, and analyze_clip.py by hand in
sequence: one upload, one response. Skipping the Gemini calls on non-LoL clips also saves you
API cost on obviously-wrong uploads. All media probing and frame extraction runs through Very Good FFmpeg.
"""

import logging
import os
import random
import tempfile
import time
from uuid import uuid4
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from very_good_service import VeryGoodVideoProcessor, validate_video_constraints
from storage_service import R2ObjectStorage, StorageConfigurationError, StorageOperationError
from classify import LOL_THRESHOLD, classify_frames
from gemini_pipeline import DEFAULT_MODEL, DEFAULT_TIMEOUT_SECONDS, GeminiRateLimitError, run_coaching_pipeline

MAX_CLIP_DURATION = 20.0
MIN_CLIP_DURATION = 1.0
MAX_UPLOAD_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024
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
    duration: Optional[float] = Form(None, description="Source video duration in seconds, read by the client before upload."),
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
    print("🔥 ENTERED /api/video/analyze", flush=True)
    print("ANALYZE 1: endpoint entered; multipart fields and upload parsed", flush=True)
    request_start = time.perf_counter()

    print("ANALYZE 2: validating upload and request parameters", flush=True)
    video_processor = VeryGoodVideoProcessor.from_env()
    if video_processor is None:
        raise HTTPException(status_code=500, detail="Very Good FFmpeg video processing is not configured")
    try:
        storage = R2ObjectStorage.from_env()
    except StorageConfigurationError as exc:
        logger.error("source storage configuration is invalid: %s", exc)
        raise HTTPException(status_code=500, detail="Source video storage is not configured correctly") from exc
    if storage is None:
        raise HTTPException(status_code=500, detail="Source video storage is not configured")
    job_id = uuid4().hex
    if not video.filename:
        raise HTTPException(status_code=400, detail="video is required")
    if start < 0:
        raise HTTPException(status_code=400, detail="start time must be >= 0")
    if end <= start:
        raise HTTPException(status_code=400, detail="end time must be greater than start time")
    if duration is not None:
        if duration <= 0:
            raise HTTPException(status_code=400, detail="Video duration must be greater than zero.")
        try:
            validate_video_constraints(file_size_bytes=0, duration_seconds=duration)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if video.size is not None and video.size > MAX_UPLOAD_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="Video must be smaller than 2 GB.")

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
    # Clamp it back to the actual bound before remote processing.
    clip_duration = max(MIN_CLIP_DURATION, min(clip_duration, MAX_CLIP_DURATION))
    end = start + clip_duration

    input_suffix = os.path.splitext(video.filename)[1] or ".mp4"
    input_fd, input_path = tempfile.mkstemp(suffix=input_suffix)
    analysis_frames_dir = tempfile.mkdtemp(prefix="analysis_frames_")

    timing: dict = {}
    source_object_key: Optional[str] = None

    try:
        print("ANALYZE 3: upload received; saving video", flush=True)
        # --- save upload ---
        logger.info("analyze: saving uploaded video to temporary storage")
        upload_start = time.perf_counter()
        total_bytes = 0
        try:
            with os.fdopen(input_fd, "wb") as f:
                while True:
                    chunk = await video.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    f.write(chunk)
        except Exception:
            logger.exception("failed to save uploaded video")
            raise HTTPException(status_code=500, detail="failed to save uploaded video")
        if total_bytes > MAX_UPLOAD_FILE_SIZE_BYTES:
            raise HTTPException(status_code=400, detail="Video must be smaller than 2 GB.")
        logger.info("STORAGE 1: validated video")
        timing["upload"] = round(time.perf_counter() - upload_start, 3)
        print(f"ANALYZE 4: video saved ({timing['upload']:.2f}s)", flush=True)
        logger.info("analyze: upload saved (%.2fs)", timing["upload"])

        # --- upload to private R2 and pass its URL to Very Good FFmpeg ---
        print("ANALYZE 5: uploading video to object storage", flush=True)
        source_storage_start = time.perf_counter()
        try:
            source_object_key = await run_in_threadpool(storage.upload_file, input_path, job_id=job_id)
            source_url = await run_in_threadpool(storage.create_presigned_get_url, source_object_key)
            # Older clients do not send duration. Keep the existing API usable;
            # updated clients provide the browser-measured duration for the
            # complete 10-minute validation.
            source_duration = duration if duration is not None else end
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except StorageOperationError as e:
            logger.exception("analyze: source storage operation failed")
            raise HTTPException(status_code=502, detail="Source video storage failed") from e
        except Exception as e:
            logger.exception("analyze: failed to create Very Good FFmpeg-accessible source URL")
            raise HTTPException(status_code=502, detail="Source video storage failed") from e
        timing["source_storage"] = round(time.perf_counter() - source_storage_start, 3)
        logger.info("perf analyze: source storage handoff=%.2fs", timing["source_storage"])
        logger.info("analyze: source duration=%.2fs, requested window=%.2fs-%.2fs", source_duration, start, end)
        print(f"ANALYZE 6: source duration validated ({source_duration:.2f}s)", flush=True)
        if end > source_duration + DURATION_EPSILON:
            raise HTTPException(
                status_code=400,
                detail=f"end time {end:.2f}s exceeds video duration {source_duration:.2f}s",
            )
        # As with the MAX/MIN clip-duration clamp above, the epsilon can let an
        # `end` that's a hair past the true source duration (float rounding from
        # the client, or remote metadata reporting slightly short) through the check
        # above. Clamp it back so remote extraction never asks for frames past
        # the actual end of the source file.
        if end > source_duration:
            end = source_duration
            clip_duration = end - start
            if clip_duration < MIN_CLIP_DURATION - DURATION_EPSILON:
                raise HTTPException(
                    status_code=400,
                    detail=f"requested clip must be at least {MIN_CLIP_DURATION:.0f}s",
                )

        # --- extract the single shared frame set used by classification and event analysis ---
        print("ANALYZE 7: extracting analysis frames", flush=True)
        logger.info("analyze: extracting analysis frames at %.2f FPS for Gemini...", fps)
        frame_extraction_start = time.perf_counter()
        try:
            analysis_paths = await run_in_threadpool(
                video_processor.extract_frames,
                source_url,
                start_seconds=start,
                duration_seconds=clip_duration,
                fps=fps,
                destination_dir=analysis_frames_dir,
                max_width=1024,
            )
        except Exception as e:
            logger.error("analyze: Very Good FFmpeg frame extraction failed: %s", e)
            raise HTTPException(status_code=502, detail=f"Very Good FFmpeg frame extraction failed: {e}")
        if not analysis_paths:
            raise HTTPException(status_code=500, detail="no frames were extracted for analysis")
        timing["frame_extraction"] = round(time.perf_counter() - frame_extraction_start, 3)
        print(f"ANALYZE 8: analysis frames extracted ({len(analysis_paths)} frames)", flush=True)
        logger.info(
            "perf analyze: Very Good frame extraction=%.2fs frames=%d",
            timing["frame_extraction"], len(analysis_paths),
        )

        # --- classify one random analysis frame with Gemini ---
        # Skippable when the caller already classified this clip (e.g. via /api/video/classify)
        # to avoid running the classification call twice.
        confidence: Optional[float] = None
        if skip_classification:
            timing["classify"] = 0.0
            print("ANALYZE 9: classification skipped by request", flush=True)
            logger.info("analyze: skipping classification (skip_classification=true)")
        else:
            classification_frame = random.choice(analysis_paths)
            logger.info("analyze: classifying one random analysis frame: %s", os.path.basename(classification_frame))
            try:
                print("ANALYZE 10: starting Gemini classification call", flush=True)
                classify_call_start = time.perf_counter()
                confidence, _per_frame = await run_in_threadpool(classify_frames, [classification_frame])
            except Exception:
                logger.exception("classification failed")
                raise HTTPException(status_code=500, detail="failed to classify video")
            classify_call_seconds = time.perf_counter() - classify_call_start
            timing["classify"] = round(classify_call_seconds, 3)
            logger.info(
                "perf analyze: Gemini one-frame classification=%.2fs confidence=%.4f",
                timing["classify"], confidence,
            )
            print(
                f"ANALYZE 11: Gemini classification complete ({timing['classify']:.2f}s)",
                flush=True,
            )
            logger.info("analyze: classification complete (%.2fs, confidence=%.4f)", timing["classify"], confidence)

            if confidence < LOL_THRESHOLD:
                timing["total"] = round(time.perf_counter() - request_start, 3)
                logger.info("analyze: clip rejected, not League of Legends (confidence=%.4f)", confidence)
                print("ANALYZE 12: clip rejected during classification", flush=True)
                rejected_response = AnalyzeResponse(
                    is_league_of_legends=False,
                    classification_confidence=round(confidence, 4),
                    timing_seconds=timing,
                )
                print("ANALYZE 13: rejection response created", flush=True)
                return rejected_response

        # --- Gemini coaching pipeline (only reached for clips that passed classification) ---
        print("ANALYZE 13: starting Gemini coaching pipeline", flush=True)
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
        print("ANALYZE 14: Gemini coaching pipeline complete", flush=True)
        logger.info("analyze: Gemini pipeline complete")

        print("ANALYZE 15: processing analysis result", flush=True)
        if not result["events_valid_json"]:
            logger.warning("analyze: stage 1 output wasn't valid JSON; raw text was passed through to stage 2")
        logger.info("analyze: analysis result received (events_valid_json=%s)", result["events_valid_json"])
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

        print("ANALYZE 16: creating response", flush=True)
        response = AnalyzeResponse(
            is_league_of_legends=True,
            classification_confidence=None if confidence is None else round(confidence, 4),
            classification_skipped=skip_classification,
            coaching=result["coaching"],
            coaching_valid_json=result["coaching_valid_json"],
            timing_seconds=timing,
        )
        print("ANALYZE 17: response created", flush=True)
        return response
    except Exception:
        print("ANALYZE ERROR: unhandled exception", flush=True)
        logger.exception("analyze: unhandled exception")
        raise
    finally:
        if source_object_key is not None:
            try:
                await run_in_threadpool(storage.delete_object, source_object_key)
            except StorageOperationError:
                logger.exception("analyze: source video cleanup failed")
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except OSError:
            logger.warning("failed to remove temp file: %s", input_path)
        for d in (analysis_frames_dir,):
            try:
                for name in os.listdir(d):
                    os.remove(os.path.join(d, name))
                os.rmdir(d)
            except OSError:
                logger.warning("failed to remove temp dir: %s", d)