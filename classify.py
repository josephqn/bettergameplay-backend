import logging
import os
import tempfile
import time
from typing import List, Tuple

from fastapi import APIRouter, File, HTTPException, UploadFile

from video_utils import FFmpegError, extract_frames_count, get_duration
from gemini_pipeline import DEFAULT_MODEL, build_client, call_gemini_frames, parse_json_block

NUM_FRAMES = 5
LOL_THRESHOLD = 0.5
CHUNK_SIZE = 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bettergameplay.classify")

router = APIRouter()

def _cleanup(*paths: str) -> None:
    for path in paths:
        try:
            if path and os.path.exists(path):
                if os.path.isdir(path):
                    for name in os.listdir(path):
                        os.remove(os.path.join(path, name))
                    os.rmdir(path)
                else:
                    os.remove(path)
        except OSError:
            logger.warning("failed to remove temp path: %s", path)


def classify_frames(frame_paths: List[str]) -> Tuple[float, List[float]]:
    """Classify frames with Gemini and return mean and per-frame LoL confidence."""
    prompt = """
You are classifying video frames by game. Analyze every image supplied and return only valid JSON.

For each frame, estimate the probability from 0.0 to 1.0 that it shows League of Legends gameplay.
League of Legends gameplay is a top-down MOBA view with recognizable League UI such as a minimap,
champion ability bar, health or mana bars, items, or lane/jungle terrain. Menus, loading screens,
other games, unrelated images, and ambiguous frames should receive a low probability.

Return exactly this JSON shape, with one value in per_frame_scores for each image in input order:
{"per_frame_scores": [0.0], "confidence": 0.0}

confidence must be the arithmetic mean of per_frame_scores. Do not include markdown or explanation.
""".strip()
    client = build_client()
    raw_result = call_gemini_frames(client, DEFAULT_MODEL, prompt, frame_paths)
    result, valid_json = parse_json_block(raw_result)
    if not valid_json or not isinstance(result, dict):
        raise RuntimeError("classification model returned invalid JSON")

    raw_scores = result.get("per_frame_scores")
    if not isinstance(raw_scores, list) or len(raw_scores) != len(frame_paths):
        raise RuntimeError("classification model returned an invalid frame score count")
    try:
        per_frame = [round(max(0.0, min(1.0, float(score))), 4) for score in raw_scores]
    except (TypeError, ValueError) as exc:
        raise RuntimeError("classification model returned invalid frame scores") from exc

    confidence = round(sum(per_frame) / len(per_frame), 4)
    return confidence, per_frame


@router.post("/api/video/classify")
async def classify_video(video: UploadFile = File(None)):
    request_start = time.perf_counter()

    if video is None or not video.filename:
        raise HTTPException(status_code=400, detail="video is required")

    input_suffix = os.path.splitext(video.filename)[1] or ".mp4"
    input_fd, input_path = tempfile.mkstemp(suffix=input_suffix)
    frames_dir = tempfile.mkdtemp()

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
        _cleanup(input_path, frames_dir)
        raise HTTPException(status_code=500, detail="failed to save uploaded video")
    upload_time = time.perf_counter() - upload_start

    try:
        duration = get_duration(input_path)
    except FFmpegError:
        _cleanup(input_path, frames_dir)
        raise HTTPException(status_code=500, detail="failed to read video")

    extract_start = time.perf_counter()
    frame_paths = extract_frames_count(input_path, frames_dir, NUM_FRAMES, duration)
    extract_time = time.perf_counter() - extract_start

    if not frame_paths:
        _cleanup(input_path, frames_dir)
        raise HTTPException(status_code=500, detail="failed to extract frames")

    classify_start = time.perf_counter()
    try:
        confidence, per_frame_scores = classify_frames(frame_paths)
    except Exception:
        logger.exception("classification failed")
        _cleanup(input_path, frames_dir)
        raise HTTPException(status_code=500, detail="failed to classify video")
    classify_time = time.perf_counter() - classify_start

    _cleanup(input_path, frames_dir)

    total_time = time.perf_counter() - request_start

    logger.info(
        "classify: total=%.2fs upload=%.2fs extract=%.2fs (%d frames) infer=%.2fs "
        "per_frame_scores=%s result=%.4f",
        total_time, upload_time, extract_time, len(frame_paths), classify_time,
        per_frame_scores, confidence,
    )

    return {
        "is_league_of_legends": confidence >= LOL_THRESHOLD,
        "confidence": round(confidence, 4),
        "frames_analyzed": len(frame_paths),
        "per_frame_scores": per_frame_scores,
        "timing_seconds": {
            "total": round(total_time, 3),
            "upload": round(upload_time, 3),
            "frame_extraction": round(extract_time, 3),
            "inference": round(classify_time, 3),
        },
    }