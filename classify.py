import logging
import os
import tempfile
import time
from typing import List, Tuple

import torch
import open_clip
from PIL import Image
from fastapi import APIRouter, File, HTTPException, UploadFile

from video_utils import FFmpegError, extract_frames_count, get_duration

MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"
NUM_FRAMES = 5
LOL_THRESHOLD = 0.5
CHUNK_SIZE = 1024 * 1024

CANDIDATE_LABELS = [
    "a screenshot of a League of Legends match, top-down view with a minimap and ability icons",
    "a screenshot of a first-person shooter video game",
    "a screenshot of Fortnite gameplay",
    "a screenshot of Rocket League gameplay",
]
LOL_INDEX = 0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gamesense.classify")

router = APIRouter()

device = "cuda" if torch.cuda.is_available() else "cpu"

_model_load_start = time.perf_counter()
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
model.to(device)
model.eval()

with torch.no_grad():
    text_tokens = tokenizer(CANDIDATE_LABELS).to(device)
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

logger.info(
    "CLIP model '%s' loaded on %s in %.2fs",
    MODEL_NAME, device, time.perf_counter() - _model_load_start,
)


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
    """Run CLIP zero-shot classification over the given frames and return
    (mean League-of-Legends confidence, per-frame scores). Shared by the standalone
    /api/video/classify endpoint and the combined /api/video/analyze endpoint."""
    images = [preprocess(Image.open(p).convert("RGB")) for p in frame_paths]
    batch = torch.stack(images).to(device)

    with torch.no_grad():
        image_features = model.encode_image(batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    lol_scores = similarity[:, LOL_INDEX]
    per_frame = [round(s, 4) for s in lol_scores.tolist()]
    return float(lol_scores.mean()), per_frame


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