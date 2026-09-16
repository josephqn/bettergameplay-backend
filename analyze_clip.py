# /// script
# requires-python = ">=3.9"
# dependencies = ["google-genai", "python-dotenv"]
# ///
"""
analyze_clip.py - CLI wrapper: validate a short video clip, extract frames from it, and run it
through the two-stage Gemini coaching pipeline in gemini_pipeline.py. Handy for testing the
pipeline directly without going through the API. The FastAPI app runs the same pipeline as
part of /api/video/analyze (see analyze.py).

Usage:
    # .env file with GEMINI_API_KEY=... in the current directory, or:
    export GEMINI_API_KEY=your_key_here
    uv run analyze_clip.py clip.mp4 --fps 2
    uv run analyze_clip.py clip.mp4 --fps 2 --keep-frames

Requires VERY_GOOD_API_KEY and a GEMINI_API_KEY environment variable
(get one at https://aistudio.google.com/apikey).
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from uuid import uuid4

from dotenv import find_dotenv, load_dotenv

from very_good_service import VeryGoodVideoProcessor, validate_video_constraints
from storage_service import R2ObjectStorage, StorageConfigurationError, StorageOperationError
from gemini_pipeline import DEFAULT_MODEL, run_coaching_pipeline

load_dotenv(find_dotenv(usecwd=True))  # reads a .env file in the current working directory (if present)

MAX_DURATION_SECONDS = 20.0
DEFAULT_FPS = 2.0

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("analyze_clip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a <=20s clip, extract structured events, then get coaching feedback via Gemini."
    )
    parser.add_argument("video", help="Path to the input video (must be at most 20 seconds)")
    parser.add_argument(
        "--fps", type=float, default=DEFAULT_FPS,
        help=f"Frames per second to sample from the clip (default {DEFAULT_FPS})",
    )
    parser.add_argument(
        "--duration", type=float, required=True,
        help="Source video duration in seconds, measured before upload",
    )
    parser.add_argument(
        "--vision-model", default=DEFAULT_MODEL,
        help=f"Gemini model id for event extraction (stage 1, needs image input). Default {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--text-model", default=DEFAULT_MODEL,
        help=f"Gemini model id for coaching (stage 2, text-only). Default {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0,
        help="Max seconds to wait for each Gemini call before giving up (default 30).",
    )
    parser.add_argument(
        "--frames-dir", default=None,
        help="Directory to save extracted frames. If omitted, a temp dir is used and deleted "
             "afterward unless --keep-frames is set.",
    )
    parser.add_argument(
        "--keep-frames", action="store_true",
        help="Don't delete extracted frames after the run (only affects the auto-created temp dir; "
             "an explicit --frames-dir is always kept).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    video_processor = VeryGoodVideoProcessor.from_env()
    if video_processor is None:
        logger.error("VERY_GOOD_API_KEY is not configured.")
        return 1
    if not os.path.isfile(args.video):
        logger.error("video file not found: %s", args.video)
        return 1

    try:
        storage = R2ObjectStorage.from_env()
    except StorageConfigurationError as e:
        logger.error("source storage is not configured correctly: %s", e)
        return 1
    if storage is None:
        logger.error("source storage is not configured")
        return 1

    user_supplied_frames_dir = args.frames_dir is not None
    frames_dir = args.frames_dir or tempfile.mkdtemp(prefix="analyze_clip_frames_")
    os.makedirs(frames_dir, exist_ok=True)
    source_object_key = None

    def cleanup_source() -> None:
        if source_object_key is not None:
            try:
                storage.delete_object(source_object_key)
            except StorageOperationError:
                logger.exception("source video cleanup failed")

    try:
        source_object_key = storage.upload_file(args.video, job_id=uuid4().hex)
        source_url = storage.create_presigned_get_url(source_object_key)
        duration = args.duration
        validate_video_constraints(os.path.getsize(args.video), duration)
    except ValueError as e:
        logger.error("video rejected: %s", e)
        cleanup_source()
        return 1
    except Exception as e:
        logger.error("failed to upload or inspect video with Very Good FFmpeg: %s", e)
        cleanup_source()
        return 1

    logger.info("video duration: %.2fs", duration)
    if duration > MAX_DURATION_SECONDS:
        logger.error(
            "video is %.2fs, which exceeds the %.0fs limit. Trim it first.",
            duration, MAX_DURATION_SECONDS,
        )
        cleanup_source()
        return 1

    total_start = time.perf_counter()
    try:
        logger.info("extracting frames at %s fps...", args.fps)
        extract_start = time.perf_counter()
        frame_paths = video_processor.extract_frames(
            source_url,
            start_seconds=0.0,
            duration_seconds=duration,
            fps=args.fps,
            destination_dir=frames_dir,
            max_width=1024,
        )
        extract_seconds = time.perf_counter() - extract_start
        if not frame_paths:
            logger.error("no frames were extracted")
            return 1
        logger.info("extracted %d frames to %s (%.2fs)", len(frame_paths), frames_dir, extract_seconds)

        logger.info(
            "running coaching pipeline (vision: %s, text: %s)...",
            args.vision_model, args.text_model,
        )
        try:
            result = run_coaching_pipeline(
                frame_paths, args.fps, args.vision_model, args.text_model, args.timeout,
            )
        except Exception as e:
            logger.error("coaching pipeline failed: %s", e)
            return 1

        if not result["events_valid_json"]:
            logger.warning("stage 1 output wasn't valid JSON; raw text was passed through to stage 2")

        total_seconds = time.perf_counter() - total_start

        print("\n" + "=" * 60)
        print("EXTRACTED EVENTS")
        print("=" * 60)
        print(result["events"])

        print("\n" + "=" * 60)
        print("COACH ANALYSIS")
        print("=" * 60)
        coaching = result["coaching"]
        print(json.dumps(coaching, indent=2) if isinstance(coaching, (dict, list)) else coaching)
        print("=" * 60)

        print(f"\nStage 1 model (extraction): {args.vision_model}")
        print(f"Stage 2 model (coaching):   {args.text_model}")
        print(
            f"Timing: frame extraction {extract_seconds:.2f}s | "
            f"event extraction {result['event_extraction_seconds']:.2f}s | "
            f"coaching {result['coaching_seconds']:.2f}s | total {total_seconds:.2f}s"
        )

        return 0
    finally:
        cleanup_source()
        if not user_supplied_frames_dir:
            if args.keep_frames:
                logger.info("frames kept at %s", frames_dir)
            else:
                shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())