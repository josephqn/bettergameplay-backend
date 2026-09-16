"""
trim.py - CLI wrapper around Very Good FFmpeg for extracting a clip from a local video file.

Usage:
    uv run trim.py test.mp4 --start 30 --end 40
    uv run trim.py test.mp4 --start 30 --end 40 --output clip.mp4

Requires VERY_GOOD_API_KEY in .env.
"""

import argparse
import logging
import os
import sys
from uuid import uuid4

from very_good_service import VeryGoodVideoProcessor, validate_video_constraints
from storage_service import R2ObjectStorage, StorageConfigurationError, StorageOperationError

MAX_CLIP_DURATION = 20
MIN_CLIP_DURATION = 1

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("bettergameplay.trim")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a clip from a video between two timestamps (seconds)."
    )
    parser.add_argument("video", help="Path to the input video file (e.g. test.mp4)")
    parser.add_argument("--start", type=float, required=True, help="Start time in seconds")
    parser.add_argument("--end", type=float, required=True, help="End time in seconds")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file path (default: <video_stem>_trimmed.mp4)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    video_processor = VeryGoodVideoProcessor.from_env()
    if video_processor is None:
        logger.error("VERY_GOOD_API_KEY is not configured.")
        return 1

    video_path = args.video
    if not os.path.isfile(video_path):
        logger.error("video file not found: %s", video_path)
        return 1

    try:
        storage = R2ObjectStorage.from_env()
    except StorageConfigurationError as e:
        logger.error("source storage is not configured correctly: %s", e)
        return 1
    if storage is None:
        logger.error("source storage is not configured")
        return 1

    source_object_key = None

    def cleanup_source() -> None:
        if source_object_key is not None:
            try:
                storage.delete_object(source_object_key)
            except StorageOperationError:
                logger.exception("source video cleanup failed")

    start_time = args.start
    end_time = args.end

    if start_time < 0:
        logger.error("start time must be >= 0")
        return 1
    if end_time <= start_time:
        logger.error("end time must be greater than start time")
        return 1

    duration = end_time - start_time
    if duration > MAX_CLIP_DURATION:
        logger.error("clip duration must not exceed %s seconds", MAX_CLIP_DURATION)
        return 1
    if duration < MIN_CLIP_DURATION:
        logger.error("clip duration must be at least %s second", MIN_CLIP_DURATION)
        return 1

    try:
        source_object_key = storage.upload_file(video_path, job_id=uuid4().hex)
        source_url = storage.create_presigned_get_url(source_object_key)
        validate_video_constraints(os.path.getsize(video_path))
    except ValueError as e:
        logger.error("video rejected: %s", e)
        cleanup_source()
        return 1
    except Exception as e:
        logger.error("failed to upload or inspect video with Very Good FFmpeg: %s", e)
        cleanup_source()
        return 1

    output_path = args.output
    if not output_path:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{stem}_trimmed.mp4"

    logger.info("extracting %ss - %ss from %s", start_time, end_time, video_path)
    try:
        output_name = os.path.basename(output_path)
        result = video_processor.run_ffmpeg_command(
            input_files={"input.mp4": source_url},
            output_files=[output_name],
            ffmpeg_command=(
                f"-ss {start_time:.3f} -i inputs/input.mp4 -t {duration:.3f} "
                f"-c:v libx264 -preset veryfast -c:a aac -movflags +faststart outputs/{output_name}"
            ),
        )
        output_url = video_processor._output_url(result, output_name)
        video_processor.download_file(output_url, output_path)
    except Exception as e:
        logger.error("Very Good FFmpeg trim failed: %s", e)
        cleanup_source()
        return 1

    logger.info("clip saved to %s", output_path)
    cleanup_source()
    return 0


if __name__ == "__main__":
    sys.exit(main())
