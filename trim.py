# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""
trim.py - CLI wrapper around video_utils.trim_clip for extracting a clip from a local video
file. Handy for testing without going through the API. The FastAPI app trims internally as
part of /api/video/analyze (see analyze.py) and does not mount this as a route.

Usage:
    uv run trim.py test.mp4 --start 30 --end 40
    uv run trim.py test.mp4 --start 30 --end 40 --output clip.mp4

Requires ffmpeg to be installed and available on PATH.
"""

import argparse
import logging
import os
import sys

from video_utils import FFmpegError, check_ffmpeg_available, trim_clip

MAX_CLIP_DURATION = 20
MIN_CLIP_DURATION = 1

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("gamesense.trim")


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

    if not check_ffmpeg_available():
        logger.error("ffmpeg not found on PATH. Install it and try again.")
        return 1

    video_path = args.video
    if not os.path.isfile(video_path):
        logger.error("video file not found: %s", video_path)
        return 1

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

    output_path = args.output
    if not output_path:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{stem}_trimmed.mp4"

    logger.info("extracting %ss - %ss from %s", start_time, end_time, video_path)
    try:
        trim_clip(video_path, output_path, start_time, end_time)
    except FFmpegError as e:
        logger.error("ffmpeg failed: %s", e)
        return 1

    logger.info("clip saved to %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
