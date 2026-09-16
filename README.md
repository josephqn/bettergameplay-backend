# Run the API locally

uv run uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Very Good FFmpeg configuration

# Copy .env.example to .env and fill in the API keys before using remote video processing.

# Required for Very Good FFmpeg-backed frame processing:

# VERY_GOOD_API_KEY

# Optional; defaults to https://verygoodffmpeg.com/api

# VERY_GOOD_API_BASE_URL=https://verygoodffmpeg.com/api

# Very Good FFmpeg accepts FFmpeg commands. The source MP4 is never uploaded or

# stored permanently by Very Good; it is supplied through a temporary Cloudflare R2 HTTPS URL.

# Cloudflare R2 configuration

# Create a private R2 bucket and an R2 API token with object read/write/delete

# access scoped to that bucket. Set R2_ENDPOINT to:

# https://<account-id>.r2.cloudflarestorage.com

# R2_ACCOUNT_ID=

# R2_ACCESS_KEY_ID=

# R2_SECRET_ACCESS_KEY=

# R2_BUCKET_NAME=

# R2_ENDPOINT=

# R2_URL_EXPIRATION_SECONDS=3600

# Processing flow:

# Browser -> Render -> private R2 object -> time-limited HTTPS presigned URL -> Very Good FFmpeg.

# Very Good FFmpeg accepts publicly reachable HTTPS input URLs; the presigned URL is temporary,

# so the source video does not need to be public. The URL is not logged.

# The R2 source object is deleted after analysis, including failed processing attempts.

# Configure an R2 lifecycle rule to expire uploads/\* after 24 hours as crash recovery.

# All video probing, trimming, and frame extraction runs through Very Good FFmpeg.

# Render/local Python only stages uploads and runs Gemini analysis.

#

# Product limits enforced before expensive work starts:

# Video must be 10 minutes or shorter.

# Video must be smaller than 2 GB.

# Health check:

curl http://127.0.0.1:8000/health

uv run roboflow_champion_detector.py "League of Legends 2026.08.29 - 00.57.53.02.DVR_trimmed.mp4" --confidence 0.15 --frames 20

Example analysis:

```json
{
  "is_league_of_legends": true,
  "classification_confidence": 0.9865,
  "classification_skipped": false,
  "coaching": {
    "good_plays": [
      {
        "text": "There wasn't much positive counterplay to highlight in this clip, as you were quickly caught out after moving up the lane.",
        "start_frame": 0,
        "end_frame": 6
      }
    ],
    "mistakes": [
      {
        "text": "You walked up the lane into the first enemy champion's range, allowing them to land a stun, follow up with attacks and damage, and ultimately secure a kill.",
        "start_frame": 8,
        "end_frame": 16
      }
    ],
    "action_items": {
      "keep_doing_this": "Continue actively moving around the map to contest space in the lane.",
      "focus_on_this_next": "Respect enemy engagement ranges when moving forward so you don't walk directly into crowd control."
    }
  },
  "coaching_valid_json": true,
  "timing_seconds": {
    "upload": 1.339,
    "classify": 4.598,
    "frame_extraction": 2.55,
    "event_extraction": 7.09,
    "coaching": 1.151,
    "total": 20.668
  }
}
```
