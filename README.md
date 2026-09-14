# Run api locally

uv run uvicorn main:app --reload

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
