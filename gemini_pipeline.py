"""gemini_pipeline.py - two-stage Gemini pipeline: structured event extraction from frames
(vision), then a text-only coaching pass over those events. Split out of analyze_clip.py so
the FastAPI endpoint (analyze.py) and the standalone CLI (analyze_clip.py) share one
implementation.
"""

import json
import logging
import os
import time
from typing import Dict, List, Tuple

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from prompts import event_detection_prompt, coaching_prompt

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_TIMEOUT_SECONDS = 120.0

logger = logging.getLogger("bettergameplay.gemini_pipeline")


class GeminiRateLimitError(RuntimeError):
    """Raised when the Gemini API rejects a call with 429 RESOURCE_EXHAUSTED
    (quota/rate limit exceeded). Callers can catch this separately from other
    Gemini failures to return a proper 429 instead of a generic error."""

# genai.Client holds an HTTP connection pool; building a new one per request means a fresh
# TLS handshake every time. Cache and reuse clients across calls, keyed by timeout (the only
# thing that varies client construction) so callers that pass a non-default --timeout still
# get their own client instead of silently inheriting someone else's.
_client_cache: Dict[float, genai.Client] = {}


def build_client(timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> genai.Client:
    cached = _client_cache.get(timeout_seconds)
    if cached is not None:
        return cached

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Get a key from https://aistudio.google.com/apikey"
        )
    logger.info("building Gemini client (timeout=%.0fs)", timeout_seconds)
    # NOTE: HttpOptions.timeout is in milliseconds, and community reports suggest Google's
    # SDK doesn't always enforce it perfectly - treat this as a best-effort safeguard, not a
    # hard guarantee.
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
    )
    _client_cache[timeout_seconds] = client
    return client


def call_gemini_text(client: genai.Client, model: str, prompt: str) -> str:
    logger.info("calling Gemini text model '%s' (prompt length=%d chars)...", model, len(prompt))
    call_start = time.perf_counter()
    try:
        response = client.models.generate_content(model=model, contents=[prompt])
    except genai_errors.ClientError as e:
        if e.code == 429:
            logger.warning("Gemini text model '%s' rate limited: %s", model, e.message or e)
            raise GeminiRateLimitError(e.message or str(e)) from e
        raise
    logger.info("Gemini text model '%s' responded in %.2fs", model, time.perf_counter() - call_start)
    if not response.text:
        raise RuntimeError(f"model '{model}' returned no text content")
    return response.text


def call_gemini_frames(client: genai.Client, model: str, prompt: str, frame_paths: List[str]) -> str:
    logger.info("calling Gemini vision model '%s' with %d frames...", model, len(frame_paths))
    parts = [prompt]
    image_bytes = 0
    for fp in frame_paths:
        with open(fp, "rb") as f:
            image_data = f.read()
            image_bytes += len(image_data)
            parts.append(types.Part.from_bytes(data=image_data, mime_type="image/jpeg"))
    logger.info(
        "perf gemini: vision request model='%s' frames=%d image_bytes=%d prompt_chars=%d",
        model, len(frame_paths), image_bytes, len(prompt),
    )
    call_start = time.perf_counter()
    try:
        response = client.models.generate_content(model=model, contents=parts)
    except genai_errors.ClientError as e:
        if e.code == 429:
            logger.warning("Gemini vision model '%s' rate limited: %s", model, e.message or e)
            raise GeminiRateLimitError(e.message or str(e)) from e
        raise
    logger.info("Gemini vision model '%s' responded in %.2fs", model, time.perf_counter() - call_start)
    if not response.text:
        raise RuntimeError(f"model '{model}' returned no text content")
    return response.text


def parse_json_block(raw_text: str) -> Tuple[object, bool]:
    """Strip optional markdown code fences and try to parse raw_text as JSON.

    Returns (parsed_object, True) on success -- parsed_object is a real dict/list, not a
    string -- or (raw_text.strip(), False) if it isn't valid JSON."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned), True
    except json.JSONDecodeError:
        return raw_text.strip(), False


def parse_events_json(raw_text: str) -> Tuple[str, bool]:
    """Try to parse the stage-1 output as JSON and return a pretty-printed *string* version.
    Kept as a string (rather than the parsed object) because it's spliced straight into the
    stage-2 coaching prompt, which wants readable text. Returns (text_to_use, was_valid_json)."""
    parsed, valid = parse_json_block(raw_text)
    if not valid:
        return parsed, False
    return json.dumps(parsed, indent=2), True


def run_coaching_pipeline(
    frame_paths: List[str],
    fps: float,
    vision_model: str = DEFAULT_MODEL,
    text_model: str = DEFAULT_MODEL,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, object]:
    """Run stage 1 (vision -> structured events) then stage 2 (text -> coaching).

    Returns {"events": str, "events_valid_json": bool, "coaching": dict | list | str,
    "coaching_valid_json": bool, "event_extraction_seconds": float, "coaching_seconds": float}.
    `coaching` is a parsed JSON object (dict/list) when the model's output was valid JSON --
    the common case, since the prompt asks for structured JSON -- and falls back to the raw
    response text if it wasn't, with coaching_valid_json indicating which happened.
    """
    client = build_client(timeout_seconds)

    logger.info("stage 1: extracting structured events from %d frames (model: %s)...", len(frame_paths), vision_model)
    stage1_start = time.perf_counter()
    raw_events = call_gemini_frames(client, vision_model, event_detection_prompt(fps), frame_paths)
    events_text, was_valid_json = parse_events_json(raw_events)
    stage1_seconds = time.perf_counter() - stage1_start
    logger.info("stage 1 complete in %.2fs (valid_json=%s)", stage1_seconds, was_valid_json)

    logger.info("stage 2: generating coaching feedback (model: %s)...", text_model)
    stage2_start = time.perf_counter()
    raw_coaching = call_gemini_text(client, text_model, coaching_prompt(events=events_text))
    coaching, coaching_valid_json = parse_json_block(raw_coaching)
    stage2_seconds = time.perf_counter() - stage2_start
    logger.info("stage 2 complete in %.2fs (valid_json=%s)", stage2_seconds, coaching_valid_json)

    return {
        "events": events_text,
        "events_valid_json": was_valid_json,
        "coaching": coaching,
        "coaching_valid_json": coaching_valid_json,
        "event_extraction_seconds": round(stage1_seconds, 3),
        "coaching_seconds": round(stage2_seconds, 3),
    }