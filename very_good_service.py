"""Very Good FFmpeg adapter for remote frame extraction and video trimming."""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

logger = logging.getLogger("bettergameplay.very_good_ffmpeg")

VERY_GOOD_API_BASE_URL = "https://verygoodffmpeg.com/api"
VERY_GOOD_FRAME_DOWNLOAD_WORKERS = 8
VERY_GOOD_JOB_TIMEOUT_SECONDS = 300
VERY_GOOD_POLL_TIMEOUT_SECONDS = 900
VERY_GOOD_POLL_INTERVAL_SECONDS = 3
VERY_GOOD_MAX_COMMAND_RUN_SECONDS = 300

MAX_VIDEO_DURATION_SECONDS = 10 * 60
MAX_VIDEO_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024


def validate_video_constraints(file_size_bytes: int | float, duration_seconds: Optional[float] = None) -> None:
    if file_size_bytes > MAX_VIDEO_FILE_SIZE_BYTES:
        raise ValueError("Video must be smaller than 2 GB.")
    if duration_seconds is not None and duration_seconds > MAX_VIDEO_DURATION_SECONDS:
        raise ValueError("Video must be 10 minutes or shorter.")


class VeryGoodVideoProcessor:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        request_timeout_seconds: float = 60.0,
        poll_interval_seconds: float = VERY_GOOD_POLL_INTERVAL_SECONDS,
        poll_timeout_seconds: float = VERY_GOOD_POLL_TIMEOUT_SECONDS,
        job_timeout_seconds: int = VERY_GOOD_JOB_TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = (api_key or os.getenv("VERY_GOOD_API_KEY") or "").strip()
        self.api_base_url = (api_base_url or os.getenv("VERY_GOOD_API_BASE_URL") or VERY_GOOD_API_BASE_URL).strip().rstrip("/")
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.poll_timeout_seconds = float(poll_timeout_seconds)
        self.job_timeout_seconds = int(job_timeout_seconds)
        if not self.api_key:
            raise ValueError("VERY_GOOD_API_KEY is required for Very Good FFmpeg processing.")

    @classmethod
    def from_env(cls) -> Optional["VeryGoodVideoProcessor"]:
        api_key = os.getenv("VERY_GOOD_API_KEY", "").strip()
        if not api_key:
            return None
        return cls(
            api_key=api_key,
            api_base_url=os.getenv("VERY_GOOD_API_BASE_URL", VERY_GOOD_API_BASE_URL),
            request_timeout_seconds=float(os.getenv("VERY_GOOD_REQUEST_TIMEOUT_SECONDS", "60")),
            poll_interval_seconds=float(os.getenv("VERY_GOOD_POLL_INTERVAL_SECONDS", str(VERY_GOOD_POLL_INTERVAL_SECONDS))),
            poll_timeout_seconds=float(os.getenv("VERY_GOOD_POLL_TIMEOUT_SECONDS", str(VERY_GOOD_POLL_TIMEOUT_SECONDS))),
            job_timeout_seconds=int(os.getenv("VERY_GOOD_JOB_TIMEOUT_SECONDS", str(VERY_GOOD_JOB_TIMEOUT_SECONDS))),
        )

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "BetterGameplay/1.0 (Very Good FFmpeg API client)",
        }

    def _request_json(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.api_base_url}{path if path.startswith('/') else '/' + path}"
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib_request.Request(url, data=body, headers=self._headers(), method=method)
        try:
            with urllib_request.urlopen(request, timeout=self.request_timeout_seconds) as response:
                response_body = response.read()
        except urllib_error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            message = "Very Good FFmpeg authentication failed" if exc.code in (401, 403) else f"Very Good FFmpeg request failed ({exc.code})"
            logger.error("%s: %s", message, details or exc.reason)
            raise RuntimeError(f"{message}: {details or exc.reason}") from exc
        except urllib_error.URLError as exc:
            logger.exception("Very Good FFmpeg API connectivity error for %s", url)
            raise RuntimeError(f"Very Good FFmpeg API unreachable: {exc.reason}") from exc
        if not response_body:
            return {}
        try:
            result = json.loads(response_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Malformed Very Good FFmpeg response from API.") from exc
        if not isinstance(result, dict):
            raise RuntimeError("Unexpected Very Good FFmpeg response shape.")
        return result

    def download_file(self, url: str, output_path: str) -> None:
        request = urllib_request.Request(url, headers={"User-Agent": "BetterGameplay/1.0"})
        try:
            with urllib_request.urlopen(request, timeout=self.request_timeout_seconds) as response:
                with open(output_path, "wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
        except (urllib_error.URLError, OSError) as exc:
            raise RuntimeError(f"Failed to download Very Good FFmpeg output {url}") from exc

    def run_ffmpeg_command(
        self,
        input_files: Dict[str, str],
        ffmpeg_command: str,
        output_files: List[str],
        max_command_run_seconds: int = VERY_GOOD_MAX_COMMAND_RUN_SECONDS,
    ) -> Dict[str, Any]:
        payload = {
            "input_files": input_files,
            "output_files": output_files,
            "ffmpeg_commands": [ffmpeg_command],
            "machine": "cpu",
            "timeout_seconds": min(int(max_command_run_seconds), self.job_timeout_seconds),
        }
        logger.info("Very Good FFmpeg command: %s", ffmpeg_command)
        logger.info("Very Good FFmpeg outputs: %d", len(output_files))
        submit_start = time.perf_counter()
        response = self._request_json("POST", "/ffmpeg?wait=false", payload)
        submission_seconds = time.perf_counter() - submit_start
        data = response.get("data")
        if not isinstance(data, dict) or not data.get("id"):
            raise RuntimeError(f"Malformed Very Good FFmpeg job response: {response}")
        job_id = str(data["id"])
        logger.info("VERY_GOOD 1: job submitted id=%s submission_seconds=%.3f", job_id, submission_seconds)
        return self.poll_job(job_id)

    def poll_job(self, job_id: str, timeout_seconds: Optional[float] = None) -> Dict[str, Any]:
        deadline = time.monotonic() + (timeout_seconds if timeout_seconds is not None else self.poll_timeout_seconds)
        while True:
            response = self._request_json("GET", f"/jobs/{job_id}")
            data = response.get("data")
            if not isinstance(data, dict):
                raise RuntimeError(f"Malformed Very Good FFmpeg status response: {response}")
            status = str(data.get("status") or "").lower()
            logger.info("VERY_GOOD 2: polling job %s -> %s", job_id, status)
            if status == "succeeded":
                logger.info("VERY_GOOD 3: processing completed job=%s", job_id)
                return data
            if status in {"failed", "cancelled"}:
                raise RuntimeError(f"Very Good FFmpeg job {status}: {data.get('error_message') or data}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Very Good FFmpeg job timed out after {timeout_seconds or self.poll_timeout_seconds} seconds.")
            time.sleep(self.poll_interval_seconds)

    @staticmethod
    def _output_url(job_result: Dict[str, Any], output_name: str) -> str:
        output_files = job_result.get("output_files")
        if not isinstance(output_files, dict):
            raise RuntimeError("Very Good FFmpeg completed without output files.")
        output_url = output_files.get(output_name)
        if not isinstance(output_url, str) or not output_url:
            raise RuntimeError(f"Very Good FFmpeg output is missing: {output_name}")
        return output_url

    def extract_frames(
        self,
        source_url: str,
        *,
        start_seconds: float,
        duration_seconds: float,
        fps: Optional[float] = None,
        frame_count: Optional[int] = None,
        destination_dir: Optional[str] = None,
        max_width: Optional[int] = None,
    ) -> List[str]:
        if destination_dir is None:
            destination_dir = os.path.join(os.getcwd(), "tmp_very_good_frames")
        os.makedirs(destination_dir, exist_ok=True)
        if fps is None and frame_count is None:
            raise ValueError("Either fps or frame_count must be provided.")

        requested_count = frame_count or max(1, int(round(float(duration_seconds) * float(fps))))
        effective_fps = float(requested_count) / float(duration_seconds) if duration_seconds > 0 else 0.0
        scale_filter = f",scale='min(iw,{int(max_width)})':-2" if max_width else ""
        output_names = [f"frame_{index:04d}.jpg" for index in range(1, requested_count + 1)]
        split_labels = "".join(f"[split{index}]" for index in range(requested_count))
        split_outputs = ";".join(
            f"[split{index}]select='eq(n,{index})'[frame{index}]"
            for index in range(requested_count)
        )
        maps = " ".join(
            f"-map [frame{index}] -frames:v 1 -q:v 3 outputs/{output_names[index]}"
            for index in range(requested_count)
        )
        ffmpeg_command = (
            f"-ss {float(start_seconds):.3f} -i inputs/input.mp4 -t {float(duration_seconds):.3f} "
            f"-filter_complex \"[0:v]fps={effective_fps}{scale_filter},split={requested_count}"
            f"{split_labels};{split_outputs}\" {maps}"
        )

        command_start = time.perf_counter()
        command_result = self.run_ffmpeg_command(
            input_files={"input.mp4": source_url},
            ffmpeg_command=ffmpeg_command,
            output_files=output_names,
            max_command_run_seconds=VERY_GOOD_MAX_COMMAND_RUN_SECONDS,
        )
        logger.info("perf very_good: processing_seconds=%.3f requested_frames=%d", time.perf_counter() - command_start, requested_count)

        download_start = time.perf_counter()

        def download_frame(output_name: str) -> str:
            frame_path = os.path.join(destination_dir, output_name)
            self.download_file(self._output_url(command_result, output_name), frame_path)
            return frame_path

        with ThreadPoolExecutor(max_workers=min(VERY_GOOD_FRAME_DOWNLOAD_WORKERS, requested_count)) as executor:
            frame_paths = list(executor.map(download_frame, output_names))
        retrieval_seconds = time.perf_counter() - download_start
        logger.info(
            "perf very_good: output_retrieval_seconds=%.3f frame_preparation_seconds=%.3f frames=%d workers=%d",
            retrieval_seconds, retrieval_seconds, len(frame_paths), min(VERY_GOOD_FRAME_DOWNLOAD_WORKERS, requested_count),
        )
        if len(frame_paths) != requested_count:
            raise RuntimeError("Very Good FFmpeg returned an unexpected number of frame outputs.")
        return frame_paths

    def build_trim_command(self, start_seconds: float, duration_seconds: float, output_name: str = "trimmed.mp4") -> Dict[str, Any]:
        return {
            "input_files": {"input.mp4": "source_url"},
            "output_files": [output_name],
            "ffmpeg_commands": [
                f"-ss {float(start_seconds):.3f} -i inputs/input.mp4 -t {float(duration_seconds):.3f} "
                f"-c:v libx264 -preset veryfast -c:a aac -movflags +faststart outputs/{output_name}"
            ],
        }