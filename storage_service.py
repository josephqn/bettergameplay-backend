"""Private Cloudflare R2 storage for short-lived source video objects."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional
from uuid import uuid4

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger("bettergameplay.storage")


class StorageConfigurationError(RuntimeError):
    pass


class StorageOperationError(RuntimeError):
    pass


class R2ObjectStorage:
    def __init__(
        self,
        account_id: str,
        access_key_id: str,
        secret_access_key: str,
        bucket_name: str,
        endpoint: str,
        url_expiration_seconds: int = 3600,
    ) -> None:
        self.bucket_name = bucket_name
        self.endpoint = endpoint.rstrip("/")
        self.url_expiration_seconds = int(url_expiration_seconds)
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name="auto",
        )

    @classmethod
    def from_env(cls) -> Optional["R2ObjectStorage"]:
        values = {
            "account_id": os.getenv("R2_ACCOUNT_ID", "").strip(),
            "access_key_id": os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            "secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            "bucket_name": os.getenv("R2_BUCKET_NAME", "").strip(),
            "endpoint": os.getenv("R2_ENDPOINT", "").strip(),
        }
        if not any(values.values()):
            return None
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise StorageConfigurationError(
                "R2 storage configuration is incomplete. Missing: " + ", ".join(missing)
            )
        return cls(
            **values,
            url_expiration_seconds=int(os.getenv("R2_URL_EXPIRATION_SECONDS", "3600")),
        )

    def upload_file(self, file_path: str, *, job_id: Optional[str] = None) -> str:
        object_key = f"uploads/{job_id or uuid4().hex}/source.mp4"
        logger.info("STORAGE 2: uploading source video to R2")
        try:
            self.client.upload_file(
                file_path,
                self.bucket_name,
                object_key,
                ExtraArgs={"ContentType": "video/mp4"},
            )
        except (BotoCoreError, ClientError, OSError) as exc:
            logger.exception("R2 source upload failed for object %s", object_key)
            raise StorageOperationError("R2 source video upload failed.") from exc
        logger.info("STORAGE 3: source video uploaded")
        return object_key

    def create_presigned_get_url(self, object_key: str) -> str:
        try:
            url = self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": object_key},
                ExpiresIn=self.url_expiration_seconds,
            )
        except (BotoCoreError, ClientError) as exc:
            logger.exception("failed to create R2 presigned URL for object %s", object_key)
            raise StorageOperationError("Could not create an accessible source video URL.") from exc
        if not url:
            raise StorageOperationError("Could not create an accessible source video URL.")
        logger.info("STORAGE 4: generated Very Good FFmpeg-accessible input URL")
        return url

    def delete_object(self, object_key: str) -> None:
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=object_key)
        except (BotoCoreError, ClientError) as exc:
            logger.exception("STORAGE cleanup failed for object %s", object_key)
            raise StorageOperationError("R2 source video cleanup failed.") from exc
        logger.info("STORAGE 6: source video cleanup completed")