"""S3-compatible object storage clients (AWS S3, Cloudflare R2, MinIO): aiobotocore for the API
process, boto3 for workers and command-line tools. ``adapters/storage.py`` builds the StoragePort
adapters on them."""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any, Protocol

import aioboto3
import boto3
from botocore.config import Config

from medlens.core.config import S3Settings


class S3Client(Protocol):
    """The part of the aiobotocore S3 client we use."""

    async def head_bucket(self, *, Bucket: str) -> Mapping[str, Any]: ...  # noqa: N803 (boto name)

    async def put_object(self, **kwargs: Any) -> Mapping[str, Any]: ...

    async def get_object(self, **kwargs: Any) -> Mapping[str, Any]: ...

    async def delete_object(self, **kwargs: Any) -> Mapping[str, Any]: ...


def _config(settings: S3Settings) -> Config:
    return Config(
        region_name=settings.region,
        connect_timeout=settings.connect_timeout_seconds,
        read_timeout=settings.read_timeout_seconds,
        retries={"max_attempts": settings.max_attempts, "mode": "standard"},
        s3={"addressing_style": settings.addressing_style},
    )


def _credentials(settings: S3Settings) -> dict[str, Any]:
    return {
        "endpoint_url": settings.endpoint_url,
        "aws_access_key_id": settings.access_key_id,
        "aws_secret_access_key": settings.secret_access_key.get_secret_value(),
        "config": _config(settings),
    }


@asynccontextmanager
async def open_s3_client(settings: S3Settings) -> AsyncIterator[S3Client]:
    session = aioboto3.Session()
    async with session.client("s3", **_credentials(settings)) as client:
        yield client


def create_sync_s3_client(settings: S3Settings) -> Any:
    """A boto3 S3 client (thread-safe; create one per process)."""
    return boto3.client("s3", **_credentials(settings))


async def check_bucket(client: S3Client, bucket: str) -> None:
    await client.head_bucket(Bucket=bucket)
