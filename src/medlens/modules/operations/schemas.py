from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import ConfigDict

from medlens.core.schemas import ReadModel


class FailedTaskRead(ReadModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0ea0",
                    "task_name": "medlens.scans.process",
                    "task_id": "c5b0c9a2-7d0e-4c55-9d7e-3f1a2b3c4d5e",
                    "args": ["0190a8d4-6f5e-7c3a-9b1e-2f4d6a8c0e80"],
                    "kwargs": {},
                    "error_type": "StorageUnavailableError",
                    "error_message": "ConnectTimeoutError",
                    "failure_count": 1,
                    "first_failed_at": "2026-11-03T09:10:00Z",
                    "last_failed_at": "2026-11-03T09:10:00Z",
                    "requeued_at": None,
                    "requeue_count": 0,
                }
            ]
        }
    )

    id: UUID
    task_name: str
    task_id: str | None
    args: list[Any]
    kwargs: dict[str, Any]
    error_type: str
    error_message: str | None
    failure_count: int
    first_failed_at: datetime
    last_failed_at: datetime
    requeued_at: datetime | None
    requeue_count: int
