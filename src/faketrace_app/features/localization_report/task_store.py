from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

from ...core.paths import PROJECT_ROOT


DB_PATH = PROJECT_ROOT / "instance" / "faketrace.sqlite3"


@dataclass(frozen=True)
class LocalizationTask:
    id: int
    test_id: str
    created_at: str
    model: str
    image_count: int


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS localization_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            model TEXT NOT NULL,
            image_count INTEGER NOT NULL
        )
        """
    )


def format_localization_test_id(task_id: int) -> str:
    return f"L-{task_id:06d}"


def create_localization_task(model: str, image_count: int) -> LocalizationTask:
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    connection = _connect()
    try:
        _ensure_schema(connection)
        cursor = connection.execute(
            "INSERT INTO localization_tasks (created_at, model, image_count) VALUES (?, ?, ?)",
            (created_at, model, image_count),
        )
        task_id = int(cursor.lastrowid)
        connection.commit()
    finally:
        connection.close()

    return LocalizationTask(
        id=task_id,
        test_id=format_localization_test_id(task_id),
        created_at=created_at,
        model=model,
        image_count=image_count,
    )
