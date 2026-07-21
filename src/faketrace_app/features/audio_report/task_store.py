from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime

from ...core.paths import PROJECT_ROOT


DB_PATH = PROJECT_ROOT / "instance" / "faketrace.sqlite3"


@dataclass(frozen=True)
class AudioTask:
    id: int
    test_id: str
    created_at: str
    model: str
    audio_count: int


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS audio_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            model TEXT NOT NULL,
            audio_count INTEGER NOT NULL
        )
        """
    )


def format_audio_test_id(task_id: int) -> str:
    return f"A-{task_id:06d}"


def create_audio_task(model: str, audio_count: int) -> AudioTask:
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    connection = _connect()
    try:
        _ensure_schema(connection)
        cursor = connection.execute(
            "INSERT INTO audio_tasks (created_at, model, audio_count) VALUES (?, ?, ?)",
            (created_at, model, audio_count),
        )
        task_id = int(cursor.lastrowid)
        connection.commit()
    finally:
        connection.close()

    return AudioTask(
        id=task_id,
        test_id=format_audio_test_id(task_id),
        created_at=created_at,
        model=model,
        audio_count=audio_count,
    )
