"""SQLite persistence for backtest requests, results, and progress events."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from tradingagents.extensions.contracts import (
    BacktestRequest,
    BacktestResult,
    RunEvent,
)

RunStatus = Literal["RUNNING", "COMPLETED", "FAILED"]


def default_store_path() -> Path:
    return Path.home() / ".tradingagents" / "paper_trading" / "runs.sqlite3"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    created_at: datetime
    updated_at: datetime
    status: RunStatus
    label: str
    symbols: tuple[str, ...]
    start: datetime
    end: datetime
    total_return: float | None
    max_drawdown: float | None


@dataclass(frozen=True, slots=True)
class StoredRun:
    run_id: str
    created_at: datetime
    updated_at: datetime
    status: RunStatus
    label: str
    request: BacktestRequest
    result: BacktestResult | None
    events: tuple[RunEvent, ...]
    error: str | None


class SQLiteRunStore:
    """Small durable run repository with one transaction per public action."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_store_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create_run(
        self,
        request: BacktestRequest,
        *,
        label: str | None = None,
        run_id: str | None = None,
    ) -> str:
        identifier = run_id or uuid4().hex
        now = utc_now().isoformat()
        display_label = (label or self._default_label(request)).strip()
        if not display_label:
            raise ValueError("run label must not be empty")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, created_at, updated_at, status, label,
                    request_json, result_json, error
                ) VALUES (?, ?, ?, 'RUNNING', ?, ?, NULL, NULL)
                """,
                (
                    identifier,
                    now,
                    now,
                    display_label,
                    request.model_dump_json(),
                ),
            )
        return identifier

    def append_event(self, run_id: str, event: RunEvent) -> None:
        now = utc_now().isoformat()
        with self._connect() as connection:
            self._require_run(connection, run_id)
            next_sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM run_events WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO run_events (run_id, sequence, event_json) VALUES (?, ?, ?)",
                (run_id, next_sequence, event.model_dump_json()),
            )
            connection.execute(
                "UPDATE runs SET updated_at = ? WHERE run_id = ?",
                (now, run_id),
            )

    def complete_run(self, run_id: str, result: BacktestResult) -> None:
        with self._connect() as connection:
            self._require_run(connection, run_id)
            connection.execute(
                """
                UPDATE runs
                SET updated_at = ?, status = 'COMPLETED', result_json = ?, error = NULL
                WHERE run_id = ?
                """,
                (utc_now().isoformat(), result.model_dump_json(), run_id),
            )

    def fail_run(self, run_id: str, error: str) -> None:
        message = error.strip()
        if not message:
            raise ValueError("run failure must include an error message")
        with self._connect() as connection:
            self._require_run(connection, run_id)
            connection.execute(
                """
                UPDATE runs
                SET updated_at = ?, status = 'FAILED', error = ?
                WHERE run_id = ?
                """,
                (utc_now().isoformat(), message, run_id),
            )

    def save_completed(
        self,
        request: BacktestRequest,
        result: BacktestResult,
        *,
        events: tuple[RunEvent, ...] | list[RunEvent] = (),
        label: str | None = None,
        run_id: str | None = None,
    ) -> str:
        identifier = self.create_run(request, label=label, run_id=run_id)
        for event in events:
            self.append_event(identifier, event)
        self.complete_run(identifier, result)
        return identifier

    def get_run(self, run_id: str) -> StoredRun:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT run_id, created_at, updated_at, status, label,
                       request_json, result_json, error
                FROM runs WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown run {run_id}")
            event_rows = connection.execute(
                "SELECT event_json FROM run_events WHERE run_id = ? ORDER BY sequence",
                (run_id,),
            ).fetchall()
        return StoredRun(
            run_id=row["run_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            status=row["status"],
            label=row["label"],
            request=BacktestRequest.model_validate_json(row["request_json"]),
            result=(
                BacktestResult.model_validate_json(row["result_json"])
                if row["result_json"] is not None
                else None
            ),
            events=tuple(RunEvent.model_validate_json(item["event_json"]) for item in event_rows),
            error=row["error"],
        )

    def list_runs(self, *, limit: int = 100) -> list[RunSummary]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, created_at, updated_at, status, label,
                       request_json, result_json
                FROM runs ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()

        summaries: list[RunSummary] = []
        for row in rows:
            request = BacktestRequest.model_validate_json(row["request_json"])
            metrics: dict[str, Any] = {}
            if row["result_json"] is not None:
                raw_result = json.loads(row["result_json"])
                metrics = raw_result.get("metrics", {})
            summaries.append(
                RunSummary(
                    run_id=row["run_id"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    status=row["status"],
                    label=row["label"],
                    symbols=tuple(request.symbols),
                    start=request.start,
                    end=request.end,
                    total_return=metrics.get("total_return"),
                    max_drawdown=metrics.get("max_drawdown"),
                )
            )
        return summaries

    def export_run(self, run_id: str) -> dict[str, Any]:
        stored = self.get_run(run_id)
        return {
            "schema_version": 1,
            "run_id": stored.run_id,
            "created_at": stored.created_at.isoformat(),
            "updated_at": stored.updated_at.isoformat(),
            "status": stored.status,
            "label": stored.label,
            "request": stored.request.model_dump(mode="json"),
            "result": stored.result.model_dump(mode="json") if stored.result else None,
            "events": [event.model_dump(mode="json") for event in stored.events],
            "error": stored.error,
        }

    def delete_run(self, run_id: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            if cursor.rowcount == 0:
                raise KeyError(f"unknown run {run_id}")

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('RUNNING', 'COMPLETED', 'FAILED')),
                    label TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS run_events (
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, sequence),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @staticmethod
    def _require_run(connection: sqlite3.Connection, run_id: str) -> None:
        exists = connection.execute(
            "SELECT 1 FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if exists is None:
            raise KeyError(f"unknown run {run_id}")

    @staticmethod
    def _default_label(request: BacktestRequest) -> str:
        symbols = ", ".join(request.symbols)
        return f"{symbols} · {request.start.date()} → {request.end.date()}"


class RunStoreObserver:
    """Persist every progress event for one already-created run."""

    def __init__(self, store: SQLiteRunStore, run_id: str) -> None:
        self.store = store
        self.run_id = run_id

    def on_event(self, event: RunEvent) -> None:
        self.store.append_event(self.run_id, event)


__all__ = [
    "RunStatus",
    "RunStoreObserver",
    "RunSummary",
    "SQLiteRunStore",
    "StoredRun",
    "default_store_path",
]
