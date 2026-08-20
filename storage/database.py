import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path("data/clair.db")


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS datasets (
                id                TEXT PRIMARY KEY,
                session_id        TEXT NOT NULL REFERENCES sessions(id),
                original_filename TEXT NOT NULL,
                file_path         TEXT NOT NULL,
                row_count         INTEGER,
                column_count      INTEGER,
                created_at        TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS analyses (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                dataset_id  TEXT,
                type        TEXT NOT NULL,
                summary     TEXT,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS artifacts (
                id            TEXT PRIMARY KEY,
                session_id    TEXT NOT NULL,
                dataset_id    TEXT,
                analysis_id   TEXT,
                artifact_type TEXT NOT NULL,
                filename      TEXT NOT NULL,
                description   TEXT,
                file_path     TEXT NOT NULL,
                created_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_turns (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT NOT NULL,
                message_json TEXT NOT NULL,
                created_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS traces (
                id                  TEXT PRIMARY KEY,
                session_id          TEXT NOT NULL,
                run_id              TEXT NOT NULL,
                step_index          INTEGER NOT NULL,
                user_message        TEXT,
                llm_decision        TEXT NOT NULL,
                tool_name           TEXT,
                tool_args           TEXT,
                tool_result_status  TEXT,
                tool_result_summary TEXT,
                execution_time_ms   REAL,
                artifact_ids        TEXT,
                final_response      TEXT,
                created_at          TEXT NOT NULL
            );
        """)


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── Sessions ──────────────────────────────────────────────────────────────────

def create_session(session_id: str) -> str:
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, created_at, updated_at) VALUES (?, ?, ?)",
            (session_id, now, now),
        )
    return session_id


def get_session(session_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None


def get_all_sessions() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


# ── Datasets ──────────────────────────────────────────────────────────────────

def register_dataset(
    session_id: str,
    dataset_id: str,
    file_path: str,
    original_filename: str,
    row_count: int,
    column_count: int,
) -> None:
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO datasets
               (id, session_id, original_filename, file_path, row_count, column_count, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (dataset_id, session_id, original_filename, file_path, row_count, column_count, now),
        )


def get_dataset(dataset_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        return dict(row) if row else None


def get_session_datasets(session_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM datasets WHERE session_id = ?", (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]


# ── Analyses ──────────────────────────────────────────────────────────────────

def create_analysis(
    session_id: str,
    dataset_id: Optional[str],
    analysis_type: str,
    summary: str = "",
) -> str:
    analysis_id = f"anal_{uuid.uuid4().hex[:8]}"
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO analyses (id, session_id, dataset_id, type, summary, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (analysis_id, session_id, dataset_id, analysis_type, summary, now),
        )
    return analysis_id


def get_analysis(analysis_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        return dict(row) if row else None


def get_session_analyses(session_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM analyses WHERE session_id = ? ORDER BY created_at", (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]


# ── Artifacts ─────────────────────────────────────────────────────────────────

def register_artifact(
    session_id: str,
    artifact_type: str,
    filename: str,
    file_path: str,
    dataset_id: Optional[str] = None,
    analysis_id: Optional[str] = None,
    description: str = "",
    artifact_id: Optional[str] = None,
) -> str:
    artifact_id = artifact_id or f"art_{uuid.uuid4().hex[:8]}"
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO artifacts
               (id, session_id, dataset_id, analysis_id, artifact_type, filename, description, file_path, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (artifact_id, session_id, dataset_id, analysis_id, artifact_type,
             filename, description, file_path, now),
        )
    return artifact_id


def get_artifact(artifact_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,)).fetchone()
        return dict(row) if row else None


def get_session_artifacts(session_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM artifacts WHERE session_id = ? ORDER BY created_at", (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]


# ── Conversation history ───────────────────────────────────────────────────────

def save_conversation_turn(session_id: str, message: dict) -> None:
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO conversation_turns (session_id, message_json, created_at) VALUES (?, ?, ?)",
            (session_id, json.dumps(message), now),
        )


def get_conversation_history(session_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT message_json FROM conversation_turns WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [json.loads(r[0]) for r in rows]


# ── Execution traces ───────────────────────────────────────────────────────────

def save_trace_step(
    *,
    run_id: str,
    session_id: str,
    step_index: int,
    user_message: Optional[str],
    llm_decision: str,
    tool_name: Optional[str],
    tool_args: Optional[str],
    tool_result_status: Optional[str],
    tool_result_summary: Optional[str],
    execution_time_ms: float,
    artifact_ids: str,
    final_response: Optional[str],
) -> None:
    step_id = f"step_{uuid.uuid4().hex[:10]}"
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO traces
               (id, session_id, run_id, step_index, user_message, llm_decision,
                tool_name, tool_args, tool_result_status, tool_result_summary,
                execution_time_ms, artifact_ids, final_response, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                step_id, session_id, run_id, step_index, user_message, llm_decision,
                tool_name, tool_args, tool_result_status, tool_result_summary,
                execution_time_ms, artifact_ids, final_response, now,
            ),
        )


def get_session_traces(session_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM traces WHERE session_id = ? ORDER BY created_at, run_id, step_index",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]
