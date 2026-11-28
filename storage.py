"""SQLite persistence layer for V2 gold forecasts (keeps forecast_log.csv compatible)."""

import csv
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

from signals import (
    VALID_SIGNALS,
    generate_signal,
    validate_finite_float,
    validate_non_negative_int,
)

logger = logging.getLogger(__name__)

# Default database file in project root (tracked in Git per project setup).
DEFAULT_DB_PATH: str = "gold_database.db"
DEFAULT_CSV_PATH: str = "forecast_log.csv"

_NULL_TOKENS: frozenset[str] = frozenset({"", "none", "nan", "null", "na", "n/a"})

# SQL schema for daily gold forecast records aligned with SPEC_V2.md.
FORECASTS_TABLE_SQL: str = """
CREATE TABLE IF NOT EXISTS forecasts (
    run_date TEXT PRIMARY KEY,
    current_price REAL NOT NULL CHECK (current_price > 0),
    predicted_price REAL NOT NULL CHECK (predicted_price > 0),
    predicted_change_pct REAL NOT NULL,
    sentiment_score REAL NOT NULL,
    article_count INTEGER NOT NULL CHECK (article_count >= 0),
    signal TEXT NOT NULL CHECK (
        signal IN ('ACCUMULATE', 'HOLD', 'PAUSE')
        AND (article_count > 0 OR signal = 'HOLD')
    ),
    actual_next_price REAL CHECK (actual_next_price IS NULL OR actual_next_price > 0),
    abs_error REAL CHECK (
        abs_error IS NULL
        OR (abs_error >= 0 AND actual_next_price IS NOT NULL)
    ),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_UPSERT_FORECAST_SQL: str = """
INSERT INTO forecasts (
    run_date,
    current_price,
    predicted_price,
    predicted_change_pct,
    sentiment_score,
    article_count,
    signal,
    actual_next_price,
    abs_error,
    updated_at
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
ON CONFLICT(run_date) DO UPDATE SET
    current_price = excluded.current_price,
    predicted_price = excluded.predicted_price,
    predicted_change_pct = excluded.predicted_change_pct,
    sentiment_score = excluded.sentiment_score,
    article_count = excluded.article_count,
    signal = excluded.signal,
    actual_next_price = COALESCE(excluded.actual_next_price, forecasts.actual_next_price),
    abs_error = COALESCE(
        excluded.abs_error,
        CASE
            WHEN COALESCE(excluded.actual_next_price, forecasts.actual_next_price) IS NOT NULL
            THEN ABS(
                COALESCE(excluded.actual_next_price, forecasts.actual_next_price)
                - excluded.predicted_price
            )
            ELSE NULL
        END
    ),
    updated_at = datetime('now');
"""


def _validate_iso_date(value: object, name: str = "run_date") -> str:
    """Validate an ISO YYYY-MM-DD date string."""
    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be an ISO date string, got {type(value).__name__}"
        )
    clean = value.strip()
    if len(clean) != 10 or clean[4] != "-" or clean[7] != "-":
        raise ValueError(f"{name} must be in YYYY-MM-DD format, got {value!r}")
    date.fromisoformat(clean)
    return clean


def _validate_positive_float(value: object, name: str) -> float:
    """Validate that value is a finite number strictly greater than zero."""
    numeric = validate_finite_float(value, name)
    if numeric <= 0:
        raise ValueError(f"{name} must be > 0, got {numeric}")
    return numeric


def _validate_signal(value: object) -> str:
    """Validate that signal is one of ACCUMULATE, HOLD, or PAUSE."""
    if not isinstance(value, str):
        raise TypeError(f"signal must be a string, got {type(value).__name__}")
    clean = value.strip()
    if clean not in VALID_SIGNALS:
        raise ValueError(
            f"signal must be one of {sorted(VALID_SIGNALS)}, got {value!r}"
        )
    return clean


def _validate_optional_positive_float(value: object, name: str) -> float | None:
    """Validate an optional positive finite float."""
    if value is None:
        return None
    return _validate_positive_float(value, name)


def _validate_optional_non_negative_float(value: object, name: str) -> float | None:
    """Validate an optional non-negative finite float."""
    if value is None:
        return None
    numeric = validate_finite_float(value, name)
    if numeric < 0:
        raise ValueError(f"{name} must be >= 0, got {numeric}")
    return numeric


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a forecast record dict before database persistence."""
    if not isinstance(record, dict):
        raise TypeError(f"record must be a dict, got {type(record).__name__}")

    required_keys = (
        "run_date",
        "current_price",
        "predicted_price",
        "predicted_change_pct",
        "sentiment_score",
        "article_count",
        "signal",
    )
    missing = [key for key in required_keys if key not in record]
    if missing:
        raise KeyError(f"Missing required forecast fields: {', '.join(missing)}")

    run_date = _validate_iso_date(record["run_date"], "run_date")
    current_price = _validate_positive_float(record["current_price"], "current_price")
    predicted_price = _validate_positive_float(
        record["predicted_price"], "predicted_price"
    )
    predicted_change_pct = validate_finite_float(
        record["predicted_change_pct"], "predicted_change_pct"
    )
    article_count = validate_non_negative_int(record["article_count"], "article_count")
    sentiment_score = validate_finite_float(
        record["sentiment_score"], "sentiment_score"
    )
    signal = _validate_signal(record["signal"])

    if article_count == 0:
        sentiment_score = 0.0
        if signal != "HOLD":
            raise ValueError(
                f"signal must be 'HOLD' when article_count == 0, got {signal!r}"
            )

    actual_next_price = _validate_optional_positive_float(
        record.get("actual_next_price"), "actual_next_price"
    )
    abs_error = _validate_optional_non_negative_float(
        record.get("abs_error"), "abs_error"
    )
    if actual_next_price is None and abs_error is not None:
        raise ValueError("abs_error cannot be set when actual_next_price is None")
    if actual_next_price is not None and abs_error is None:
        abs_error = abs(actual_next_price - predicted_price)

    return {
        "run_date": run_date,
        "current_price": current_price,
        "predicted_price": predicted_price,
        "predicted_change_pct": predicted_change_pct,
        "sentiment_score": sentiment_score,
        "article_count": article_count,
        "signal": signal,
        "actual_next_price": actual_next_price,
        "abs_error": abs_error,
    }


def _record_params(normalized: dict[str, Any]) -> tuple[Any, ...]:
    """Convert a normalized forecast record into positional SQL parameters."""
    return (
        normalized["run_date"],
        normalized["current_price"],
        normalized["predicted_price"],
        normalized["predicted_change_pct"],
        normalized["sentiment_score"],
        normalized["article_count"],
        normalized["signal"],
        normalized["actual_next_price"],
        normalized["abs_error"],
    )


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with row-by-name access enabled."""
    resolved = Path(db_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(resolved, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _db_session(db_path: str | Path = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    """Ensure the forecasts table exists, run within a transaction, and close safely."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(FORECASTS_TABLE_SQL)
            yield conn
    finally:
        conn.close()


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create database tables if they do not already exist."""
    with _db_session(db_path):
        pass


def upsert_forecast(
    record: dict[str, Any], db_path: str | Path = DEFAULT_DB_PATH
) -> None:
    """Insert a daily forecast record or update it cleanly on run_date conflict."""
    normalized = _normalize_record(record)
    with _db_session(db_path) as conn:
        conn.execute(_UPSERT_FORECAST_SQL, _record_params(normalized))


def backfill_actuals(
    trading_date: str,
    actual_price: float,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Backfill actual_next_price and abs_error on the latest prior open forecast.

    Finds the most recent row with run_date < trading_date and actual_next_price IS NULL,
    handling weekends and market holidays cleanly. Returns 1 if updated, 0 otherwise.
    """
    clean_date = _validate_iso_date(trading_date, "trading_date")
    clean_price = _validate_positive_float(actual_price, "actual_price")

    select_sql = """
    SELECT run_date, predicted_price
    FROM forecasts
    WHERE run_date < ? AND actual_next_price IS NULL
    ORDER BY run_date DESC
    LIMIT 1;
    """
    update_sql = """
    UPDATE forecasts
    SET actual_next_price = ?, abs_error = ?, updated_at = datetime('now')
    WHERE run_date = ?;
    """

    with _db_session(db_path) as conn:
        row = conn.execute(select_sql, (clean_date,)).fetchone()
        if row is None:
            return 0
        abs_error = abs(clean_price - float(row["predicted_price"]))
        cursor = conn.execute(update_sql, (clean_price, abs_error, row["run_date"]))
        return cursor.rowcount


def _first_present(row: dict[str, Any], *keys: str) -> str | None:
    """Return the first non-empty stripped string value among candidate CSV keys."""
    for key in keys:
        val = row.get(key)
        if val is not None:
            stripped = str(val).strip()
            if stripped != "":
                return stripped
    return None


def _parse_csv_float(raw: str, name: str) -> float:
    """Parse a finite float from a CSV field with contextual error messages."""
    try:
        parsed = float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    return validate_finite_float(parsed, name)


def _parse_optional_csv_float(raw: str | None, name: str) -> float | None:
    """Parse an optional finite numeric CSV field, treating null-like tokens as None."""
    if raw is None or raw.strip().lower() in _NULL_TOKENS:
        return None
    return _parse_csv_float(raw, name)


def _parse_csv_int(raw: str, name: str) -> int:
    """Parse an integer CSV field without silently truncating non-integer floats."""
    stripped = raw.strip()
    try:
        return int(stripped)
    except ValueError:
        as_float = _parse_csv_float(stripped, name)
        if not as_float.is_integer():
            raise ValueError(f"{name} must be a whole number, got {raw!r}")
        return int(as_float)


def _parse_csv_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a V1 or V2 forecast_log.csv row into a normalized V2 forecast dict."""
    if all(val is None or str(val).strip() == "" for val in row.values()):
        return None

    raw_date = _first_present(row, "run_date", "Date", "date", "forecast_date")
    if raw_date is None:
        raise ValueError(f"CSV row missing date column: {row!r}")

    raw_current = _first_present(row, "current_price", "Today_Gold_Price")
    raw_predicted = _first_present(row, "predicted_price", "Predicted_Tomorrow_Price")
    raw_sentiment = _first_present(row, "sentiment_score", "Avg_Sentiment")
    raw_count = _first_present(row, "article_count", "Article_Count")
    if None in (raw_current, raw_predicted, raw_sentiment, raw_count):
        raise ValueError(f"CSV row missing required numeric columns: {row!r}")

    current_price = _validate_positive_float(
        _parse_csv_float(raw_current, "current_price"), "current_price"
    )
    predicted_price = _validate_positive_float(
        _parse_csv_float(raw_predicted, "predicted_price"), "predicted_price"
    )
    article_count = validate_non_negative_int(
        _parse_csv_int(raw_count, "article_count"), "article_count"
    )
    parsed_sentiment = _parse_csv_float(raw_sentiment, "sentiment_score")
    sentiment_score = 0.0 if article_count == 0 else parsed_sentiment

    raw_pct = _first_present(row, "predicted_change_pct")
    predicted_change_pct = (
        _parse_csv_float(raw_pct, "predicted_change_pct")
        if raw_pct is not None
        else ((predicted_price - current_price) / current_price) * 100.0
    )

    raw_signal = _first_present(row, "signal")
    signal = (
        raw_signal
        if raw_signal is not None
        else generate_signal(predicted_change_pct, sentiment_score, article_count)
    )

    actual_next_price = _parse_optional_csv_float(
        _first_present(row, "actual_next_price"), "actual_next_price"
    )
    abs_error = _parse_optional_csv_float(_first_present(row, "abs_error"), "abs_error")

    return _normalize_record(
        {
            "run_date": raw_date,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "predicted_change_pct": predicted_change_pct,
            "sentiment_score": sentiment_score,
            "article_count": article_count,
            "signal": signal,
            "actual_next_price": actual_next_price,
            "abs_error": abs_error,
        }
    )


def migrate_csv_if_needed(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Import historical rows from forecast_log.csv into SQLite atomically.

    Skips dates already present in SQLite (idempotent), backfills sequential
    historical actual_next_price/abs_error across consecutive migrated rows when
    missing, and returns the number of newly inserted rows.
    """
    source = Path(csv_path)
    if not source.exists():
        return 0

    parsed_rows: list[dict[str, Any]] = []
    with source.open(mode="r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_num, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(
                    f"Malformed CSV row at line {line_num} with extra columns: {row!r}"
                )
            parsed = _parse_csv_row(row)
            if parsed is not None:
                parsed_rows.append(parsed)

    if not parsed_rows:
        return 0

    # Deduplicate same-date rows in CSV (keep last occurrence in file order), then sort.
    deduped_by_date: dict[str, dict[str, Any]] = {}
    for item in parsed_rows:
        deduped_by_date[item["run_date"]] = item
    ordered_rows = [deduped_by_date[k] for k in sorted(deduped_by_date)]

    for idx in range(len(ordered_rows) - 1):
        prev_row = ordered_rows[idx]
        next_row = ordered_rows[idx + 1]
        if prev_row["actual_next_price"] is None:
            actual = next_row["current_price"]
            prev_row["actual_next_price"] = actual
            prev_row["abs_error"] = abs(actual - prev_row["predicted_price"])

    migrated = 0
    with _db_session(db_path) as conn:
        existing_dates = {
            row["run_date"]
            for row in conn.execute("SELECT run_date FROM forecasts;").fetchall()
        }
        to_insert = [
            _record_params(item)
            for item in ordered_rows
            if item["run_date"] not in existing_dates
        ]
        if to_insert:
            conn.executemany(_UPSERT_FORECAST_SQL, to_insert)
            migrated = len(to_insert)

    if migrated:
        logger.info("Migrated %d historical rows from %s", migrated, source)
    return migrated


def get_forecast(
    run_date: str, db_path: str | Path = DEFAULT_DB_PATH
) -> dict[str, Any] | None:
    """Fetch a single forecast record by run_date, or None if not found."""
    clean_date = _validate_iso_date(run_date, "run_date")
    with _db_session(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM forecasts WHERE run_date = ?;", (clean_date,)
        ).fetchone()
        return dict(row) if row is not None else None


def list_forecasts(db_path: str | Path = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Return all forecast records ordered chronologically by run_date."""
    with _db_session(db_path) as conn:
        rows = conn.execute("SELECT * FROM forecasts ORDER BY run_date ASC;").fetchall()
        return [dict(row) for row in rows]
