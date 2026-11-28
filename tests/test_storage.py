"""Automated unit tests for storage.py (SQLite persistence, backfill, CSV migration)."""

import csv
import math
import sqlite3
from pathlib import Path

import pytest
from storage import (
    backfill_actuals,
    get_forecast,
    init_db,
    list_forecasts,
    migrate_csv_if_needed,
    upsert_forecast,
)


def _sample_record(**overrides: object) -> dict[str, object]:
    """Build a valid forecast record dict with optional field overrides."""
    base: dict[str, object] = {
        "run_date": "2026-11-20",
        "current_price": 3500.0,
        "predicted_price": 3535.0,
        "predicted_change_pct": 1.0,
        "sentiment_score": 0.25,
        "article_count": 12,
        "signal": "ACCUMULATE",
    }
    base.update(overrides)
    return base


# ==============================================================================
# 1. SCHEMA INITIALIZATION & SQLITE CHECK CONSTRAINTS
# ==============================================================================
def test_init_db_creates_spec_schema_and_accepts_str_path(tmp_path: Path) -> None:
    """init_db accepts str paths and creates all SPEC_V2 columns."""
    db_file = str(tmp_path / "nested" / "test.db")
    init_db(db_file)

    with sqlite3.connect(db_file) as conn:
        cols = {
            row[1] for row in conn.execute("PRAGMA table_info(forecasts);").fetchall()
        }

    expected_cols = {
        "run_date",
        "current_price",
        "predicted_price",
        "predicted_change_pct",
        "sentiment_score",
        "article_count",
        "signal",
        "actual_next_price",
        "abs_error",
    }
    assert expected_cols.issubset(cols)


def test_sqlite_check_constraints_reject_invalid_direct_inserts(tmp_path: Path) -> None:
    """Table CHECK constraints guard signal, article_count, and positive prices."""
    db_file = tmp_path / "checks.db"
    init_db(db_file)

    with sqlite3.connect(db_file) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO forecasts (
                    run_date, current_price, predicted_price, predicted_change_pct,
                    sentiment_score, article_count, signal
                ) VALUES ('2026-11-20', 3500.0, 3520.0, 0.57, 0.2, 5, 'INVALID');
                """
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO forecasts (
                    run_date, current_price, predicted_price, predicted_change_pct,
                    sentiment_score, article_count, signal
                ) VALUES ('2026-11-20', 3500.0, 3520.0, 0.57, 0.2, -1, 'HOLD');
                """
            )


# ==============================================================================
# 2. IDEMPOTENT SAME-DAY UPSERT & INPUT VALIDATION
# ==============================================================================
def test_upsert_forecast_is_idempotent_and_preserves_actuals(tmp_path: Path) -> None:
    """Re-running upsert for the same run_date updates forecast fields without duplicates."""
    db_file = tmp_path / "upsert.db"
    upsert_forecast(_sample_record(), db_file)
    assert backfill_actuals("2026-11-23", 3550.0, db_file) == 1

    # Re-run same day with updated forecast values and omitted actual_next_price
    upsert_forecast(
        _sample_record(
            predicted_price=3540.0,
            predicted_change_pct=1.14,
            sentiment_score=0.30,
            article_count=20,
        ),
        db_file,
    )

    rows = list_forecasts(db_file)
    assert len(rows) == 1
    row = rows[0]
    assert row["predicted_price"] == pytest.approx(3540.0)
    assert row["article_count"] == 20
    # Existing actual_next_price is preserved and abs_error recalculates against new predicted_price
    assert row["actual_next_price"] == pytest.approx(3550.0)
    assert row["abs_error"] == pytest.approx(10.0)


def test_upsert_zero_articles_normalizes_sentiment_to_zero(tmp_path: Path) -> None:
    """Per SPEC_V2, article_count == 0 forces stored sentiment_score to 0.0."""
    db_file = tmp_path / "zero_articles.db"
    upsert_forecast(
        _sample_record(article_count=0, sentiment_score=0.42, signal="HOLD"),
        db_file,
    )
    row = get_forecast("2026-11-20", db_file)
    assert row is not None
    assert row["article_count"] == 0
    assert row["sentiment_score"] == 0.0
    assert row["signal"] == "HOLD"


@pytest.mark.parametrize(
    "overrides, exc_type",
    [
        ({"run_date": ""}, ValueError),
        ({"run_date": "20261120"}, ValueError),
        ({"run_date": "2026-02-30"}, ValueError),
        ({"current_price": 0.0}, ValueError),
        ({"current_price": -100.0}, ValueError),
        ({"predicted_price": math.nan}, ValueError),
        ({"predicted_change_pct": math.inf}, ValueError),
        ({"sentiment_score": -math.inf}, ValueError),
        ({"current_price": True}, TypeError),
        ({"article_count": True}, TypeError),
        ({"article_count": -1}, ValueError),
        ({"article_count": 2.5}, TypeError),
        ({"signal": "BUY"}, ValueError),
        ({"article_count": 0, "signal": "ACCUMULATE"}, ValueError),
        ({"actual_next_price": None, "abs_error": 5.0}, ValueError),
    ],
)
def test_upsert_rejects_invalid_inputs(
    tmp_path: Path, overrides: dict[str, object], exc_type: type[Exception]
) -> None:
    """Invalid types, non-finite numbers, booleans, and bad dates raise before SQLite write."""
    db_file = tmp_path / "invalid.db"
    with pytest.raises(exc_type):
        upsert_forecast(_sample_record(**overrides), db_file)


# ==============================================================================
# 3. WEEKEND & MARKET HOLIDAY BACKFILLS
# ==============================================================================
def test_backfill_actuals_bridges_weekend_and_holiday_gaps(tmp_path: Path) -> None:
    """Friday forecast is backfilled on Monday (or post-holiday) using latest prior open row."""
    db_file = tmp_path / "backfill.db"

    # Thursday already backfilled; Friday open across weekend + Monday holiday
    upsert_forecast(
        _sample_record(
            run_date="2026-11-19",
            current_price=3600.0,
            predicted_price=3610.0,
            actual_next_price=3580.0,
            abs_error=30.0,
            signal="HOLD",
        ),
        db_file,
    )
    upsert_forecast(
        _sample_record(
            run_date="2026-11-20",
            current_price=3580.0,
            predicted_price=3595.0,
            signal="HOLD",
        ),
        db_file,
    )

    # Tuesday trading session backfills Friday (2026-11-20)
    updated = backfill_actuals("2026-11-24", 3605.0, db_file)
    assert updated == 1

    friday = get_forecast("2026-11-20", db_file)
    assert friday is not None
    assert friday["actual_next_price"] == pytest.approx(3605.0)
    assert friday["abs_error"] == pytest.approx(10.0)

    # Thursday must remain untouched
    thursday = get_forecast("2026-11-19", db_file)
    assert thursday is not None
    assert thursday["actual_next_price"] == pytest.approx(3580.0)

    # Re-running backfill on the same trading date does nothing
    assert backfill_actuals("2026-11-24", 3605.0, db_file) == 0


def test_backfill_actuals_returns_zero_when_no_prior_row(tmp_path: Path) -> None:
    """Backfill returns 0 when DB is empty or only same-day/future rows exist."""
    db_file = tmp_path / "empty_backfill.db"
    assert backfill_actuals("2026-11-20", 3500.0, db_file) == 0

    upsert_forecast(_sample_record(run_date="2026-11-20"), db_file)
    assert backfill_actuals("2026-11-20", 3510.0, db_file) == 0


# ==============================================================================
# 4. CSV MIGRATION (V1 forecast_log.csv FORMAT & ATOMICITY)
# ==============================================================================
def test_migrate_v1_forecast_log_csv_and_idempotency(tmp_path: Path) -> None:
    """V1 forecast_log.csv rows migrate cleanly, derive signals, and skip duplicates."""
    csv_file = tmp_path / "forecast_log.csv"
    db_file = tmp_path / "migrated.db"

    # Write with UTF-8 BOM to verify utf-8-sig handling
    with csv_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Date",
                "Today_Gold_Price",
                "Avg_Sentiment",
                "Article_Count",
                "Predicted_Tomorrow_Price",
                "Expected_Move_USD",
            ]
        )
        # Row 1: +1.0% move, +0.30 sentiment, 15 articles -> ACCUMULATE
        writer.writerow(["2026-11-20", "3500.0", "0.30", "15", "3535.0", "35.0"])
        # Row 2: -1.0% move, -0.30 sentiment, 0 articles -> HOLD (0-article guardrail)
        writer.writerow(["2026-11-22", "3550.0", "-0.30", "0", "3514.5", "-35.5"])
        # Row 3: -1.0% move, -0.25 sentiment, 10 articles -> PAUSE
        writer.writerow(["2026-11-24", "3520.0", "-0.25", "10", "3484.8", "-35.2"])

    migrated = migrate_csv_if_needed(csv_file, db_file)
    assert migrated == 3

    # Re-running migration must be idempotent and insert 0 duplicates
    assert migrate_csv_if_needed(csv_file, db_file) == 0

    rows = list_forecasts(db_file)
    assert [r["run_date"] for r in rows] == ["2026-11-20", "2026-11-22", "2026-11-24"]

    # Historical sequential backfill populated for prior rows; latest row stays NULL
    assert rows[0]["signal"] == "ACCUMULATE"
    assert rows[0]["actual_next_price"] == pytest.approx(3550.0)
    assert rows[0]["abs_error"] == pytest.approx(15.0)

    assert rows[1]["signal"] == "HOLD"
    assert rows[1]["sentiment_score"] == 0.0
    assert rows[1]["actual_next_price"] == pytest.approx(3520.0)
    assert rows[1]["abs_error"] == pytest.approx(5.5)

    assert rows[2]["signal"] == "PAUSE"
    assert rows[2]["actual_next_price"] is None
    assert rows[2]["abs_error"] is None

    # Next trading day backfills the latest migrated row (2026-11-24)
    assert backfill_actuals("2026-11-25", 3500.0, db_file) == 1
    latest = get_forecast("2026-11-24", db_file)
    assert latest is not None
    assert latest["actual_next_price"] == pytest.approx(3500.0)
    assert latest["abs_error"] == pytest.approx(15.2)


def test_migrate_csv_missing_file_returns_zero(tmp_path: Path) -> None:
    """Missing CSV initializes DB cleanly and returns 0."""
    db_file = tmp_path / "no_csv.db"
    assert migrate_csv_if_needed(tmp_path / "missing.csv", db_file) == 0
    assert list_forecasts(db_file) == []


def test_migrate_csv_is_atomic_on_malformed_row(tmp_path: Path) -> None:
    """If any CSV row is invalid, migration fails without committing partial rows."""
    csv_file = tmp_path / "bad.csv"
    db_file = tmp_path / "atomic.db"

    with csv_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Date",
                "Today_Gold_Price",
                "Avg_Sentiment",
                "Article_Count",
                "Predicted_Tomorrow_Price",
                "Expected_Move_USD",
            ]
        )
        writer.writerow(["2026-11-20", "3500.0", "0.30", "15", "3535.0", "35.0"])
        writer.writerow(["2026-11-21", "NOT_A_NUMBER", "0.10", "5", "3510.0", "10.0"])

    with pytest.raises(ValueError):
        migrate_csv_if_needed(csv_file, db_file)

    assert list_forecasts(db_file) == []
