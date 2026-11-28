"""Offline end-to-end unit tests for run-pipeline.py."""

import csv
import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from storage import get_forecast, list_forecasts

_PIPELINE_PATH = Path(__file__).resolve().parent.parent / "run-pipeline.py"


def _load_pipeline_module() -> ModuleType:
    """Import run-pipeline.py by file path since its filename contains a hyphen."""
    spec = importlib.util.spec_from_file_location("run_pipeline", _PIPELINE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_gold_df() -> pd.DataFrame:
    """Create deterministic tz-naive Gold Futures test prices."""
    dates = pd.date_range("2026-11-20", periods=6, freq="B")
    prices = [3600.0, 3610.0, 3620.0, 3630.0, 3640.0, 3650.0]
    return pd.DataFrame({"Date": dates, "Gold_price": prices})


def test_pipeline_migrates_backfills_upserts_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full V2 run migrates V1 CSV, backfills prior row, upserts today, and handles re-runs."""
    pipeline = _load_pipeline_module()
    db_file = tmp_path / "gold.db"
    csv_file = tmp_path / "forecast_log.csv"

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
        writer.writerow(["2026-11-26", "3640.0", "0.10", "12", "3645.0", "5.0"])

    monkeypatch.setattr(pipeline, "fetch_gold_prices", _sample_gold_df)
    monkeypatch.setattr(
        pipeline,
        "fetch_news_sentiment",
        lambda: pd.DataFrame(
            [
                {"Date": "2026-11-27", "Headline": "Gold rallies", "Sentiment": 0.4},
                {"Date": "2026-11-27", "Headline": "Strong demand", "Sentiment": 0.2},
            ]
        ),
    )

    first = pipeline.main(db_path=db_file, csv_path=csv_file)
    assert first["run_date"] == "2026-11-27"
    assert first["migrated"] == 1
    assert first["backfilled"] == 1
    assert first["article_count"] == 2
    assert first["sentiment_score"] == pytest.approx(0.3)

    prior = get_forecast("2026-11-26", db_file)
    assert prior is not None
    assert prior["actual_next_price"] == pytest.approx(3650.0)
    assert prior["abs_error"] == pytest.approx(5.0)

    # Same-day re-run must not duplicate DB rows or CSV rows
    second = pipeline.main(db_path=db_file, csv_path=csv_file)
    assert second["migrated"] == 0
    assert second["backfilled"] == 0
    assert len(list_forecasts(db_file)) == 2

    df_csv = pd.read_csv(csv_file)
    assert list(df_csv["Date"]) == ["2026-11-26", "2026-11-27"]


def test_pipeline_empty_news_forces_zero_sentiment_and_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty RSS feed results in article_count == 0, sentiment_score == 0.0, and HOLD."""
    pipeline = _load_pipeline_module()
    db_file = tmp_path / "empty_news.db"
    csv_file = tmp_path / "empty_news.csv"

    monkeypatch.setattr(pipeline, "fetch_gold_prices", _sample_gold_df)
    monkeypatch.setattr(pipeline, "fetch_news_sentiment", lambda: pd.DataFrame())

    result = pipeline.main(db_path=db_file, csv_path=csv_file)
    assert result["article_count"] == 0
    assert result["sentiment_score"] == 0.0
    assert result["signal"] == "HOLD"
