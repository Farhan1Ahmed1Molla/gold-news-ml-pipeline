# SPEC_V2: SQLite Persistence & Long-Term Gold Accumulation Signals

## 1. Objective
Upgrade `run-pipeline.py` to persist daily forecasts in SQLite (`gold_database.db`), backfill next-trading-day actual prices and errors, emit long-term accumulation signals (`ACCUMULATE`, `HOLD`, `PAUSE`), and continue updating `forecast_log.csv` for backwards compatibility.

## 2. Module Boundaries & Interfaces (Deep Modules)
- `storage.py`:
  - `init_db(db_path: str = "gold_database.db") -> None`
  - `migrate_csv_if_needed(csv_path: str = "forecast_log.csv", db_path: str = "gold_database.db") -> int`
  - `upsert_forecast(record: dict, db_path: str = "gold_database.db") -> None`
  - `backfill_actuals(trading_date: str, actual_price: float, db_path: str = "gold_database.db") -> int`
- `signals.py`:
  - Named constants: `PRICE_THRESHOLD_PCT = 0.5`, `SENTIMENT_THRESHOLD = 0.15`
  - `generate_signal(predicted_change_pct: float, sentiment_score: float, article_count: int) -> str`

## 3. Database Schema (`forecasts` table)
- `run_date TEXT PRIMARY KEY` (ISO `YYYY-MM-DD`)
- `current_price REAL NOT NULL`
- `predicted_price REAL NOT NULL`
- `predicted_change_pct REAL NOT NULL`
- `sentiment_score REAL NOT NULL`
- `article_count INTEGER NOT NULL`
- `signal TEXT NOT NULL` (`ACCUMULATE`, `HOLD`, or `PAUSE`)
- `actual_next_price REAL` (nullable until next trading day)
- `abs_error REAL` (nullable until next trading day)

## 4. Long-Term Signal Rules (`signals.py`)
- **`ACCUMULATE`**: `article_count > 0` AND `predicted_change_pct >= 0.5` AND `sentiment_score >= 0.15`
- **`PAUSE`**: `article_count > 0` AND `predicted_change_pct <= -0.5` AND `sentiment_score <= -0.15`
- **`HOLD`**: All other cases, including when `article_count == 0`

## 5. Edge Cases & Security Hygiene
1. **Idempotent same-day re-runs:** Upsert on `run_date` so re-running updates the row without duplicates.
2. **Zero articles (`article_count == 0`):** Set `sentiment_score = 0.0` and return `HOLD`.
3. **Weekends & market holidays:** Backfill `actual_next_price` on the most recent prior row where `actual_next_price IS NULL` and `run_date < trading_date`.
4. **SQL & data safety:** Use parameterized `?` SQL queries everywhere and validate numeric inputs (`float`/`int`, non-NaN) before DB writes.

## 6. Out of Scope (Deferred)
- Modifying `Gold_News_Pipeline.ipynb` or `.github/workflows`
- Retraining or changing the ML model
- Building a web dashboard UI

## 7. Verification Criteria
1. `uv run pytest` passes unit tests for `ACCUMULATE`/`HOLD`/`PAUSE` rules, `article_count == 0`, same-day upserts, and weekend gap backfills.
2. Existing `forecast_log.csv` rows migrate cleanly into `gold_database.db` without duplication.
3. Running `uv run python run-pipeline.py` updates both `gold_database.db` and `forecast_log.csv`.
