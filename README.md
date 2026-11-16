# Automated Gold Price & News Sentiment Forecasting Pipeline

An end-to-end, automated data science pipeline that ingests live commodity market data and financial news headlines daily to forecast next-day Gold Futures (`GC=F`) prices.

## Business Problem
Commodity buyers, jewelers, and institutional desks face daily price volatility when timing gold procurement. Relying solely on historical price charts ignores sudden macroeconomic shifts driven by breaking news, while manual tracking does not scale. This pipeline automates daily market ingestion, NLP sentiment scoring, and price forecasting into a self-updating historical log.

## System Architecture & Tech Stack
* **Data Ingestion:** `yfinance` (5-year historical & daily Gold Futures `GC=F` closing prices) + `feedparser` (live Google News RSS feed).
* **Natural Language Processing:** `vaderSentiment` to score financial news headlines (`compound` polarity).
* **Data Engineering:** `pandas` for timezone normalization (`tz_localize(None)`), date alignment, missing-value imputation, and daily incremental logging (`forecast_log.csv`).
* **Machine Learning:** `scikit-learn` (`LinearRegression`) trained on de-biased features (`Gold_price` and `Avg_Sentiment`).
* **Automation / CI/CD:** **GitHub Actions** (`daily_pipeline.yml`) scheduled via cron (`22:00 UTC` daily) to execute the pipeline and commit updated forecasts automatically.

## Key Engineering & Modeling Decisions
1. **Model Selection (Linear Regression vs. Random Forest):** Tree-based models like Random Forest suffer from an extrapolation ceiling on upward-trending asset prices, resulting in severe under-prediction (`~$440` MAE) compared to Linear Regression (`~$57` MAE). Linear Regression was selected to preserve continuous trend extrapolation.
2. **Eliminating Survivorship Bias:** Google News RSS feeds only return recent headlines, causing historical rows (older than ~30 days) to default to `Article_Count = 0`. Including raw `Article_Count` introduced artificial negative bias because older dates had lower gold prices and zero recorded articles. Dropping `Article_Count` from the feature matrix removed this leakage.
3. **Idempotent Daily Logging:** `run-pipeline.py` deduplicates by date before appending (`existing_log[existing_log['Date'] != today_date_str]`), ensuring manual re-runs update cleanly without corrupting the historical database.

## Repository Structure
* `run-pipeline.py` — Standalone production script that fetches data, trains the model, and updates `forecast_log.csv`.
* `forecast_log.csv` — Self-updating daily log of Gold closing prices, news sentiment, article volume, and next-day predictions.
* `Gold_News_Pipeline.ipynb` — Exploratory data analysis, model benchmarking, and residual diagnostics.
* `.github/workflows/daily_pipeline.yml` — GitHub Actions CI/CD workflow for automated daily execution.

## Roadmap
* **V1 (Completed):** Automated daily ingestion, VADER sentiment scoring, Linear Regression forecasting, and GitHub Actions CI/CD logging.
* **V2 (Planned):** Automated procurement decision rules (`BUY` / `HOLD` / `NEUTRAL`) and SQLite database storage (`gold_database.db`).
* **V3 (Planned):** Interactive public dashboard deployed via Streamlit Community Cloud.
* 
