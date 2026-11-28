import math
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

import feedparser
import pandas as pd
import yfinance as yf
from signals import generate_signal
from sklearn.linear_model import LinearRegression
from storage import (
    DEFAULT_CSV_PATH,
    DEFAULT_DB_PATH,
    backfill_actuals,
    migrate_csv_if_needed,
    upsert_forecast,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

RSS_URL = (
    "https://news.google.com/rss/search?q=gold+price+market&hl=en-US&gl=US&ceid=US:en"
)
HTTP_TIMEOUT_SECONDS = 20


def fetch_gold_prices(period: str = "5y") -> pd.DataFrame:
    """Fetch Gold Futures (GC=F) closing prices with tz-naive normalized dates."""
    gold_ticker = yf.Ticker("GC=F")
    df_gold = gold_ticker.history(period=period)[["Close"]].dropna().reset_index()
    if len(df_gold) < 2:
        raise RuntimeError(
            "Insufficient Gold Futures data returned by yfinance (need >= 2 rows)."
        )
    df_gold.columns = ["Date", "Gold_price"]
    dates = pd.to_datetime(df_gold["Date"])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    df_gold["Date"] = dates.dt.normalize()
    return df_gold


def fetch_news_sentiment(rss_url: str = RSS_URL) -> pd.DataFrame:
    """Fetch live Google News RSS headlines and score them with VADER."""
    req = Request(rss_url, headers={"User-Agent": "gold-news-ml-pipeline/2.0"})
    with urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as response:
        feed = feedparser.parse(response.read())

    analyzer = SentimentIntensityAnalyzer()
    news_list: list[dict[str, object]] = []
    for entry in feed.entries:
        pub_parsed = getattr(entry, "published_parsed", None)
        title = getattr(entry, "title", "")
        if pub_parsed is None or not isinstance(title, str) or not title.strip():
            continue
        pub_date = datetime(*pub_parsed[:6]).strftime("%Y-%m-%d")
        score = analyzer.polarity_scores(title)["compound"]
        news_list.append({"Date": pub_date, "Headline": title, "Sentiment": score})

    return pd.DataFrame(news_list)


def update_csv_log(
    new_log_row: pd.DataFrame, csv_path: str | Path = DEFAULT_CSV_PATH
) -> None:
    """Update forecast_log.csv idempotently without duplicating the same Date."""
    target = Path(csv_path)
    today_date_str = str(new_log_row["Date"].iloc[0])

    if target.exists() and target.stat().st_size > 0:
        existing_log = pd.read_csv(target)
        existing_log = existing_log[existing_log["Date"].astype(str) != today_date_str]
        updated_log = pd.concat([existing_log, new_log_row], ignore_index=True)
    else:
        updated_log = new_log_row

    updated_log["Date"] = updated_log["Date"].astype(str)
    updated_log = updated_log.sort_values("Date").reset_index(drop=True)
    updated_log.to_csv(target, index=False)


def main(
    db_path: str | Path = DEFAULT_DB_PATH,
    csv_path: str | Path = DEFAULT_CSV_PATH,
) -> dict[str, object]:
    """Run daily gold price + news sentiment pipeline and persist V1 CSV + V2 SQLite."""
    # 1. Fetch 5 years of historical Gold Futures data
    df_gold = fetch_gold_prices()

    # 2. Fetch live Gold news from Google News RSS
    df_news = fetch_news_sentiment()

    # 3. Aggregate news by date and merge with Gold prices
    if not df_news.empty:
        df_news["Date"] = pd.to_datetime(df_news["Date"]).dt.normalize()
        daily_sentiment = (
            df_news.groupby("Date")
            .agg(
                Avg_Sentiment=("Sentiment", "mean"),
                Article_Count=("Headline", "count"),
            )
            .reset_index()
        )
        df_merged = pd.merge(df_gold, daily_sentiment, on="Date", how="left")
    else:
        df_merged = df_gold.copy()
        df_merged["Avg_Sentiment"] = 0.0
        df_merged["Article_Count"] = 0

    df_merged["Avg_Sentiment"] = df_merged["Avg_Sentiment"].fillna(0.0)
    df_merged["Article_Count"] = df_merged["Article_Count"].fillna(0)

    # 4. Create target column (Tomorrow's Price) and split historical vs. live row
    df_merged["Tommorow_Price"] = df_merged["Gold_price"].shift(-1)

    historical_data = df_merged.dropna(subset=["Tommorow_Price"])
    today_row = df_merged.tail(1)

    clean_features = ["Gold_price", "Avg_Sentiment"]
    X_clean = historical_data[clean_features]
    y = historical_data["Tommorow_Price"]
    test_X_clean = today_row[clean_features]

    # 5. Train de-biased Linear Regression model and predict tomorrow's price
    clean_model = LinearRegression()
    clean_model.fit(X_clean, y)

    today_price = float(today_row["Gold_price"].values[0])
    if not math.isfinite(today_price) or today_price <= 0:
        raise ValueError(f"Invalid today gold price: {today_price}")

    clean_pred = float(clean_model.predict(test_X_clean)[0])
    if not math.isfinite(clean_pred) or clean_pred <= 0:
        raise ValueError(
            f"Model predicted non-positive or non-finite price ({clean_pred}); cannot proceed."
        )
    clean_move = clean_pred - today_price
    predicted_change_pct = (clean_move / today_price) * 100.0
    today_date_str = pd.to_datetime(today_row["Date"].values[0]).strftime("%Y-%m-%d")

    article_count = int(today_row["Article_Count"].values[0])
    sentiment_score = (
        0.0 if article_count == 0 else float(today_row["Avg_Sentiment"].values[0])
    )
    signal = generate_signal(predicted_change_pct, sentiment_score, article_count)

    # 6. Persist V2 record to SQLite (migrate CSV -> backfill prior row -> upsert today)
    migrated_count = migrate_csv_if_needed(csv_path, db_path)
    backfilled_count = backfill_actuals(today_date_str, today_price, db_path)
    record: dict[str, object] = {
        "run_date": today_date_str,
        "current_price": today_price,
        "predicted_price": clean_pred,
        "predicted_change_pct": predicted_change_pct,
        "sentiment_score": sentiment_score,
        "article_count": article_count,
        "signal": signal,
    }
    upsert_forecast(record, db_path)

    # 7. Build today's 1-row forecast log and update forecast_log.csv idempotently
    new_log_row = pd.DataFrame(
        {
            "Date": [today_date_str],
            "Today_Gold_Price": [round(today_price, 2)],
            "Avg_Sentiment": [round(sentiment_score, 4)],
            "Article_Count": [article_count],
            "Predicted_Tomorrow_Price": [round(clean_pred, 2)],
            "Expected_Move_USD": [round(clean_move, 2)],
        }
    )
    update_csv_log(new_log_row, csv_path)

    print(
        f"Daily pipeline completed! Signal={signal} "
        f"(pred_change={predicted_change_pct:+.3f}%, "
        f"migrated={migrated_count}, backfilled={backfilled_count})"
    )
    print(new_log_row)
    return {
        **record,
        "migrated": migrated_count,
        "backfilled": backfilled_count,
    }


if __name__ == "__main__":
    main()
