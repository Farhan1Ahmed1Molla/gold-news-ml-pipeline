import os
import yfinance as yf
import feedparser
import pandas as pd
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression

# 1. Fetch 5 years of historical Gold Futures data
gold_ticker = yf.Ticker("GC=F")
df_gold = gold_ticker.history(period="5y")[['Close']].reset_index()
df_gold.columns = ['Date', 'Gold_price']
df_gold['Date'] = pd.to_datetime(df_gold['Date']).dt.tz_localize(None).dt.normalize()

# 2. Fetch live Gold news from Google News RSS
rss_url = "https://news.google.com/rss/search?q=gold+price+market&hl=en-US&gl=US&ceid=US:en"
feed = feedparser.parse(rss_url)
analyzer = SentimentIntensityAnalyzer()

news_list = []
for entry in feed.entries:
    pub_date = datetime(*entry.published_parsed[:6]).strftime('%Y-%m-%d')
    score = analyzer.polarity_scores(entry.title)['compound']
    news_list.append({'Date': pub_date, 'Headline': entry.title, 'Sentiment': score})

df_news = pd.DataFrame(news_list)

# 3. Aggregate news by date and merge with Gold prices
if not df_news.empty:
    df_news['Date'] = pd.to_datetime(df_news['Date']).dt.normalize()
    daily_sentiment = df_news.groupby('Date').agg(
        Avg_Sentiment=('Sentiment', 'mean'),
        Article_Count=('Headline', 'count')
    ).reset_index()
    df_merged = pd.merge(df_gold, daily_sentiment, on='Date', how='left')
else:
    df_merged = df_gold.copy()
    df_merged['Avg_Sentiment'] = 0.0
    df_merged['Article_Count'] = 0

df_merged['Avg_Sentiment'] = df_merged['Avg_Sentiment'].fillna(0.0)
df_merged['Article_Count'] = df_merged['Article_Count'].fillna(0)

# 4. Create target column (Tomorrow's Price) and split historical vs. live row
df_merged['Tommorow_Price'] = df_merged['Gold_price'].shift(-1)

historical_data = df_merged.dropna(subset=['Tommorow_Price'])
today_row = df_merged.tail(1)

clean_features = ['Gold_price', 'Avg_Sentiment']
X_clean = historical_data[clean_features]
y = historical_data['Tommorow_Price']
test_X_clean = today_row[clean_features]

# 5. Train de-biased Linear Regression model and predict tomorrow's price
clean_model = LinearRegression()
clean_model.fit(X_clean, y)

today_price = today_row['Gold_price'].values[0]
clean_pred = clean_model.predict(test_X_clean)[0]
clean_move = clean_pred - today_price
today_date_str = pd.to_datetime(today_row['Date'].values[0]).strftime('%Y-%m-%d')

# 6. Build today's 1-row forecast log
new_log_row = pd.DataFrame({
    'Date': [today_date_str],
    'Today_Gold_Price': [round(today_price, 2)],
    'Avg_Sentiment': [round(today_row['Avg_Sentiment'].values[0], 4)],
    'Article_Count': [int(today_row['Article_Count'].values[0])],
    'Predicted_Tomorrow_Price': [round(clean_pred, 2)],
    'Expected_Move_USD': [round(clean_move, 2)]
})

# 7. Append to forecast_log.csv without duplicating the same date
csv_file = 'forecast_log.csv'
if os.path.exists(csv_file):
    existing_log = pd.read_csv(csv_file)
    # Remove today's row if it already exists so re-runs update cleanly
    existing_log = existing_log[existing_log['Date'] != today_date_str]
    updated_log = pd.concat([existing_log, new_log_row], ignore_index=True)
else:
    updated_log = new_log_row

updated_log.to_csv(csv_file, index=False)
print("Daily pipeline completed! Latest row saved:")
print(new_log_row)



  
    

  
