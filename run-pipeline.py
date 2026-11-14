import os
import yfinance as yf
import pandas as pd
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression

gold_ticker = yf.Ticker("GC=F")
df_gold = gold_ticker.history(period="5y")[['Close']].reset_index()
df_gold.columns = ['Date', 'Gold_price']
df_gold['Date'] = pd.to_datetime(df_gold['Date']).dt.tz_localize(None).dt.normalize()

rssl_url = "https://news.google.com/rss/search?q=gold+price+market&hl=en-US&gl=US&cied=US:en"
feed = feedparser.parse(rss_url)
analyzer = SentimentIntensityAnalyzer()

news_list = []
for entry in feed.entries:
  pub_date = datetime(*entry.published_parsed[:6]).strftime('%Y-%m-%d')
  score = analyzer.polarity_scores(entry.title)['compound']
  news_list.append({'Date': pub_date, 'Headline': entry.title, 'Sentiment: score})

df_news = pd.DataFrame(news_list)

if not df_news.empty:
  df_news['Date'] = pd.to_datetime(df_news['Date']).dt.normalize()
  daily_sentiment = df_news.groupby(Date').agg(
    Avg_Sentiment=('Sentiment', 'mean'),
    Article_Count=('Headline', 'count')
    ).reset_index()
    df.merged= pd.merge(df_gold, daily_sentiment, on='Date', how='left')
else:
  df_merged = df_gold.copy()
  df_merged['Avg_Sentiment'] = 0.0
  df_merged['Article_Count'] = 0

df_merged['Avg_Sentiment'] = df_merged['Avg_Sentiment'].fillna(0.0)
df_merged['Article_Count'] = df_merged['Article_Count'].fillna(0.0)


  
    

  
