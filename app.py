# ============================================
# Real-Time Reddit Sentiment Analysis Using BERT
# Final Year Project (Python)
# ============================================

import praw
import pandas as pd
import streamlit as st
import time
from transformers import pipeline

# -------------------------------
# Reddit API Configuration
# -------------------------------
CLIENT_ID = "YOUR_CLIENT_ID"
CLIENT_SECRET = "YOUR_CLIENT_SECRET"
USER_AGENT = "final-year-bert-sentiment-project"

reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# -------------------------------
# BERT Sentiment Model
# -------------------------------
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

# -------------------------------
# Utility Functions
# -------------------------------
def predict_sentiment(text):
    try:
        result = sentiment_model(text[:512])[0]
        return result["label"]
    except Exception:
        return "UNKNOWN"

def load_data():
    try:
        return pd.read_csv("live_reddit_sentiment.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["Text", "Sentiment", "Subreddit"])

def save_data(df):
    df.to_csv("live_reddit_sentiment.csv", index=False)

# -------------------------------
# Streamlit UI Setup
# -------------------------------
st.set_page_config(page_title="Reddit Sentiment Analysis", layout="wide")
st.title("📊 Real-Time Reddit Sentiment Analysis Using BERT")

subreddit_name = st.text_input("Enter Subreddit Name", "technology")
start_stream = st.button("Start Live Analysis")

col1, col2 = st.columns(2)

# -------------------------------
# Live Data Streaming
# -------------------------------
if start_stream:
    subreddit = reddit.subreddit(subreddit_name)
    data = load_data()

    st.success(f"Streaming live posts from r/{subreddit_name}")

    for post in subreddit.stream.submissions(skip_existing=True):
        text = (post.title + " " + post.selftext).strip()
        if len(text) == 0:
            continue

        sentiment = predict_sentiment(text)

        new_row = {
            "Text": post.title,
            "Sentiment": sentiment,
            "Subreddit": subreddit_name
        }

        data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
        save_data(data)

        with col1:
            st.subheader("Sentiment Distribution")
            st.bar_chart(data["Sentiment"].value_counts())

        with col2:
            st.subheader("Latest Reddit Posts")
            st.dataframe(data.tail(10), use_container_width=True)

        time.sleep(2)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    "Final Year Project | Reddit API (Free Tier) | BERT Sentiment Model | Streamlit Dashboard"
)
