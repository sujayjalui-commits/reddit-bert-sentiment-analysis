# reddit-bert-sentiment-analysis

## 🎓 Reddit Post + Comment Sentiment Intelligence (Master's-Level Project)

This project is a research-oriented sentiment pipeline that uses the **EnsembleData Reddit scraping API** (not the official Reddit API) to:

1. Fetch subreddit posts.
2. Fetch comments under each post.
3. Run transformer-based sentiment inference on posts and comments.
4. Produce record-level and post-level sentiment outputs for analysis.

## Core features

- **External data source**: EnsembleData Reddit scraping API.
- **Hierarchical ingestion**: post + comments per post.
- **Transformer sentiment inference**: `cardiffnlp/twitter-roberta-base-sentiment`.
- **Batch inference with confidence scores**.
- **Post-level aggregate sentiment summaries**.
- **CSV export** for downstream statistics and modeling.

## Tech stack

- Python
- Streamlit
- Requests
- Transformers (Hugging Face)
- Pandas

## Setup

### 1) Install dependencies

```bash
pip install streamlit requests pandas transformers torch
```

### 2) Configure EnsembleData credentials

Set environment variables:

```bash
export ENSEMBLEDATA_BASE_URL="https://your-ensembledata-api-base-url"
export ENSEMBLEDATA_API_KEY="your_api_key"

# Optional configuration
export ENSEMBLEDATA_POSTS_ENDPOINT="/reddit/posts"
export ENSEMBLEDATA_COMMENTS_ENDPOINT="/reddit/comments"
export ENSEMBLEDATA_API_KEY_HEADER="x-api-key"
export ENSEMBLEDATA_POST_ID_PARAM="post_id"
export ENSEMBLEDATA_TIMEOUT_SECONDS="30"
```

You may also use `.streamlit/secrets.toml`:

```toml
ENSEMBLEDATA_BASE_URL = "https://your-ensembledata-api-base-url"
ENSEMBLEDATA_API_KEY = "your_api_key"
ENSEMBLEDATA_POSTS_ENDPOINT = "/reddit/posts"
ENSEMBLEDATA_COMMENTS_ENDPOINT = "/reddit/comments"
ENSEMBLEDATA_API_KEY_HEADER = "x-api-key"
ENSEMBLEDATA_POST_ID_PARAM = "post_id"
ENSEMBLEDATA_TIMEOUT_SECONDS = 30
```

### 3) Run the app

```bash
streamlit run app.py
```

## Notes on API mapping

Different scraping providers may return different JSON keys. This app includes flexible field mapping for common key variants such as:

- post id: `id`, `post_id`, `reddit_id`
- comment body: `body`, `text`, `content`
- timestamps: `created_utc`, `created_at`, `timestamp`

If your endpoint schema differs, adjust the field lists in normalization helpers inside `app.py`.
