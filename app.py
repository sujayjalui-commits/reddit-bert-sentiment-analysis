from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from typing import Any, Iterable

import pandas as pd
import requests
import streamlit as st
from transformers import pipeline


# ---------------------------------
# Configuration and constants
# ---------------------------------
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
LABEL_MAP = {"LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE"}
SENTIMENT_SCORE_MAP = {"NEGATIVE": -1.0, "NEUTRAL": 0.0, "POSITIVE": 1.0, "UNKNOWN": 0.0}


@dataclass
class AppConfig:
    base_url: str
    api_key: str
    posts_endpoint: str
    comments_endpoint: str
    api_key_header: str
    post_id_param: str
    timeout_seconds: int = 30


# ---------------------------------
# Resource loaders
# ---------------------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return pipeline("sentiment-analysis", model=MODEL_NAME)


@st.cache_resource(show_spinner=False)
def init_http_session() -> requests.Session:
    return requests.Session()


# ---------------------------------
# Core NLP + data processing
# ---------------------------------
def normalize_label(raw_label: str) -> str:
    return LABEL_MAP.get(raw_label, raw_label)


def predict_sentiments(texts: Iterable[str], model, batch_size: int = 16) -> list[tuple[str, float]]:
    prepared = [t[:512] if t else "" for t in texts]
    results: list[tuple[str, float]] = []

    for i in range(0, len(prepared), batch_size):
        batch = prepared[i : i + batch_size]
        try:
            predictions = model(batch)
            for pred in predictions:
                label = normalize_label(pred.get("label", "UNKNOWN"))
                confidence = float(pred.get("score", 0.0))
                results.append((label, confidence))
        except Exception:
            results.extend([("UNKNOWN", 0.0)] * len(batch))

    return results


def _first_value(item: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return default


def _parse_datetime(value: Any) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)

    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)

    if isinstance(value, str):
        maybe = value.strip()
        if maybe.isdigit():
            return datetime.fromtimestamp(float(maybe), tz=timezone.utc)
        try:
            normalized = maybe.replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return datetime.now(timezone.utc)

    return datetime.now(timezone.utc)


def _extract_payload_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        for key in ["data", "results", "items", "posts", "comments"]:
            candidate = payload.get(key)
            if isinstance(candidate, list):
                return [x for x in candidate if isinstance(x, dict)]

    return []


def _normalize_post_record(post: dict[str, Any], subreddit_name: str) -> dict[str, Any]:
    post_id = str(_first_value(post, ["id", "post_id", "name", "reddit_id"], ""))
    title = str(_first_value(post, ["title", "post_title"], ""))
    body = str(_first_value(post, ["selftext", "body", "text", "content"], ""))
    permalink = str(_first_value(post, ["permalink", "url", "post_url"], ""))

    url = permalink
    if permalink.startswith("/"):
        url = f"https://reddit.com{permalink}"

    return {
        "entity_type": "post",
        "subreddit": str(_first_value(post, ["subreddit", "subreddit_name"], subreddit_name)),
        "post_id": post_id,
        "parent_post_id": post_id,
        "author": str(_first_value(post, ["author", "author_name", "username"], "[deleted]")),
        "created_utc": _parse_datetime(_first_value(post, ["created_utc", "created_at", "timestamp"])),
        "score": int(_first_value(post, ["score", "upvotes", "ups"], 0) or 0),
        "text": f"{title} {body}".strip(),
        "title": title,
        "url": url,
    }


def _normalize_comment_record(comment: dict[str, Any], subreddit_name: str, fallback_post: dict[str, Any]) -> dict[str, Any]:
    comment_id = str(_first_value(comment, ["id", "comment_id", "name", "reddit_id"], ""))
    body = str(_first_value(comment, ["body", "text", "content"], ""))
    permalink = str(_first_value(comment, ["permalink", "url", "comment_url"], ""))

    url = permalink
    if permalink.startswith("/"):
        url = f"https://reddit.com{permalink}"

    parent_post_id = str(
        _first_value(
            comment,
            ["post_id", "link_id", "parent_post_id", "submission_id"],
            fallback_post.get("post_id", ""),
        )
    )

    return {
        "entity_type": "comment",
        "subreddit": str(_first_value(comment, ["subreddit", "subreddit_name"], subreddit_name)),
        "post_id": comment_id,
        "parent_post_id": parent_post_id,
        "author": str(_first_value(comment, ["author", "author_name", "username"], "[deleted]")),
        "created_utc": _parse_datetime(_first_value(comment, ["created_utc", "created_at", "timestamp"])),
        "score": int(_first_value(comment, ["score", "upvotes", "ups"], 0) or 0),
        "text": body,
        "title": fallback_post.get("title", ""),
        "url": url,
    }


def _build_headers(config: AppConfig) -> dict[str, str]:
    return {
        config.api_key_header: config.api_key,
        "Accept": "application/json",
    }


def _get_json(session: requests.Session, url: str, headers: dict[str, str], params: dict[str, Any], timeout_seconds: int) -> Any:
    response = session.get(url, headers=headers, params=params, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()


def fetch_dataset(
    session: requests.Session,
    config: AppConfig,
    subreddit_name: str,
    sort_mode: str,
    post_limit: int,
    comment_limit: int,
) -> pd.DataFrame:
    headers = _build_headers(config)

    posts_url = f"{config.base_url.rstrip('/')}/{config.posts_endpoint.lstrip('/')}"
    comments_url = f"{config.base_url.rstrip('/')}/{config.comments_endpoint.lstrip('/')}"

    posts_payload = _get_json(
        session=session,
        url=posts_url,
        headers=headers,
        params={"subreddit": subreddit_name, "sort": sort_mode, "limit": post_limit},
        timeout_seconds=config.timeout_seconds,
    )

    posts = _extract_payload_list(posts_payload)
    rows: list[dict[str, Any]] = []

    for post in posts:
        post_record = _normalize_post_record(post, subreddit_name)
        if post_record["text"]:
            rows.append(post_record)

        if not post_record["post_id"]:
            continue

        comments_payload = _get_json(
            session=session,
            url=comments_url,
            headers=headers,
            params={config.post_id_param: post_record["post_id"], "limit": comment_limit},
            timeout_seconds=config.timeout_seconds,
        )
        comments = _extract_payload_list(comments_payload)

        for comment in comments[:comment_limit]:
            comment_record = _normalize_comment_record(comment, subreddit_name, post_record)
            if comment_record["text"].strip():
                rows.append(comment_record)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df[df["text"].str.strip().astype(bool)].copy()


def enrich_with_sentiment(df: pd.DataFrame, model) -> pd.DataFrame:
    if df.empty:
        return df

    preds = predict_sentiments(df["text"].tolist(), model)
    sentiments, confidences = zip(*preds)

    enriched = df.copy()
    enriched["sentiment"] = sentiments
    enriched["confidence"] = confidences
    enriched["sentiment_score"] = enriched["sentiment"].map(SENTIMENT_SCORE_MAP).fillna(0.0)
    return enriched


def post_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby("parent_post_id", as_index=False)
        .agg(
            subreddit=("subreddit", "first"),
            title=("title", "first"),
            post_url=("url", "first"),
            post_count=("entity_type", lambda x: int((x == "post").sum())),
            comment_count=("entity_type", lambda x: int((x == "comment").sum())),
            avg_sentiment_score=("sentiment_score", "mean"),
            avg_confidence=("confidence", "mean"),
        )
        .sort_values("avg_sentiment_score", ascending=False)
    )

    grouped["post_sentiment"] = grouped["avg_sentiment_score"].apply(
        lambda x: "POSITIVE" if x > 0.15 else ("NEGATIVE" if x < -0.15 else "NEUTRAL")
    )
    return grouped


def ensemble_config_from_env_or_secrets() -> AppConfig | None:
    base_url = os.getenv("ENSEMBLEDATA_BASE_URL")
    api_key = os.getenv("ENSEMBLEDATA_API_KEY")
    posts_endpoint = os.getenv("ENSEMBLEDATA_POSTS_ENDPOINT", "/reddit/posts")
    comments_endpoint = os.getenv("ENSEMBLEDATA_COMMENTS_ENDPOINT", "/reddit/comments")
    api_key_header = os.getenv("ENSEMBLEDATA_API_KEY_HEADER", "x-api-key")
    post_id_param = os.getenv("ENSEMBLEDATA_POST_ID_PARAM", "post_id")
    timeout_seconds = int(os.getenv("ENSEMBLEDATA_TIMEOUT_SECONDS", "30"))

    if (not base_url or not api_key) and hasattr(st, "secrets"):
        base_url = base_url or st.secrets.get("ENSEMBLEDATA_BASE_URL")
        api_key = api_key or st.secrets.get("ENSEMBLEDATA_API_KEY")
        posts_endpoint = st.secrets.get("ENSEMBLEDATA_POSTS_ENDPOINT", posts_endpoint)
        comments_endpoint = st.secrets.get("ENSEMBLEDATA_COMMENTS_ENDPOINT", comments_endpoint)
        api_key_header = st.secrets.get("ENSEMBLEDATA_API_KEY_HEADER", api_key_header)
        post_id_param = st.secrets.get("ENSEMBLEDATA_POST_ID_PARAM", post_id_param)
        timeout_seconds = int(st.secrets.get("ENSEMBLEDATA_TIMEOUT_SECONDS", timeout_seconds))

    if not base_url or not api_key:
        return None

    return AppConfig(
        base_url=base_url,
        api_key=api_key,
        posts_endpoint=posts_endpoint,
        comments_endpoint=comments_endpoint,
        api_key_header=api_key_header,
        post_id_param=post_id_param,
        timeout_seconds=timeout_seconds,
    )


# ---------------------------------
# Streamlit UI
# ---------------------------------
st.set_page_config(page_title="Reddit Sentiment Research Dashboard", layout="wide")
st.title("🎓 Reddit Post + Comment Sentiment Intelligence")
st.caption(
    "Research dashboard using EnsembleData Reddit scraping API + transformers for "
    "post/comment sentiment modeling and post-level aggregation."
)

with st.sidebar:
    st.header("Collection settings")
    subreddit_name = st.text_input("Subreddit", value="technology").strip() or "technology"
    sort_mode = st.selectbox("Post ranking", options=["hot", "new", "top"], index=0)
    post_limit = st.slider("Number of posts", min_value=5, max_value=50, value=15)
    comment_limit = st.slider("Max comments per post", min_value=5, max_value=100, value=25)
    run_analysis = st.button("Fetch and analyze")

config = ensemble_config_from_env_or_secrets()
if config is None:
    st.error(
        "Missing EnsembleData configuration. Set ENSEMBLEDATA_BASE_URL and ENSEMBLEDATA_API_KEY "
        "(environment variables or .streamlit/secrets.toml)."
    )
    st.stop()

if run_analysis:
    try:
        with st.spinner("Loading model, fetching posts/comments from EnsembleData, running sentiment inference..."):
            session = init_http_session()
            model = load_sentiment_model()
            raw_df = fetch_dataset(session, config, subreddit_name, sort_mode, post_limit, comment_limit)
            data = enrich_with_sentiment(raw_df, model)
    except requests.HTTPError as exc:
        st.error(f"EnsembleData API HTTP error: {exc}")
        st.stop()
    except requests.RequestException as exc:
        st.error(f"Network error while calling EnsembleData API: {exc}")
        st.stop()

    if data.empty:
        st.warning("No textual posts/comments found for the selected subreddit configuration.")
        st.stop()

    post_summary = post_level_summary(data)

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Analyzed records", f"{len(data):,}")
    kpi2.metric("Posts", f"{(data['entity_type'] == 'post').sum():,}")
    kpi3.metric("Comments", f"{(data['entity_type'] == 'comment').sum():,}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Sentiment distribution")
        st.bar_chart(data["sentiment"].value_counts())
    with c2:
        st.subheader("Entity distribution")
        st.bar_chart(data["entity_type"].value_counts())

    st.subheader("Post-level sentiment summary")
    st.dataframe(
        post_summary[
            [
                "parent_post_id",
                "title",
                "comment_count",
                "avg_sentiment_score",
                "avg_confidence",
                "post_sentiment",
            ]
        ],
        use_container_width=True,
    )

    st.subheader("Record-level output (posts + comments)")
    st.dataframe(
        data[
            [
                "entity_type",
                "parent_post_id",
                "author",
                "score",
                "sentiment",
                "confidence",
                "text",
                "url",
            ]
        ].sort_values(["parent_post_id", "entity_type"]),
        use_container_width=True,
        height=400,
    )

    csv_bytes = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download record-level CSV",
        data=csv_bytes,
        file_name=f"reddit_sentiment_{subreddit_name}.csv",
        mime="text/csv",
    )

st.markdown("---")
st.markdown("Built for advanced academic sentiment research with EnsembleData + Transformers + Streamlit.")
