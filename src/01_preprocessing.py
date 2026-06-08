# preprocessing.py
# Loads the PromptKaban dataset from LEAF-promptkaban-dataset/dataset.json,
# runs exploratory feature engineering (engagement score, content length, etc.),
# and builds the composite `semantic_text` field (title + category + subcategory
# + tags + content) that feeds both the dense embedder and the BM25 retriever.
# Run this first, all other modules depend on `work` and `tokenized_corpus`.

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Prefer static PNG outputs so charts remain visible on GitHub.
# If Kaleido/Chrome is unavailable, fall back to the normal VS Code/Jupyter renderer.
try:
    import plotly.express as _px_test
    _fig_test = _px_test.bar(x=["test"], y=[1])
    _fig_test.to_image(format="png")
    pio.renderers.default = "png"
    print("Plotly renderer: png")
except Exception as exc:
    pio.renderers.default = "notebook_connected"
    print("Plotly renderer: notebook_connected (PNG export unavailable)")
    print("PNG export issue:", type(exc).__name__)

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from rank_bm25 import BM25Okapi

from sentence_transformers import SentenceTransformer, CrossEncoder

# Dataset loading 

DATA_PATH = Path("LEAF-promptkaban-dataset/dataset.json")
FIELDS_PATH = Path("LEAF-promptkaban-dataset/FIELDS.md")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_json(DATA_PATH)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(sorted(df.columns.tolist()))

df.head(3)

# Exploratory Data Analysis

eda = df.copy()
eda["created_at"] = pd.to_datetime(eda["created_at"], errors="coerce", utc=True)
eda["content_len"] = eda["content"].fillna("").str.len()
eda["title_len"] = eda["title"].fillna("").str.len()
eda["tags_count"] = eda["tags"].apply(lambda x: len(x) if isinstance(x, list) else 0)
eda["engagement_score"] = (
    0.35 * np.log1p(eda["likes"].fillna(0))
    + 0.30 * np.log1p(eda["upvotes"].fillna(0))
    + 0.20 * np.log1p(eda["uses"].fillna(0))
    + 0.15 * np.log1p(eda["views"].fillna(0))
)

# 1) Category coverage
cat = eda["category"].value_counts().head(20).reset_index()
cat.columns = ["category", "count"]
fig_cat = px.bar(cat, x="category", y="count", title="Top 20 Categories")
fig_cat.update_layout(xaxis_tickangle=-40)
fig_cat.show()

# 2) Language distribution (bar, not pie)
lang = eda["language"].value_counts().head(12).reset_index()
lang.columns = ["language", "count"]
fig_lang = px.bar(lang, x="language", y="count", title="Language Distribution (Top 12)")
fig_lang.show()

# 3) Text complexity distribution
fig_len = px.histogram(
    eda,
    x="content_len",
    nbins=80,
    title="Prompt Content Length Distribution",
    marginal="box"
)
fig_len.show()

# 4) Engagement trend over time
monthly = (
    eda.dropna(subset=["created_at"])
    .assign(month=lambda x: x["created_at"].dt.to_period("M").astype(str))
    .groupby("month", as_index=False)
    .agg(prompts=("id", "count"), mean_engagement=("engagement_score", "mean"))
)
fig_month = px.line(monthly, x="month", y=["prompts", "mean_engagement"], title="Volume and Mean Engagement Over Time")
fig_month.update_layout(xaxis_tickangle=-40)
fig_month.show()

# 5) Engagement by category (quality signal)
cat_eng = (
    eda.groupby("category", as_index=False)
    .agg(mean_engagement=("engagement_score", "mean"), prompts=("id", "count"))
    .sort_values("mean_engagement", ascending=False)
    .head(15)
)
fig_cat_eng = px.bar(cat_eng, x="category", y="mean_engagement", color="prompts", title="Top Categories by Mean Engagement")
fig_cat_eng.update_layout(xaxis_tickangle=-40)
fig_cat_eng.show()

# 6) Correlation heatmap (metadata behavior)
corr_cols = ["likes", "upvotes", "downvotes", "views", "uses", "author_reputation", "fork_count", "version", "content_len", "tags_count"]
corr = eda[corr_cols].fillna(0).corr()
fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Metadata Correlation Heatmap")
fig_corr.show()

# 7) Semantic landscape preview via PCA on sample embeddings
sample_n = min(3000, len(eda))
sidx = np.random.RandomState(42).choice(len(eda), size=sample_n, replace=False)
sample_text = (
    "Title: " + eda.iloc[sidx]["title"].fillna("").astype(str) + " | "
    + eda.iloc[sidx]["content"].fillna("").astype(str)
).tolist()
sample_emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode(sample_text, batch_size=128, show_progress_bar=False)
xy = PCA(n_components=2, random_state=42).fit_transform(sample_emb)
plot_sem = pd.DataFrame({
    "x": xy[:, 0],
    "y": xy[:, 1],
    "category": eda.iloc[sidx]["category"].fillna("unknown").values,
    "difficulty": eda.iloc[sidx]["difficulty"].fillna("unknown").values,
    "title": eda.iloc[sidx]["title"].fillna("").values,
})
fig_sem = px.scatter(
    plot_sem,
    x="x",
    y="y",
    color="category",
    symbol="difficulty",
    hover_data=["title"],
    opacity=0.6,
    title="Semantic Map Preview (PCA projection of prompt embeddings)"
)
fig_sem.show()

# Semantic text construction 

def safe_text(x):
    if isinstance(x, list):
        return " ".join(map(str, x))
    if pd.isna(x):
        return ""
    return str(x)

work = df.copy()
work["semantic_text"] = (
    "Title: " + work["title"].map(safe_text) + "\n"
    + "Category: " + work["category"].map(safe_text) + "\n"
    + "Subcategory: " + work["subcategory"].map(safe_text) + "\n"
    + "Tags: " + work["tags"].map(safe_text) + "\n"
    + "Prompt: " + work["content"].map(safe_text)
)

# BM25 tokenization shares the same view as the dense embedder for fair comparison.
work["tokenized_for_bm25"] = work["semantic_text"].str.lower().str.findall(r"\b\w+\b")
work[["id", "title", "semantic_text"]].head(2)
