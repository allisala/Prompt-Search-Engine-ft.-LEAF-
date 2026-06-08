# preprocessing.py
# Loads the PromptKaban dataset from LEAF-promptkaban-dataset/dataset.json,
# runs exploratory data analysis (EDA), and builds the composite `semantic_text`
# field (title + category + subcategory + tags + content) that feeds both the
# dense embedder and the BM25 retriever.
#
# This module mirrors Chapters 1-2 of main.ipynb. Every figure is exported to
# assets/ as a transparent-background PNG via save_show(), so the charts stay
# visible on GitHub and match the notebook exactly.
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
from plotly.subplots import make_subplots

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

# Shared Plotly styling + figure export
# A fully transparent paper/plot background is what keeps the exported PNGs
# readable on both light and dark GitHub themes.
FIG = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=50, b=40),
)

# Resolve the project root so figures land in <root>/assets regardless of cwd.
PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "LEAF-promptkaban-dataset").exists():
    for parent in Path.cwd().resolve().parents:
        if (parent / "LEAF-promptkaban-dataset").exists():
            PROJECT_ROOT = parent
            break

ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
SAVE_FIGURES = True


def save_show(fig, filename: str, width=None, height=None, scale=2):
    """Display the figure and export it to assets/ as a transparent PNG."""
    fig.show()
    if not SAVE_FIGURES or ASSETS_DIR is None:
        return fig
    path_out = Path(ASSETS_DIR) / filename
    path_out.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"scale": scale}
    if width:
        kwargs["width"] = width
    if height:
        kwargs["height"] = height
    try:
        fig.write_image(str(path_out), **kwargs)
        print(f"Saved figure -> assets/{filename}")
    except Exception as exc:
        print(f"Figure not exported ({type(exc).__name__}): {filename}")
    return fig


def apply_fig_layout(fig, preset=None, **overrides):
    layout = dict(preset or FIG)
    layout.update(overrides)
    fig.update_layout(**layout)
    return fig


# Dataset loading
DATA_PATH = PROJECT_ROOT / "LEAF-promptkaban-dataset" / "dataset.json"
FIELDS_PATH = PROJECT_ROOT / "LEAF-promptkaban-dataset" / "FIELDS.md"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_json(DATA_PATH)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(sorted(df.columns.tolist()))

df.head(3)

# Exploratory Data Analysis
# Each chart ends with one concrete implication for a later pipeline stage.
eda = df.copy()
eda["created_at"] = pd.to_datetime(eda["created_at"], errors="coerce", utc=True)

# 1) How fast is the corpus growing?
# Creation is ~flat over time, so recency is not encoded in volume -> we add an
# explicit freshness decay in the metadata scorer.
weekly = (
    eda.dropna(subset=["created_at"])
    .assign(week=lambda x: x["created_at"].dt.to_period("W").dt.start_time)
    .groupby("week", as_index=False)
    .agg(prompts=("id", "count"))
)
fig = px.line(weekly, x="week", y="prompts",
              title="Prompts created per week",
              labels={"week": "week", "prompts": "count"})
apply_fig_layout(fig, FIG, height=320)
save_show(fig, "eda_prompts_per_week.png", height=320)
print(f"~{weekly['prompts'].mean():.0f} prompts/week on average — creation rate is roughly flat.")

# 2) How concentrated is the corpus across categories?
# Long tail of niche categories -> normalize popularity rather than use raw global counts.
cat_counts = df["category"].value_counts()
total = len(df)
top25 = cat_counts.head(25)
print(f"distinct categories : {df['category'].nunique()}")
print(f"top-25 share        : {top25.sum() / total * 100:.1f}%")
top25_df = top25.iloc[::-1].reset_index()
top25_df.columns = ["category", "prompts"]
fig = px.bar(top25_df, y="category", x="prompts", orientation="h",
             title="Top-25 categories by prompt count",
             labels={"prompts": "prompts", "category": "category"})
apply_fig_layout(fig, FIG, height=520, margin=dict(l=160, r=20, t=50, b=40), showlegend=False)
save_show(fig, "eda_top_categories.png", height=520)

# 3) Difficulty mix — skewed toward intermediate; used as a soft compatibility signal.
diff_order = ["beginner", "intermediate", "advanced", "expert"]
diff_counts = df["difficulty"].value_counts().reindex(diff_order)
fig = px.bar(diff_counts.reset_index(), x="difficulty", y="count",
             title="Prompts per difficulty",
             labels={"count": "count", "difficulty": "difficulty"})
apply_fig_layout(fig, FIG, height=300, showlegend=False)
save_show(fig, "eda_difficulty_distribution.png", height=300)

# 4) Engagement, on the right scale — heavy-tailed, so we view it on log1p.
ENG_COLS = ["likes", "upvotes", "downvotes", "views", "uses", "fork_count"]
log_eng = df[ENG_COLS].apply(np.log1p)
fig = go.Figure()
for col in ["likes", "upvotes", "views", "uses"]:
    fig.add_trace(go.Histogram(x=log_eng[col], name=col, opacity=0.55, nbinsx=40))
apply_fig_layout(
    fig,
    FIG,
    height=360,
    barmode="overlay",
    title="log(1+x) of engagement",
    xaxis_title="log(1+x)",
    yaxis_title="prompts",
)
save_show(fig, "eda_engagement_log_distribution.png", height=360)
med = df[ENG_COLS].median()
print(
    f"Median engagement (raw): ~{med['likes']:.0f} likes, ~{med['upvotes']:.0f} upvotes, "
    f"~{med['views']:.0f} views, ~{med['uses']:.0f} uses"
)

# 5) Are popular prompts actually useful? Proxy usefulness with uses / views.
sub = df[df["views"] >= 50].copy()
sub["use_rate"] = sub["uses"] / sub["views"]
sample_n = min(4000, len(sub))
fig = px.scatter(
    sub.sample(sample_n, random_state=1729),
    x="views", y="use_rate", color="difficulty",
    log_x=True, opacity=0.45,
    title="Use-rate vs view-count (sampled)",
    labels={"views": "views (log)", "use_rate": "uses / views"},
)
apply_fig_layout(fig, FIG, height=380)
save_show(fig, "eda_use_rate_vs_views.png", height=380)

# 6) The target_model deep dive — volume, engagement, category mix, difficulty mix.
target_counts = df["target_model"].value_counts()
fig = px.bar(target_counts.reset_index(), x="target_model", y="count",
             title="Prompts per target_model",
             labels={"count": "count", "target_model": "target_model"})
apply_fig_layout(fig, FIG, height=320, showlegend=False)
save_show(fig, "eda_prompts_per_target_model.png", height=320)

target_eng = (
    df.groupby("target_model")[["likes", "upvotes", "views", "uses"]]
    .mean()
    .round(1)
    .sort_values("likes", ascending=False)
)
fig = px.bar(
    target_eng.reset_index().melt(id_vars="target_model", var_name="metric", value_name="mean"),
    x="target_model", y="mean", color="metric", barmode="group",
    title="Mean engagement by target_model",
)
apply_fig_layout(fig, FIG, height=380)
fig.update_yaxes(type="log")
save_show(fig, "eda_engagement_by_target_model.png", height=380)
any_likes = target_eng.loc["any", "likes"] if "any" in target_eng.index else np.nan
model_mean = target_eng.drop(index="any", errors="ignore")["likes"].mean()
if pd.notna(any_likes) and model_mean:
    print(f"'any' mean likes: {any_likes:.0f}  vs  model-specific average: {model_mean:.0f}")

# Category mix and difficulty mix are nearly flat across models -> target_model
# is only a light compatibility signal, not a strong ranking signal.
top_cats = df["category"].value_counts().head(15).index.tolist()
sub_tm = df[df["category"].isin(top_cats)]
ct = pd.crosstab(sub_tm["target_model"], sub_tm["category"], normalize="index").round(3) * 100
ct = ct[top_cats]
fig = px.imshow(
    ct, aspect="auto", color_continuous_scale="Blues",
    title="% of each model's prompts in top-15 categories",
    labels=dict(x="category", y="target_model", color="%"),
)
apply_fig_layout(fig, FIG, height=420, margin=dict(l=40, r=20, t=50, b=80))
save_show(fig, "eda_model_category_heatmap.png", height=420)

ct_diff = pd.crosstab(df["target_model"], df["difficulty"], normalize="index").round(3) * 100
ct_diff = ct_diff[["beginner", "intermediate", "advanced", "expert"]]
fig = px.imshow(
    ct_diff, aspect="auto", color_continuous_scale="Viridis",
    title="% of each model's prompts at each difficulty level",
    labels=dict(x="difficulty", y="target_model", color="%"),
)
apply_fig_layout(fig, FIG, height=380)
save_show(fig, "eda_model_difficulty_heatmap.png", height=380)

# 7) How long are titles and bodies? Justifies the 700-character semantic_text cap.
df_len = pd.DataFrame({
    "title_chars": df["title"].str.len(),
    "content_chars": df["content"].str.len(),
})
fig = make_subplots(rows=1, cols=2, subplot_titles=("Title characters", "Content characters"))
fig.add_trace(go.Histogram(x=df_len["title_chars"], nbinsx=40, marker_color="#636efa", name="title"), row=1, col=1)
fig.add_trace(go.Histogram(x=df_len["content_chars"], nbinsx=40, marker_color="#ef553b", name="content"), row=1, col=2)
apply_fig_layout(
    fig,
    FIG,
    height=380,
    showlegend=False,
    title_text="Character-length distributions",
    margin=dict(l=40, r=20, t=70, b=40),
)
fig.update_xaxes(title_text="characters", row=1, col=1)
fig.update_xaxes(title_text="characters", row=1, col=2)
fig.update_yaxes(title_text="prompts", row=1, col=1)
save_show(fig, "eda_content_length_distribution.png", height=380)

# 8) Tags — another long tail.
all_tags = [t for tags in df["tags"] for t in tags]
tag_counts = pd.Series(all_tags).value_counts()
print(f"total tag tokens : {len(all_tags):,}")
print(f"unique tags      : {tag_counts.size:,}")

# 9) A few more checks — language and metadata correlations.
# Almost everything is English -> English-tuned MiniLM is appropriate.
lang = df["language"].value_counts().head(8).reset_index()
lang.columns = ["language", "count"]
fig = px.bar(lang, x="language", y="count", title="Language distribution (top 8)")
apply_fig_layout(fig, FIG, height=300)
save_show(fig, "eda_language_distribution.png", height=300)

# Engagement fields move together (one popularity signal); author_reputation is
# a separate quality signal.
corr_cols = ["likes", "upvotes", "downvotes", "views", "uses", "author_reputation", "fork_count"]
corr = df[corr_cols].corr()
fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Engagement & reputation correlations")
apply_fig_layout(fig, FIG, height=420)
save_show(fig, "eda_metadata_correlation.png", height=420)

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
tokenized_corpus = work["tokenized_for_bm25"].tolist()
work[["id", "title", "semantic_text"]].head(2)
