# reranking.py
# Two-stage relevance refinement after first-pass retrieval:
#  1. Cross-encoder reranker (ms-marco-MiniLM-L-6-v2): reads each
#     (query, candidate) pair together for a more accurate relevance score.
#  2. Metadata-aware scoring: blends the rerank score with a popularity
#     signal (likes, upvotes, views, uses), a quality proxy (author
#     reputation, fork count, version), and a freshness decay on created_at.
# Also provides query intent detection (coding/marketing/data-analysis/etc.)
# for adaptive weight selection, and MMR diversification to avoid near-duplicates.
# Depends on: preprocessing.py (work), retrieval.py (_normalize_scores, embedder)

# Intent detection and adaptive weights

def detect_query_intent(query: str):
    q = query.lower()
    if any(k in q for k in ["code", "python", "api", "debug", "bug", "sql"]):
        return "coding"
    if any(k in q for k in ["email", "campaign", "ad", "brand", "linkedin", "seo", "marketing"]):
        return "marketing"
    if any(k in q for k in ["analysis", "dashboard", "kpi", "forecast", "trend", "data"]):
        return "data-analysis"
    if any(k in q for k in ["story", "poem", "creative", "script", "fiction"]):
        return "creative-writing"
    return "general"


def intent_adaptive_weights(intent):
    if intent == "marketing":
        return dict(w_sem=0.64, w_pop=0.20, w_quality=0.10, w_fresh=0.06)
    if intent == "coding":
        # Coding queries downweight popularity: a technically correct prompt is
        # better than a popular but generic one.
        return dict(w_sem=0.78, w_pop=0.10, w_quality=0.09, w_fresh=0.03)
    if intent == "data-analysis":
        return dict(w_sem=0.74, w_pop=0.12, w_quality=0.10, w_fresh=0.04)
    if intent == "creative-writing":
        return dict(w_sem=0.70, w_pop=0.14, w_quality=0.08, w_fresh=0.08)
    return dict(w_sem=0.72, w_pop=0.14, w_quality=0.09, w_fresh=0.05)


def mmr_diversify(df_in, lambda_rel=0.80, top_k=10):
    if len(df_in) <= top_k:
        return df_in.copy()

    cand = df_in.copy().reset_index(drop=True)
    vecs = embedder.encode(cand["semantic_text"].tolist(), normalize_embeddings=True, show_progress_bar=False)
    vecs = np.asarray(vecs)

    selected = [0]
    remaining = set(range(1, len(cand)))
    rel = _normalize_scores(cand["final_score"].values)

    while len(selected) < min(top_k, len(cand)) and remaining:
        best_i, best_val = None, -1e9
        for i in list(remaining):
            sim_to_sel = max(np.dot(vecs[i], vecs[j]) for j in selected)
            mmr = lambda_rel * rel[i] - (1 - lambda_rel) * sim_to_sel
            if mmr > best_val:
                best_i, best_val = i, mmr
        selected.append(best_i)
        remaining.remove(best_i)

    out = cand.iloc[selected].copy()
    out["mmr_rank"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)

# Cross-encoder reranker and metadata scoring

reranker = CrossEncoder(RE_RANKER_NAME)

meta_cols = ["likes", "upvotes", "views", "uses", "author_reputation", "fork_count", "version"]
meta_df = work[meta_cols].fillna(0).copy()
meta_scaled = pd.DataFrame(MinMaxScaler().fit_transform(meta_df), columns=meta_cols)

work_meta = work.copy()
for c in meta_cols:
    work_meta[f"{c}_n"] = meta_scaled[c]

work_meta["created_at"] = pd.to_datetime(work_meta["created_at"], errors="coerce", utc=True)
max_dt = work_meta["created_at"].max()
work_meta["age_days"] = (max_dt - work_meta["created_at"]).dt.total_seconds() / 86400
work_meta["age_days"] = work_meta["age_days"].fillna(work_meta["age_days"].median())
# Exponential decay with 180-day half-life: a 6-month-old prompt retains ~37% freshness.
work_meta["freshness"] = np.exp(-work_meta["age_days"] / 180.0)


def rerank(query, candidates_df, top_k=20):
    pairs = [(query, txt) for txt in candidates_df["semantic_text"].tolist()]
    rr_scores = reranker.predict(pairs)
    out = candidates_df.copy()
    out["rerank_score"] = rr_scores
    out = out.sort_values("rerank_score", ascending=False).head(top_k)
    return out.reset_index(drop=True)


def add_metadata_score(df_in, w_sem=0.72, w_pop=0.14, w_quality=0.09, w_fresh=0.05):
    out = df_in.merge(
        work_meta[["id", "likes_n", "upvotes_n", "views_n", "uses_n", "author_reputation_n", "fork_count_n", "version_n", "freshness"]],
        on="id",
        how="left"
    )

    out["popularity"] = (
        0.45 * out["likes_n"]
        + 0.30 * out["upvotes_n"]
        + 0.15 * out["views_n"]
        + 0.10 * out["uses_n"]
    )
    out["quality"] = (
        0.6 * out["author_reputation_n"]
        + 0.25 * out["fork_count_n"]
        + 0.15 * out["version_n"]
    )

    sem = _normalize_scores(out["rerank_score"].values if "rerank_score" in out else out["dense_score"].values)
    out["semantic_n"] = sem

    out["final_score"] = (
        w_sem * out["semantic_n"]
        + w_pop * out["popularity"]
        + w_quality * out["quality"]
        + w_fresh * out["freshness"]
    )
    out = out.sort_values("final_score", ascending=False)
    return out.reset_index(drop=True)
