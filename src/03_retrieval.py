# retrieval.py
# Implements three retrieval strategies over the prompt corpus:
# - dense: cosine similarity via FAISS on sentence-transformer embeddings
# - bm25: lexical BM25 scoring over tokenized semantic_text
# - hybrid: min-max normalised weighted fusion of dense + BM25 (alpha=0.65)
# The hybrid mode is the default for all downstream pipelines.
# Depends on: preprocessing.py (work, tokenized_for_bm25), embeddings.py (embedder, index, use_faiss)

bm25 = BM25Okapi(work["tokenized_for_bm25"].tolist())


def _normalize_scores(vals):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return vals
    vmin, vmax = vals.min(), vals.max()
    if np.isclose(vmin, vmax):
        return np.ones_like(vals)
    return (vals - vmin) / (vmax - vmin)


def retrieve_dense(query, top_k=20):
    qvec = embedder.encode([query], normalize_embeddings=True)
    qvec = np.asarray(qvec, dtype=np.float32)

    if use_faiss:
        sims, idxs = index.search(qvec, top_k)
        sims, idxs = sims[0], idxs[0]
    else:
        dists, idxs = index.kneighbors(qvec, n_neighbors=top_k)
        sims, idxs = 1 - dists[0], idxs[0]

    out = work.iloc[idxs].copy()
    out["dense_score"] = sims
    return out.reset_index(drop=True)


def retrieve_bm25(query, top_k=20):
    toks = re.findall(r"\b\w+\b", query.lower())
    scores = bm25.get_scores(toks)
    idxs = np.argsort(scores)[::-1][:top_k]
    out = work.iloc[idxs].copy()
    out["bm25_score"] = scores[idxs]
    return out.reset_index(drop=True)


def retrieve_hybrid(query, top_k=20, alpha=0.65):
    # alpha weights the dense side; 0.65 was the cheapest quality boost found
    # across all experiments — enough to prioritise semantics without ignoring
    # exact keyword matches from BM25.
    d = retrieve_dense(query, top_k=top_k * 4)[["id", "title", "semantic_text", "dense_score"]]
    b = retrieve_bm25(query, top_k=top_k * 4)[["id", "bm25_score"]]

    m = d.merge(b, on="id", how="outer").fillna(0)
    m["dense_n"] = _normalize_scores(m["dense_score"].values)
    m["bm25_n"] = _normalize_scores(m["bm25_score"].values)
    m["hybrid_score"] = alpha * m["dense_n"] + (1 - alpha) * m["bm25_n"]

    m = m.sort_values("hybrid_score", ascending=False).head(top_k)
    out = m.merge(work, on="id", how="left", suffixes=("", "_full"))
    return out.reset_index(drop=True)
