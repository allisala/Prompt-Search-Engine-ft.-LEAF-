# evaluation.py
# Evaluation harness for the semantic search pipeline.
# Implements three metrics: Precision@K, MRR (Mean Reciprocal Rank),
# and nDCG@10 (graded relevance). Provides two gold query sets:
#  - gold_eval_queries: 12 queries with category + keyword grading
#  - STRICT_GOLD_QUERIES: stricter version using word-boundary matching
#    and requiring >=2 term hits, correcting for evaluation leakage found
#    in the first benchmark (see Chapter 11 Audit in the notebook).
# Also includes the embedding model head-to-head benchmark, the random-search
# weight tuning cache, and the three-pipeline A/B comparison runner
# (baseline vs agentic-full vs agentic-no-rerank).
# Depends on: preprocessing.py, retrieval.py, reranking.py, pipeline.py, query_rewriter.py

from IPython.display import display

# Basic metrics and initial evaluation

eval_queries = [
    ("debug a slow python api endpoint", "coding"),
    ("write a linkedin post for product launch", "marketing"),
    ("analyze monthly sales trends and anomalies", "data-analysis"),
    ("create a bedtime story for kids", "creative-writing"),
    ("professional email to customer support", "email"),
]


def precision_at_k(pred_categories, true_category, k=10):
    top = pred_categories[:k]
    return np.mean([1 if c == true_category else 0 for c in top])


def reciprocal_rank(pred_categories, true_category):
    for i, c in enumerate(pred_categories, start=1):
        if c == true_category:
            return 1.0 / i
    return 0.0

rows = []
for q, true_cat in eval_queries:
    out, lat = search_pipeline(q, top_k=10, mode="hybrid")
    cats = out["category"].fillna("unknown").tolist()
    rows.append({
        "query": q,
        "expected_category": true_cat,
        "P@10": precision_at_k(cats, true_cat, k=10),
        "RR": reciprocal_rank(cats, true_cat),
        "latency_ms": lat,
    })

ev = pd.DataFrame(rows)
print("Mean Precision@10:", round(ev["P@10"].mean(), 4))
print("MRR:", round(ev["RR"].mean(), 4))
print("Avg latency (ms):", round(ev["latency_ms"].mean(), 2))
ev

# Embedding model comparison

def evaluate_current_pipeline(eval_set, adaptive=False, diversify=False):
    rows = []
    for q, true_cat in eval_set:
        out, lat = search_pipeline(q, top_k=10, mode="hybrid", adaptive=adaptive, diversify=diversify)
        cats = out["category"].fillna("unknown").tolist()
        rows.append({
            "query": q,
            "P@10": precision_at_k(cats, true_cat, 10),
            "RR": reciprocal_rank(cats, true_cat),
            "latency_ms": lat,
        })
    e = pd.DataFrame(rows)
    return {
        "Mean Precision@10": float(e["P@10"].mean()),
        "MRR": float(e["RR"].mean()),
        "Avg latency (ms)": float(e["latency_ms"].mean()),
    }

benchmark_rows = []
base_texts = work["semantic_text"].tolist()

for model_name in CANDIDATE_EMBED_MODELS:
    embedder, emb, index, use_faiss = build_embedding_index(
        model_name,
        base_texts,
        max_chars=700,
        batch_size=64,
    )

    m_base = evaluate_current_pipeline(eval_queries, adaptive=False, diversify=False)
    m_adv = evaluate_current_pipeline(eval_queries, adaptive=True, diversify=True)

    benchmark_rows.append({"model": model_name, "setting": "precision_default", **m_base})
    benchmark_rows.append({"model": model_name, "setting": "adaptive_plus_diversity", **m_adv})

bench = pd.DataFrame(benchmark_rows).sort_values(["MRR", "Mean Precision@10"], ascending=False)
bench

# Benchmark visualisation and model selection

bench_display = bench.copy()
bench_display["Mean Precision@10"] = bench_display["Mean Precision@10"].round(3)
bench_display["MRR"] = bench_display["MRR"].round(3)
bench_display["Avg latency (ms)"] = bench_display["Avg latency (ms)"].round(1)
display(bench_display)

quality_long = bench.melt(
    id_vars=["model", "setting", "Avg latency (ms)"],
    value_vars=["Mean Precision@10", "MRR"],
    var_name="metric",
    value_name="score",
)
quality_long["score_label"] = quality_long["score"].round(2).astype(str)

fig_quality = px.bar(
    quality_long,
    x="model",
    y="score",
    color="metric",
    facet_col="setting",
    barmode="group",
    text="score_label",
    title="Embedding Model Quality: Precision@10 and MRR",
)
fig_quality.update_yaxes(range=[0, 1], title="score")
fig_quality.update_xaxes(tickangle=-25)
fig_quality.update_traces(textposition="outside")
fig_quality.show()

fig_tradeoff = px.scatter(
    bench,
    x="Avg latency (ms)",
    y="MRR",
    size="Mean Precision@10",
    color="model",
    symbol="setting",
    hover_data={
        "Mean Precision@10": ":.3f",
        "MRR": ":.3f",
        "Avg latency (ms)": ":.1f",
    },
    title="Quality-Latency Tradeoff by Embedding Model",
)
fig_tradeoff.update_yaxes(range=[0, 1], title="MRR (higher is better)")
fig_tradeoff.update_xaxes(title="average latency, ms (lower is better)")
fig_tradeoff.show()

best_row = bench.iloc[0]
BEST_EMBED_MODEL = best_row["model"]
BEST_SETTING = best_row["setting"]
print("Selected model:", BEST_EMBED_MODEL)
print("Selected setting:", BEST_SETTING)
print("Selected MRR:", round(best_row["MRR"], 3))
print("Selected Precision@10:", round(best_row["Mean Precision@10"], 3))
print("Selected avg latency (ms):", round(best_row["Avg latency (ms)"], 1))

# Validation metric visualisation

fig_eval = px.bar(ev, x="query", y=["P@10", "RR"], barmode="group", title="Validation Metrics by Query")
fig_eval.update_layout(xaxis_tickangle=-30)
fig_eval.show()

fig_lat = px.line(ev, x="query", y="latency_ms", markers=True, title="Latency per Query (ms)")
fig_lat.update_layout(xaxis_tickangle=-30)
fig_lat.show()

# Random-search weight tuning 


def build_weight_tuning_cache(queries, retrieve_k=60, rerank_k=40):
    cached = []
    for q, true_cat in queries:
        stage1 = retrieve_hybrid(q, top_k=retrieve_k)
        stage2 = rerank(q, stage1, top_k=rerank_k)
        cached.append({
            "query": q,
            "true_cat": true_cat,
            "reranked_candidates": stage2,
        })
    return cached


def evaluate_weight_config_fast(weight_cfg, cached_queries):
    rows = []
    for item in cached_queries:
        stage3 = add_metadata_score(item["reranked_candidates"], **weight_cfg).head(10)
        cats = stage3["category"].fillna("unknown").tolist()
        rows.append(reciprocal_rank(cats, item["true_cat"]))
    return float(np.mean(rows))


# Seed fixed at 42 for reproducibility of the random weight search.
rng = np.random.RandomState(42)
search_space = []
for _ in range(25):
    w_sem = rng.uniform(0.58, 0.85)
    w_pop = rng.uniform(0.05, 0.25)
    w_quality = rng.uniform(0.04, 0.18)
    w_fresh = rng.uniform(0.02, 0.10)
    s = w_sem + w_pop + w_quality + w_fresh
    search_space.append(dict(
        w_sem=w_sem / s,
        w_pop=w_pop / s,
        w_quality=w_quality / s,
        w_fresh=w_fresh / s,
    ))

print("Precomputing reranked validation candidates once...")
t0 = time.perf_counter()
tuning_cache = build_weight_tuning_cache(eval_queries, retrieve_k=60, rerank_k=40)
print(f"Cache built in {(time.perf_counter() - t0):.1f}s")

scores = []
for cfg in search_space:
    mrr_val = evaluate_weight_config_fast(cfg, tuning_cache)
    scores.append({**cfg, "MRR": mrr_val})

opt_df = pd.DataFrame(scores).sort_values("MRR", ascending=False).reset_index(drop=True)
best_cfg = opt_df.iloc[0].to_dict()
print("Best config:")
print(best_cfg)

fig_opt = px.scatter(
    opt_df,
    x="w_sem",
    y="MRR",
    color="w_pop",
    size="w_quality",
    hover_data=["w_fresh"],
    title="Weight Search Landscape (Fast Random Search over Metadata Fusion)"
)
fig_opt.show()

opt_df.head(10)

# Gold evaluation harness (12 queries, graded relevance) 

# gold_eval_queries: mixes vocabulary-mismatch cases, in-vocabulary cases,
# and ambiguous label-space probes to give agentic rewriting a fair chance.
gold_eval_queries = [
    {"query": "draft a contract clause for a software license",
     "expected_category": None,
     "relevance_keywords": ["contract", "clause", "legal", "agreement", "terms", "license"]},
    {"query": "ai assistant for terms and conditions review",
     "expected_category": None,
     "relevance_keywords": ["legal", "contract", "terms", "policy", "compliance", "review"]},
    {"query": "improve seo of my landing page",
     "expected_category": "marketing",
     "relevance_keywords": ["seo", "search", "ranking", "google", "keyword", "meta"]},
    {"query": "fix a bug in my javascript code",
     "expected_category": "coding",
     "relevance_keywords": ["javascript", "js", "bug", "debug", "error", "fix", "frontend"]},
    {"query": "summarize a research paper for a non-expert",
     "expected_category": None,
     "relevance_keywords": ["summary", "summarize", "abstract", "paper", "research", "explain", "tldr"]},
    {"query": "write a marketing email to existing customers",
     "expected_category": "marketing",
     "relevance_keywords": ["email", "marketing", "newsletter", "campaign", "customer"]},
    {"query": "creative bedtime story for a 5 year old",
     "expected_category": "creative-writing",
     "relevance_keywords": ["story", "bedtime", "kids", "children", "fairy", "tale", "narrative"]},
    {"query": "monthly sales dashboard analysis",
     "expected_category": "data-analysis",
     "relevance_keywords": ["sales", "dashboard", "kpi", "analysis", "report", "trend"]},
    {"query": "professional networking message on linkedin",
     "expected_category": "marketing",
     "relevance_keywords": ["linkedin", "networking", "outreach", "message", "profile"]},
    {"query": "explain a complex concept to a beginner",
     "expected_category": None,
     "relevance_keywords": ["explain", "simple", "beginner", "tutorial", "teaching", "intro"]},
    {"query": "negotiation tactics for a salary discussion",
     "expected_category": None,
     "relevance_keywords": ["negotiate", "negotiation", "salary", "raise", "compensation", "interview"]},
    {"query": "translate text into formal business english",
     "expected_category": None,
     "relevance_keywords": ["translate", "translation", "business english", "formal", "rewrite"]},
]


def _flatten_text_for_relevance(row):
    title = str(row.get("title", "") or "")
    sub = str(row.get("subcategory", "") or "")
    tags = row.get("tags", [])
    if isinstance(tags, list):
        tags_str = " ".join(str(t) for t in tags)
    else:
        tags_str = str(tags or "")
    return (title + " " + sub + " " + tags_str).lower()


def grade_relevance(query_spec, row) -> int:
    """Graded relevance 0/1/2 for a (query, prompt) pair."""
    cat_match = (
        query_spec.get("expected_category") is not None
        and row.get("category") == query_spec["expected_category"]
    )
    blob = _flatten_text_for_relevance(row)
    kws = query_spec.get("relevance_keywords", []) or []
    kw_match = any(kw.lower() in blob for kw in kws)
    if cat_match and kw_match:
        return 2
    if cat_match or kw_match:
        return 1
    return 0


def _dataset_relevance_grades(query_spec):
    """Compute relevance grades for every prompt in the dataset for this query."""
    rows = work[["category", "title", "subcategory", "tags"]].to_dict("records")
    return np.asarray([grade_relevance(query_spec, r) for r in rows], dtype=int)


def dcg_at_k(grades, k=10):
    grades = list(grades)[:k]
    return float(sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(grades)))


def ndcg_at_k(predicted_grades, ideal_grades_full, k=10):
    """predicted_grades: graded relevance of the top-k predicted results, in order.
       ideal_grades_full: graded relevance of *every* prompt in the dataset (any order).
    """
    dcg = dcg_at_k(predicted_grades, k)
    idcg = dcg_at_k(sorted(list(ideal_grades_full), reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


# Precompute the per-query "ideal" relevance vectors over the whole dataset so
# the A/B loop below can compute nDCG cheaply.
print("Precomputing gold relevance grades for", len(gold_eval_queries), "queries...")
GOLD_IDEAL = {q["query"]: _dataset_relevance_grades(q) for q in gold_eval_queries}
n_relevant = {q: int((g > 0).sum()) for q, g in GOLD_IDEAL.items()}
print("Relevant prompts found per query (graded > 0):")
for q, n in n_relevant.items():
    print(f"  {n:5d}   {q}")

# A/B benchmark: baseline vs agentic-full vs agentic-no-rerank

BENCH_N_REWRITES   = 4
BENCH_POOL         = 80
BENCH_RERANK_BASE  = 40
BENCH_RERANK_AGENT = 60
BENCH_LAMBDA_BASE  = 0.82
BENCH_LAMBDA_AGENT = 0.65
TOPK = 10


def _score_topk(out_df, spec, k=TOPK):
    if len(out_df) == 0:
        return 0.0, 0.0, 0.0
    topk_records = out_df.head(k)[["category", "title", "subcategory", "tags"]].to_dict("records")
    grades = [grade_relevance(spec, r) for r in topk_records]
    if not grades:
        return 0.0, 0.0, 0.0
    p_at_k = float(np.mean([1 if g > 0 else 0 for g in grades]))
    rr = 0.0
    for j, g in enumerate(grades, start=1):
        if g > 0:
            rr = 1.0 / j
            break
    ndcg = ndcg_at_k(grades, GOLD_IDEAL[spec["query"]], k=k)
    return p_at_k, rr, ndcg


def _finalize(scored_pre, intent, lambda_rel, top_k=TOPK):
    scored = add_metadata_score(scored_pre, **intent_adaptive_weights(intent))
    scored["query_intent"] = intent
    if len(scored) > top_k:
        return mmr_diversify(scored, lambda_rel=lambda_rel, top_k=top_k)
    return scored.head(top_k).reset_index(drop=True)


def evaluate_all_pipelines_fast(gold_set):
    rows = []
    n = len(gold_set)
    for i, spec in enumerate(gold_set, 1):
        q = spec["query"]
        intent = detect_query_intent(q)
        print(f"  [{i:2}/{n}] {q[:70]}")

        # Baseline
        t0 = time.perf_counter()
        b_stage1 = retrieve_hybrid(q, top_k=60)
        b_stage2 = rerank(q, b_stage1, top_k=BENCH_RERANK_BASE)
        b_out = _finalize(b_stage2, intent, lambda_rel=BENCH_LAMBDA_BASE)
        baseline_lat = (time.perf_counter() - t0) * 1000

        # Shared agentic stages (rewrites + per-rewrite hybrid + RRF) run once
        # and reused for both agentic variants to avoid double LLM cost.
        t0 = time.perf_counter()
        rewrites = rewriter.rewrite(q, n=BENCH_N_REWRITES)
        rank_lists = [retrieve_hybrid(rq, top_k=BENCH_POOL)[["id"]] for rq in rewrites]
        fused = reciprocal_rank_fusion(rank_lists, k_const=60, top_n=BENCH_POOL * 2)
        fused_full = fused.merge(work, on="id", how="left")
        shared_ms = (time.perf_counter() - t0) * 1000

        # Agentic full: rerank -> metadata -> MMR
        t0 = time.perf_counter()
        rer = rerank(q, fused_full, top_k=BENCH_RERANK_AGENT)
        a_out = _finalize(rer, intent, lambda_rel=BENCH_LAMBDA_AGENT)
        agentic_full_lat = shared_ms + (time.perf_counter() - t0) * 1000

        # Agentic ablation: no cross-encoder; use RRF rank as semantic signal
        t0 = time.perf_counter()
        nor = fused_full.head(BENCH_RERANK_AGENT).copy()
        nor["rerank_score"] = nor["rrf_score"]
        nor_out = _finalize(nor, intent, lambda_rel=BENCH_LAMBDA_AGENT)
        agentic_norerank_lat = shared_ms + (time.perf_counter() - t0) * 1000

        for name, out_df, lat in [
            ("Baseline (best non-agentic)", b_out, baseline_lat),
            ("Agentic (rewrite + RRF + rerank + MMR)", a_out, agentic_full_lat),
            ("Agentic ablation (no cross-encoder)", nor_out, agentic_norerank_lat),
        ]:
            p, r, nd = _score_topk(out_df, spec)
            rows.append({"pipeline": name, "query": q,
                         "P@10": p, "RR": r, "nDCG@10": nd,
                         "latency_ms": lat})

    return pd.DataFrame(rows)


print(f"Running A/B benchmark on {len(gold_eval_queries)} gold queries...")
print(f"Active rewriter backend: {rewriter.backend}")
print(f"Knobs: n_rewrites={BENCH_N_REWRITES}, candidate_pool={BENCH_POOL}, "
      f"rerank_top_k(baseline/agentic)={BENCH_RERANK_BASE}/{BENCH_RERANK_AGENT}")
print(f"Shared LLM+RRF stages computed ONCE per query, then reused for full + ablation.\n")

t_total = time.perf_counter()
all_df = evaluate_all_pipelines_fast(gold_eval_queries)
print(f"\nTotal benchmark time: {(time.perf_counter() - t_total):.1f}s")

summary = (
    all_df.groupby("pipeline", as_index=False)
    .agg(**{
        "Mean P@10":   ("P@10", "mean"),
        "MRR":         ("RR", "mean"),
        "Mean nDCG@10":("nDCG@10", "mean"),
        "Avg latency (ms)": ("latency_ms", "mean"),
    })
)
print("\n=== Summary across", len(gold_eval_queries), "gold queries ===")
display(summary.round(3))

quality_long = summary.melt(
    id_vars=["pipeline"],
    value_vars=["Mean P@10", "MRR", "Mean nDCG@10"],
    var_name="metric", value_name="value",
)
fig_q = px.bar(
    quality_long, x="metric", y="value", color="pipeline",
    barmode="group", text=quality_long["value"].round(3).astype(str),
    title="Agentic vs Baseline vs Ablation — retrieval quality (gold harness)",
)
fig_q.update_yaxes(range=[0, 1])
fig_q.update_traces(textposition="outside")
fig_q.show()

fig_l = px.bar(
    summary, x="pipeline", y="Avg latency (ms)", color="pipeline",
    text=summary["Avg latency (ms)"].round(1).astype(str),
    title="Average latency per query (ms, lower is better)",
)
fig_l.update_traces(textposition="outside")
fig_l.update_layout(showlegend=False)
fig_l.show()

per_q = (
    all_df[["pipeline", "query", "nDCG@10"]]
    .pivot(index="query", columns="pipeline", values="nDCG@10")
    .reset_index()
)
agentic_col = "Agentic (rewrite + RRF + rerank + MMR)"
baseline_col = "Baseline (best non-agentic)"
per_q["delta_nDCG"] = per_q[agentic_col] - per_q[baseline_col]
per_q = per_q.sort_values("delta_nDCG", ascending=False)
display(per_q.round(3))

fig_delta = px.bar(
    per_q, x="query", y="delta_nDCG",
    color=per_q["delta_nDCG"].apply(
        lambda v: "agentic better" if v > 1e-6 else ("tie" if abs(v) <= 1e-6 else "baseline better")
    ),
    title="Per-query nDCG@10 gain from going agentic (positive = agentic better)",
)
fig_delta.update_layout(xaxis_tickangle=-25, yaxis_title="nDCG@10 (agentic) - nDCG@10 (baseline)")
fig_delta.show()

# Strict gold harness (corrected for evaluation leakage)
# This corrects the first benchmark which was too permissive: single-keyword
# substring matches and category-alone were enough to mark a prompt as relevant,
# inflating MRR to 1.0 for all pipelines.

STRICT_GOLD_QUERIES = [
    {
        "query": "draft a contract clause for a software license",
        "expected_category": None,
        "terms": ["contract", "clause", "agreement", "license", "licensing", "legal"],
        "strong_phrases": ["contract clause", "software license", "license agreement"],
    },
    {
        "query": "ai assistant for terms and conditions review",
        "expected_category": None,
        "terms": ["terms", "conditions", "legal", "policy", "compliance", "review"],
        "strong_phrases": ["terms and conditions", "legal review", "policy review"],
    },
    {
        "query": "improve seo of my landing page",
        "expected_category": "marketing",
        "terms": ["seo", "landing", "page", "keyword", "ranking", "search"],
        "strong_phrases": ["landing page", "search engine", "seo"],
    },
    {
        "query": "fix a bug in my javascript code",
        "expected_category": "coding",
        "terms": ["javascript", "js", "bug", "debug", "error", "code"],
        "strong_phrases": ["javascript code", "debug javascript", "fix bug"],
    },
    {
        "query": "summarize a research paper for a non-expert",
        "expected_category": None,
        "terms": ["summarize", "summary", "research", "paper", "non-expert", "layperson"],
        "strong_phrases": ["research paper", "non expert", "non-expert", "layperson summary"],
    },
    {
        "query": "write a marketing email to existing customers",
        "expected_category": "marketing",
        "terms": ["marketing", "email", "customer", "customers", "campaign", "newsletter"],
        "strong_phrases": ["marketing email", "email campaign", "existing customers"],
    },
    {
        "query": "creative bedtime story for a 5 year old",
        "expected_category": "creative-writing",
        "terms": ["bedtime", "children", "kids", "child", "story", "fairy"],
        "strong_phrases": ["bedtime story", "children story", "kids story"],
    },
    {
        "query": "monthly sales dashboard analysis",
        "expected_category": "data-analysis",
        "terms": ["monthly", "sales", "dashboard", "analysis", "kpi", "report"],
        "strong_phrases": ["sales dashboard", "monthly sales", "kpi dashboard"],
    },
    {
        "query": "professional networking message on linkedin",
        "expected_category": "marketing",
        "terms": ["linkedin", "networking", "outreach", "message", "professional", "connection"],
        "strong_phrases": ["linkedin message", "networking message", "connection request"],
    },
    {
        "query": "explain a complex concept to a beginner",
        "expected_category": None,
        "terms": ["complex", "concept", "beginner", "explain", "simple", "analogy"],
        "strong_phrases": ["complex concept", "explain simply", "beginner explanation"],
    },
    {
        "query": "negotiation tactics for a salary discussion",
        "expected_category": None,
        "terms": ["negotiation", "negotiate", "salary", "raise", "compensation", "tactics"],
        "strong_phrases": ["salary negotiation", "negotiate salary", "compensation discussion"],
    },
    {
        "query": "translate text into formal business english",
        "expected_category": None,
        "terms": ["translate", "translation", "formal", "business", "english", "rewrite"],
        "strong_phrases": ["business english", "formal english", "translate text"],
    },
]


def _strict_blob(row):
    vals = [
        str(row.get("title", "") or ""),
        str(row.get("subcategory", "") or ""),
        str(row.get("category", "") or ""),
        str(row.get("content", "") or "")[:600],
    ]
    tags = row.get("tags", [])
    vals.append(" ".join(map(str, tags)) if isinstance(tags, list) else str(tags or ""))
    return " ".join(vals).lower().replace("-", " ")


def _contains_term(blob, term):
    term = term.lower().replace("-", " ").strip()
    return re.search(r"\b" + re.escape(term) + r"\b", blob) is not None


def strict_grade_relevance(query_spec, row):
    """Stricter 0/1/2 relevance grade.

    Grade 2:
      - category matches AND at least 2 exact term hits; OR
      - at least one strong phrase AND at least 2 exact term hits.

    Grade 1:
      - at least 2 exact term hits; OR
      - expected category matches AND at least 1 exact term hit.

    Grade 0:
      - everything else.

    Importantly: category alone and one generic keyword alone are not enough.
    """
    blob = _strict_blob(row)
    terms = query_spec.get("terms", []) or []
    phrases = query_spec.get("strong_phrases", []) or []
    term_hits = sum(1 for t in terms if _contains_term(blob, t))
    phrase_hit = any(_contains_term(blob, p) for p in phrases)
    expected_cat = query_spec.get("expected_category")
    category_match = expected_cat is not None and row.get("category") == expected_cat

    if (category_match and term_hits >= 2) or (phrase_hit and term_hits >= 2):
        return 2
    if term_hits >= 2 or (category_match and term_hits >= 1):
        return 1
    return 0


def _strict_dataset_grades(query_spec):
    rows = work[["category", "title", "subcategory", "tags", "content"]].to_dict("records")
    return np.asarray([strict_grade_relevance(query_spec, r) for r in rows], dtype=int)


def _strict_score_topk(out_df, spec, k=TOPK):
    if len(out_df) == 0:
        return 0.0, 0.0, 0.0
    topk_records = out_df.head(k)[["category", "title", "subcategory", "tags", "content"]].to_dict("records")
    grades = [strict_grade_relevance(spec, r) for r in topk_records]
    if not grades:
        return 0.0, 0.0, 0.0
    p_at_k = float(np.mean([1 if g > 0 else 0 for g in grades]))
    rr = 0.0
    for j, g in enumerate(grades, start=1):
        if g > 0:
            rr = 1.0 / j
            break
    ndcg = ndcg_at_k(grades, STRICT_IDEAL[spec["query"]], k=k)
    return p_at_k, rr, ndcg


print("Building strict relevance diagnostics...")
STRICT_IDEAL = {q["query"]: _strict_dataset_grades(q) for q in STRICT_GOLD_QUERIES}
strict_diag_rows = []
for q in STRICT_GOLD_QUERIES:
    grades = STRICT_IDEAL[q["query"]]
    n_rel = int((grades > 0).sum())
    rel_pct = 100 * n_rel / len(grades)
    strict_diag_rows.append({
        "query": q["query"],
        "relevant_prompts": n_rel,
        "relevant_%": rel_pct,
        "grade2_prompts": int((grades == 2).sum()),
        "status": "too broad" if rel_pct > 5 else "ok",
    })

strict_diag = pd.DataFrame(strict_diag_rows)
display(strict_diag)


def evaluate_all_pipelines_strict(gold_set):
    rows = []
    n = len(gold_set)
    for i, spec in enumerate(gold_set, 1):
        q = spec["query"]
        intent = detect_query_intent(q)
        print(f"  [{i:2}/{n}] {q[:70]}")

        t0 = time.perf_counter()
        b_stage1 = retrieve_hybrid(q, top_k=60)
        b_stage2 = rerank(q, b_stage1, top_k=BENCH_RERANK_BASE)
        b_out = _finalize(b_stage2, intent, lambda_rel=BENCH_LAMBDA_BASE)
        baseline_lat = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        rewrites = rewriter.rewrite(q, n=BENCH_N_REWRITES)
        rank_lists = [retrieve_hybrid(rq, top_k=BENCH_POOL)[["id"]] for rq in rewrites]
        fused = reciprocal_rank_fusion(rank_lists, k_const=60, top_n=BENCH_POOL * 2)
        fused_full = fused.merge(work, on="id", how="left")
        shared_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        rer = rerank(q, fused_full, top_k=BENCH_RERANK_AGENT)
        a_out = _finalize(rer, intent, lambda_rel=BENCH_LAMBDA_AGENT)
        agentic_full_lat = shared_ms + (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        nor = fused_full.head(BENCH_RERANK_AGENT).copy()
        nor["rerank_score"] = nor["rrf_score"]
        nor_out = _finalize(nor, intent, lambda_rel=BENCH_LAMBDA_AGENT)
        agentic_norerank_lat = shared_ms + (time.perf_counter() - t0) * 1000

        for name, out_df, lat in [
            ("Baseline (strict)", b_out, baseline_lat),
            ("Agentic full (strict)", a_out, agentic_full_lat),
            ("Agentic no-rerank (strict)", nor_out, agentic_norerank_lat),
        ]:
            p, r, nd = _strict_score_topk(out_df, spec)
            rows.append({
                "pipeline": name,
                "query": q,
                "P@10": p,
                "RR": r,
                "nDCG@10": nd,
                "latency_ms": lat,
            })

    return pd.DataFrame(rows)


print(f"Running corrected strict A/B benchmark on {len(STRICT_GOLD_QUERIES)} queries...")
print(f"Active rewriter backend: {rewriter.backend}")
t_total = time.perf_counter()
strict_all_df = evaluate_all_pipelines_strict(STRICT_GOLD_QUERIES)
print(f"\nTotal strict benchmark time: {(time.perf_counter() - t_total):.1f}s")

strict_summary = (
    strict_all_df.groupby("pipeline", as_index=False)
    .agg(**{
        "Mean P@10": ("P@10", "mean"),
        "MRR": ("RR", "mean"),
        "Mean nDCG@10": ("nDCG@10", "mean"),
        "Avg latency (ms)": ("latency_ms", "mean"),
    })
)
print("\n=== Strict summary ===")
display(strict_summary.round(3))

strict_quality_long = strict_summary.melt(
    id_vars=["pipeline"],
    value_vars=["Mean P@10", "MRR", "Mean nDCG@10"],
    var_name="metric",
    value_name="value",
)
fig_strict_q = px.bar(
    strict_quality_long,
    x="metric",
    y="value",
    color="pipeline",
    barmode="group",
    text=strict_quality_long["value"].round(3).astype(str),
    title="Strict benchmark: retrieval quality after fixing proxy relevance",
)
fig_strict_q.update_yaxes(range=[0, 1])
fig_strict_q.update_traces(textposition="outside")
fig_strict_q.show()

strict_per_q = (
    strict_all_df[["pipeline", "query", "nDCG@10"]]
    .pivot(index="query", columns="pipeline", values="nDCG@10")
    .reset_index()
)
strict_per_q["delta_nDCG_full_vs_baseline"] = (
    strict_per_q["Agentic full (strict)"] - strict_per_q["Baseline (strict)"]
)
strict_per_q = strict_per_q.sort_values("delta_nDCG_full_vs_baseline", ascending=False)
display(strict_per_q.round(3))

fig_strict_delta = px.bar(
    strict_per_q,
    x="query",
    y="delta_nDCG_full_vs_baseline",
    color=strict_per_q["delta_nDCG_full_vs_baseline"].apply(
        lambda v: "agentic better" if v > 1e-6 else ("tie" if abs(v) <= 1e-6 else "baseline better")
    ),
    title="Strict per-query nDCG@10 delta: agentic full minus baseline",
)
fig_strict_delta.update_layout(
    xaxis_tickangle=-25,
    yaxis_title="nDCG@10 (agentic full) - nDCG@10 (baseline)",
)
fig_strict_delta.show()
