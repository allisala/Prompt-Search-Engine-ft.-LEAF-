# pipeline.py
# Top-level orchestration: wires retrieval -> reranking -> metadata scoring
# -> MMR diversification into a single search_pipeline() function.
# Supports three retrieval modes (dense, bm25, hybrid) and two flags:
#  - adaptive: enables intent-based metadata weight selection
#  - diversify: enables MMR post-processing
# Also provides the demo_search() wrapper and the interactive ipywidgets UI
# used in Chapter 13 of the notebook.
# Entry point for search demos; Streamlit UI is app.py in this folder.
# evaluation.py (benchmarks) depends on the same stack.
# Depends on: retrieval.py, reranking.py

# Main search pipeline

def search_pipeline(query, top_k=10, mode="hybrid", adaptive=False, diversify=False):
    t0 = time.perf_counter()

    if mode == "dense":
        stage1 = retrieve_dense(query, top_k=max(60, top_k * 6))
    elif mode == "bm25":
        stage1 = retrieve_bm25(query, top_k=max(60, top_k * 6))
        stage1 = stage1.merge(work[["id", "semantic_text"]], on="id", how="left")
    else:
        stage1 = retrieve_hybrid(query, top_k=max(60, top_k * 6))

    stage2 = rerank(query, stage1, top_k=max(40, top_k * 4))

    intent = detect_query_intent(query)
    if adaptive:
        stage3 = add_metadata_score(stage2, **intent_adaptive_weights(intent))
    else:
        stage3 = add_metadata_score(stage2)

    stage3["query_intent"] = intent

    if diversify:
        out = mmr_diversify(stage3, lambda_rel=0.82, top_k=top_k)
    else:
        out = stage3.head(top_k).reset_index(drop=True)

    latency_ms = (time.perf_counter() - t0) * 1000
    return out, latency_ms

# End-to-end demo run

query = "write a professional cold email to a potential B2B client"
results, latency_ms = search_pipeline(query, top_k=10, mode="hybrid")

print(f"Query: {query}")
print(f"Latency: {latency_ms:.2f} ms")

show_cols = [
    "id", "title", "category", "language", "final_score", "rerank_score",
    "popularity", "quality", "freshness", "likes", "upvotes", "views"
]

results[show_cols]

# Result visualisation

viz = results.copy()
fig_table = go.Figure(data=[go.Table(
    header=dict(values=["Rank", "Title", "Category", "Final", "Rerank", "Likes", "Upvotes"],
                fill_color="#1f2937", font=dict(color="white", size=12), align="left"),
    cells=dict(values=[
        list(range(1, len(viz) + 1)),
        viz["title"],
        viz["category"],
        viz["final_score"].round(4),
        viz["rerank_score"].round(4),
        viz["likes"],
        viz["upvotes"],
    ],
    fill_color="#f9fafb",
    align="left")
)])
fig_table.update_layout(title="Top Results Table")
fig_table.show()

melt_scores = viz[["id", "semantic_n", "popularity", "quality", "freshness", "final_score"]].copy()
plot_scores = melt_scores.melt(id_vars=["id"], var_name="component", value_name="value")
fig_comp = px.bar(plot_scores, x="id", y="value", color="component", barmode="group", title="Score Components by Result ID")
fig_comp.show()

# Interactive demo function

def demo_search(query: str, mode: str = "agentic_no_rerank", top_k: int = 5, show_rewrites: bool = True):
    """Run a query through one of the three pipelines and pretty-print the top results.

    Parameters
    ----------
    query : str
        Free-text natural-language query.
    mode : {"baseline", "agentic_full", "agentic_no_rerank"}
        Which pipeline to use. Defaults to the strict-benchmark winner.
    top_k : int
        Number of results to display.
    show_rewrites : bool
        For agentic modes, print the LLM-generated paraphrases used for RRF fusion.
    """
    rewrites = None
    if mode == "baseline":
        results, latency = search_pipeline(query, top_k=top_k, mode="hybrid",
                                           adaptive=True, diversify=True)
    elif mode == "agentic_full":
        results, latency, rewrites = agentic_search(query, top_k=top_k, use_reranker=True)
    elif mode == "agentic_no_rerank":
        results, latency, rewrites = agentic_search(query, top_k=top_k, use_reranker=False)
    else:
        raise ValueError(f"unknown mode: {mode!r}. Use baseline, agentic_full, or agentic_no_rerank.")

    bar = "=" * 78
    print(bar)
    print(f"Query     : {query}")
    print(f"Pipeline  : {mode}")
    print(f"Top-K     : {top_k}")
    print(f"Latency   : {latency:,.0f} ms")
    print(bar)

    if show_rewrites and rewrites:
        print("\nLLM rewrites fed to RRF:")
        for i, r in enumerate(rewrites):
            tag = "  (original query)" if i == 0 else ""
            print(f"  {i+1}. {r}{tag}")

    print("\nTop results:")
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        title = str(row.get("title", "<untitled>"))
        cat = str(row.get("category", "?"))
        score = float(row.get("final_score", float("nan")))
        content_full = str(row.get("content", "") or "")
        snippet = content_full[:280].replace("\n", " ").strip()
        ellipsis = "..." if len(content_full) > 280 else ""
        print(f"\n  #{rank}  [{cat}]  {title}")
        print(f"        score={score:.3f}")
        print(f"        {snippet}{ellipsis}")
    print()
    return results


_demo_results = demo_search(
    "write a cold email to a potential B2B client",
    mode="agentic_no_rerank",
    top_k=5,
)

# ipywidgets interactive search UI (Chapter 13)

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    _HAS_WIDGETS = True
except Exception:
    _HAS_WIDGETS = False

if _HAS_WIDGETS:
    _query_box = widgets.Text(
        value="write a cold email to a potential B2B client",
        placeholder="Type a natural-language query...",
        description="Query:",
        layout=widgets.Layout(width="90%"),
    )
    _mode_dropdown = widgets.Dropdown(
        options=[
            ("Baseline  (fast, ~4 s)", "baseline"),
            ("Agentic full  (slow, ~22 s)", "agentic_full"),
            ("Agentic no-rerank  (recommended, ~14 s)", "agentic_no_rerank"),
        ],
        value="agentic_no_rerank",
        description="Pipeline:",
        layout=widgets.Layout(width="50%"),
    )
    _topk_slider = widgets.IntSlider(value=5, min=3, max=15, description="Top K:")
    _go_button = widgets.Button(description="Search", button_style="primary", icon="search")
    _out = widgets.Output()

    def _on_click(_b):
        with _out:
            clear_output()
            q = _query_box.value.strip()
            if not q:
                print("Please type a query.")
                return
            try:
                demo_search(q, mode=_mode_dropdown.value, top_k=_topk_slider.value)
            except Exception as e:
                print(f"Error: {e}")

    _go_button.on_click(_on_click)

    display(widgets.VBox([
        widgets.HTML("<h3 style='margin-bottom:0'>PromptKaban interactive search</h3>"
                     "<div style='color:#666;margin-bottom:10px'>Type a query, pick a pipeline, hit Search.</div>"),
        _query_box,
        widgets.HBox([_mode_dropdown, _topk_slider]),
        _go_button,
        _out,
    ]))
else:
    print("ipywidgets not installed — interactive UI unavailable in this environment.")
    print("Install it with:  pip install ipywidgets")
    print("In the meantime you can still call demo_search('your query here') from any cell.")
