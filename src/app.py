"""PromptKaban — Streamlit demo (LEAF × LUISS).

Run from this folder (src/):
    cd src && streamlit run app.py

Uses search_runtime.py — same logic as main.ipynb (2026 pipeline).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from search_runtime import SearchRuntime

EXAMPLE_QUERIES = [
    "write a professional cold email to a potential B2B client",
    "fix a bug in my javascript code",
    "improve seo of my landing page",
    "summarize a research paper for a non-expert",
]

PIPELINE_INFO = {
    "baseline": {
        "label": "Baseline",
        "subtitle": "Hybrid + rerank + metadata + MMR",
        "latency": "~1.6 s",
        "badge": "Default route",
    },
    "agentic_no_rerank": {
        "label": "Agentic no-rerank",
        "subtitle": "Rewrite + RRF + metadata + MMR",
        "latency": "~6 s",
        "badge": "Best strict nDCG (0.58)",
    },
    "agentic_full": {
        "label": "Agentic full",
        "subtitle": "Rewrite + RRF + rerank + metadata",
        "latency": "~7.6 s",
        "badge": "Comparison only",
    },
}


@st.cache_resource(show_spinner="Loading dataset, embeddings, reranker, rewriter…")
def get_runtime() -> SearchRuntime:
    return SearchRuntime.load()


def _score_chart(row: pd.Series) -> None:
    parts = {
        "Semantic": float(row.get("semantic_n", 0)),
        "Popularity": float(row.get("popularity", 0)),
        "Quality": float(row.get("quality", 0)),
        "Freshness": float(row.get("freshness", 0)),
    }
    st.bar_chart(pd.Series(parts), height=120)


st.set_page_config(
    page_title="PromptKaban Search",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetricValue"] { font-size: 1.1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.title("PromptKaban")
    st.caption("LEAF × LUISS · semantic prompt search")

    pipeline = st.radio(
        "Pipeline",
        options=list(PIPELINE_INFO.keys()),
        format_func=lambda k: PIPELINE_INFO[k]["label"],
        index=1,
        help="Matches the three pipelines benchmarked in main.ipynb (strict harness).",
    )
    info = PIPELINE_INFO[pipeline]
    st.info(f"**{info['badge']}** · {info['subtitle']} · {info['latency']}")

    top_k = st.slider("Results (top-K)", 3, 15, 8)

    if pipeline != "baseline":
        st.markdown("**Agentic settings**")
        n_rewrites = st.slider("LLM rewrites", 2, 8, 6)
        pool = st.slider("Candidates per rewrite", 40, 160, 120, step=10)
    else:
        n_rewrites, pool = 6, 120

    show_rewrites = st.checkbox("Show query rewrites", value=True)
    show_scores = st.checkbox("Score breakdown charts", value=True)

    st.divider()
    st.markdown(
        "**Tip:** Search for a *task*, not a command.  \n"
        "e.g. `make my writing more professional`"
    )

try:
    runtime = get_runtime()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

# --- Header ---
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.header("Find the right prompt")
    st.caption(
        f"**{len(runtime.ctx['work']):,}** prompts indexed · "
        f"embeddings: `{runtime.ctx['embed_model']}` · "
        f"rewriter: `{runtime.rewriter.backend}`"
    )
with col_h2:
    st.metric("Indexed", f"{len(runtime.ctx['work']):,}")

# --- Query input ---
if "query_text" not in st.session_state:
    st.session_state.query_text = EXAMPLE_QUERIES[0]

qcol, bcol = st.columns([5, 1])
with qcol:
    query = st.text_input(
        "Your search",
        key="query_text",
        placeholder="e.g. debug a slow python api endpoint",
        label_visibility="collapsed",
    )
with bcol:
    run = st.button("Search", type="primary", use_container_width=True)

st.caption("Try an example:")
ex_cols = st.columns(len(EXAMPLE_QUERIES))
for i, (col, ex) in enumerate(zip(ex_cols, EXAMPLE_QUERIES)):
    with col:
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.query_text = ex
            st.session_state.run_search = True
            st.rerun()

if st.session_state.pop("run_search", False):
    run = True

# --- Search ---
if run:
    if not query.strip():
        st.warning("Type a search query first.")
        st.stop()

    agentic_kw = {}
    if pipeline != "baseline":
        agentic_kw = dict(n_rewrites=n_rewrites, candidate_pool=pool)

    with st.spinner(f"Running **{info['label']}**…"):
        results, latency_ms, meta = runtime.run(
            pipeline, query.strip(), top_k=top_k, **agentic_kw
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pipeline", info["label"])
    m2.metric("Latency", f"{latency_ms:,.0f} ms")
    m3.metric("Returned", len(results))
    m4.metric("Intent", meta.get("intent", "—"))

    hints = []
    if meta.get("target_model"):
        hints.append(f"target model: **{meta['target_model']}**")
    if meta.get("intent"):
        hints.append(f"intent: **{meta['intent']}**")
    if hints:
        st.markdown("Detected from query: " + " · ".join(hints))

    if show_rewrites and meta.get("rewrites"):
        with st.expander("Query rewrites (fed to RRF)", expanded=True):
            for i, r in enumerate(meta["rewrites"]):
                tag = " *(original)*" if i == 0 else ""
                st.markdown(f"{i + 1}. {r}{tag}")

    st.subheader(f"Top {len(results)} prompts")

    if len(results) == 0:
        st.info("No results.")
    else:
        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            title = str(row.get("title", "Untitled"))
            cat = str(row.get("category", "—"))
            score = float(row.get("final_score", 0))
            tm = str(row.get("target_model", "") or "—")
            header = f"#{rank} · {title} · **{score:.3f}** · [{cat}] · model: {tm}"

            with st.expander(header, expanded=(rank == 1)):
                left, right = st.columns([2, 1])
                with left:
                    tags = row.get("tags", [])
                    tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags or "—")
                    st.markdown(
                        f"**Subcategory:** {row.get('subcategory', '—')}  \n"
                        f"**Tags:** {tags_str}  \n"
                        f"**Language:** {row.get('language', '—')} · **Difficulty:** {row.get('difficulty', '—')}"
                    )
                    content = str(row.get("content", "") or "")
                    st.text_area("Prompt text", content, height=180, disabled=True, label_visibility="collapsed")
                with right:
                    st.markdown(
                        f"**Likes** {int(row.get('likes') or 0):,}  \n"
                        f"**Upvotes** {int(row.get('upvotes') or 0):,}  \n"
                        f"**Views** {int(row.get('views') or 0):,}"
                    )
                    if "constraint_mult" in row:
                        st.caption(f"Compatibility mult: {float(row['constraint_mult']):.2f}")
                    if show_scores:
                        _score_chart(row)
else:
    st.info("Enter a query and click **Search**. First run loads models (~30–60 s).")
