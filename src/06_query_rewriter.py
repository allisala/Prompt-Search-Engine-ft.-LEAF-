# query_rewriter.py
# Implements the agentic retrieval layer:
#  1. QueryRewriter: generates N paraphrase intents for a user query using
#     a cascading backend (Ollama local LLM -> HF flan-t5-base -> heuristic).
#     Uses a multi-pass strategy (one rewrite per call, different prompt
#     templates) to avoid seq2seq concatenation artifacts common in small LLMs.
#  2. reciprocal_rank_fusion: fuses multiple ranked lists by position
#     (k_const=60, from Cormack-Clarke-Buettcher), independent of score scale.
#  3. agentic_search: orchestrates rewrite -> per-rewrite hybrid retrieval
#     -> RRF -> optional cross-encoder rerank -> metadata scoring -> MMR.
# Depends on: retrieval.py, reranking.py, pipeline.py

from typing import List

# Multi-pass strategy: small open-source LLMs (especially flan-t5-base on CPU)
# struggle to follow "give me 5 rewrites separated by newlines" — they tend to
# glue everything onto one line. Instead we ask for ONE rewrite at a time, with
# a different instruction each pass. This guarantees N distinct outputs and is
# also easier for an LLM-as-rewriter to follow.
PROMPT_VARIANTS = [
    "Paraphrase this search query using different words but the same meaning. Reply with only the paraphrase.\nQuery: {query}\nParaphrase:",
    "Rewrite this search query using synonyms. Reply with only the rewritten query.\nQuery: {query}\nRewrite:",
    "Make this search query more specific and detailed. Reply with only the new query.\nQuery: {query}\nDetailed query:",
    "Reformulate this search query as if asking an AI assistant. Reply with only the reformulation.\nQuery: {query}\nReformulation:",
    "Restate this search query focusing on the underlying intent. Reply with only the new query.\nQuery: {query}\nIntent:",
    "Expand this search query with related terminology. Reply with only the expanded query.\nQuery: {query}\nExpanded:",
    "Rewrite this search query in a more formal style. Reply with only the rewritten query.\nQuery: {query}\nFormal:",
    "Generate an alternative phrasing of this search query that uses different vocabulary. Reply with only the alternative.\nQuery: {query}\nAlternative:",
]


class QueryRewriter:
    """Generate paraphrase intents for a query using an open-source LLM.

    Backends are tried in this order:
      1. local Ollama server (matches the Week 9 RAG lab setup)
      2. Hugging Face Transformers, google/flan-t5-base (CPU-friendly seq2seq)
      3. heuristic rule-based expansion (always available)

    The rewriter uses a multi-pass strategy: it asks the LLM for ONE rewrite
    per call, repeating with a different instruction each time. This is
    significantly more reliable on small open-source models than asking for
    "N rewrites in one shot", which tends to produce a single glued blob.
    """

    def __init__(self, prefer="auto", hf_model="google/flan-t5-base", ollama_model="llama3"):
        self.prefer = prefer
        self.hf_model_name = hf_model
        self.ollama_model = ollama_model
        self.backend = None
        self._tokenizer = None
        self._llm = None
        self._init_backend()

    def _try_ollama(self):
        try:
            import requests
            r = requests.get("http://localhost:11434/api/tags", timeout=1.5)
            if r.status_code == 200:
                self.backend = "ollama"
                return True
        except Exception:
            pass
        return False

    def _try_hf(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            print(f"Loading rewriter model: {self.hf_model_name} (first time only, ~250MB)")
            self._tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
            self._llm = AutoModelForSeq2SeqLM.from_pretrained(self.hf_model_name)
            self.backend = "hf"
            return True
        except Exception as exc:
            print("Hugging Face backend unavailable:", type(exc).__name__, str(exc)[:140])
            return False

    def _init_backend(self):
        if self.prefer in ("auto", "ollama") and self._try_ollama():
            return
        if self.prefer in ("auto", "hf") and self._try_hf():
            return
        self.backend = "heuristic"

    def _ollama_generate(self, prompt: str) -> str:
        import requests
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 64},
            },
            timeout=60,
        )
        return r.json().get("response", "")

    def _hf_generate(self, prompt: str) -> str:
        ids = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=256
        ).input_ids
        out = self._llm.generate(
            ids,
            max_new_tokens=48,
            do_sample=True,
            top_p=0.92,
            temperature=0.9,
            no_repeat_ngram_size=3,
        )
        return self._tokenizer.decode(out[0], skip_special_tokens=True)

    def _generate_one(self, prompt: str) -> str:
        if self.backend == "ollama":
            return self._ollama_generate(prompt)
        return self._hf_generate(prompt)

    def _heuristic(self, query: str, n: int) -> List[str]:
        intent = detect_query_intent(query)
        seeds = {
            "coding":           ["programming assistant prompt", "developer-focused prompt", "code generation request", "prompt to help debug code"],
            "marketing":        ["marketing copy prompt", "ad copywriting prompt", "brand voice content prompt", "social media campaign prompt"],
            "data-analysis":    ["analytics assistant prompt", "data interpretation prompt", "kpi reporting prompt", "trend analysis prompt"],
            "creative-writing": ["story generation prompt", "creative writing assistant", "narrative scene prompt", "imaginative prose prompt"],
            "general":          ["assistant prompt for this task", "AI helper prompt for", "prompt to assist with"],
        }[intent]
        return [f"{s}: {query}" for s in seeds][: max(0, n - 1)]

    @staticmethod
    def _clean(line: str) -> str:
        line = line.strip()
        line = line.split("\n", 1)[0].strip()
        line = re.sub(r"^[\-\*\d\.\)\(\s]+", "", line).strip().strip('"').strip("'")
        return line

    def rewrite(self, query: str, n: int = 5) -> List[str]:
        """Return original query + up to (n-1) reformulations.

        Multi-pass: one LLM call per rewrite, with a different prompt template
        each time, so we get N genuinely distinct paraphrases.
        """
        if self.backend == "heuristic":
            return [query] + self._heuristic(query, n)

        cands: List[str] = []
        seen = {query.lower().strip()}

        variants = []
        i = 0
        while len(variants) < (n - 1):
            variants.append(PROMPT_VARIANTS[i % len(PROMPT_VARIANTS)])
            i += 1

        for v in variants:
            try:
                text = self._generate_one(v.format(query=query))
            except Exception as exc:
                print("LLM generation failed mid-loop:", type(exc).__name__, str(exc)[:120])
                break

            cleaned = self._clean(text)
            if not cleaned or len(cleaned) < 4:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            cands.append(cleaned)
            if len(cands) >= n - 1:
                break

        if not cands:
            return [query] + self._heuristic(query, n)

        if len(cands) < n - 1:
            for h in self._heuristic(query, n - len(cands)):
                key = h.lower()
                if key not in seen:
                    seen.add(key)
                    cands.append(h)
                if len(cands) >= n - 1:
                    break

        return [query] + cands


rewriter = QueryRewriter(prefer="auto")
print("Active rewriter backend:", rewriter.backend)

example_query = "legal contract drafting prompt"
example_rewrites = rewriter.rewrite(example_query, n=5)
print("\nExample rewrites for:", example_query)
for i, r in enumerate(example_rewrites):
    tag = "  (original)" if i == 0 else ""
    print(f"  {i+1}. {r}{tag}")

# Reciprocal Rank Fusion and agentic search

def reciprocal_rank_fusion(rank_lists, k_const: int = 60, top_n: int = 200) -> pd.DataFrame:
    """Fuse multiple ranked lists with the Reciprocal Rank Fusion algorithm.

    For each ranked list and each item at rank r (1-indexed), we add
    1 / (k_const + r) to that item's fused score. RRF is robust because it
    does not need score normalization across heterogeneous retrievers/queries:
    only the *position* matters. k_const = 60 is the standard from the
    original Cormack-Clarke-Buettcher paper.
    """
    scores = {}
    for rl in rank_lists:
        if rl is None or len(rl) == 0:
            continue
        for rank, _id in enumerate(rl["id"].tolist()):
            scores[_id] = scores.get(_id, 0.0) + 1.0 / (k_const + rank + 1)
    if not scores:
        return pd.DataFrame(columns=["id", "rrf_score"])
    rows = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
    return pd.DataFrame(rows, columns=["id", "rrf_score"])


def agentic_search(query: str,
                   top_k: int = 10,
                   n_rewrites: int = 6,
                   candidate_pool: int = 120,
                   lambda_rel: float = 0.65,
                   adaptive: bool = True,
                   diversify: bool = True,
                   use_reranker: bool = True,
                   verbose: bool = False):
    """End-to-end agentic retrieval pipeline.

    Pipeline:
      1. Rewrite the query into n_rewrites paraphrase intents (via QueryRewriter).
      2. Run hybrid retrieval (dense + BM25) once per rewrite.
      3. Fuse ranked candidate lists with Reciprocal Rank Fusion.
      4. (Optional) Cross-encoder rerank the fused candidates against the
         ORIGINAL query. Reranking is optional so we can run an ablation:
         "RRF without cross-encoder" isolates how much the cross-encoder
         absorbs the recall gains from RRF.
      5. Apply intent-adaptive metadata fusion (semantic + popularity + quality + freshness).
      6. Apply MMR diversification to the top-k.

    Notes on the tuned defaults:
      n_rewrites = 6     -> more paraphrase diversity for RRF to fuse over
      candidate_pool=120 -> wider per-rewrite recall pool
      lambda_rel = 0.65  -> the agentic candidate set is already more diverse,
                            so we can push MMR harder to break near-duplicates
      rerank ceiling = max(60, top_k * 6) -> let the cross-encoder see more of
                            what RRF surfaced; otherwise its 40-candidate
                            ceiling silently re-collapses recall gains.
    """
    t0 = time.perf_counter()

    rewrites = rewriter.rewrite(query, n=n_rewrites)
    if verbose:
        print("Rewrites used:")
        for i, r in enumerate(rewrites):
            print(f"  {i+1}. {r}{'  (original)' if i == 0 else ''}")

    rank_lists = []
    for q in rewrites:
        cand = retrieve_hybrid(q, top_k=candidate_pool)[["id"]]
        rank_lists.append(cand)
    fused = reciprocal_rank_fusion(rank_lists, k_const=60, top_n=candidate_pool * 2)

    if len(fused) == 0:
        latency_ms = (time.perf_counter() - t0) * 1000
        return work.iloc[0:0].copy(), latency_ms, rewrites

    fused_full = fused.merge(work, on="id", how="left")

    if use_reranker:
        rer = rerank(query, fused_full, top_k=max(60, top_k * 6))
    else:
        rer = fused_full.head(max(60, top_k * 6)).copy()
        rer["rerank_score"] = rer["rrf_score"]

    intent = detect_query_intent(query)
    if adaptive:
        scored = add_metadata_score(rer, **intent_adaptive_weights(intent))
    else:
        scored = add_metadata_score(rer)
    scored["query_intent"] = intent
    scored["n_rewrites"] = len(rewrites)

    if diversify and len(scored) > top_k:
        out = mmr_diversify(scored, lambda_rel=lambda_rel, top_k=top_k)
    else:
        out = scored.head(top_k).reset_index(drop=True)

    latency_ms = (time.perf_counter() - t0) * 1000
    return out, latency_ms, rewrites


print("Agentic pipeline ready.")
print(" - reciprocal_rank_fusion: RRF over multi-query candidates")
print(" - agentic_search: rewrite -> retrieve x N -> RRF -> [rerank] -> metadata -> MMR")
print("   defaults: n_rewrites=6, candidate_pool=120, rerank ceiling=max(60, top_k*6),")
print("             lambda_rel=0.65, use_reranker=True (set False for ablation)")

# Agentic demo run

demo_query = "write a professional cold email to a potential B2B client"
print("=" * 72)
print("AGENTIC SEARCH — worked example")
print("=" * 72)
print(f"Query: {demo_query}")
print(f"Rewriter backend: {rewriter.backend}\n")

ag_results, ag_latency, ag_rewrites = agentic_search(
    demo_query, top_k=10, n_rewrites=4, verbose=True
)
print(f"\nLatency (full agentic pipeline): {ag_latency:.1f} ms")
print(f"Detected intent: {ag_results['query_intent'].iloc[0] if len(ag_results) else 'n/a'}")

print("\nTop 10 results from the agentic pipeline:")
ag_show_cols = [c for c in [
    "id", "title", "category", "final_score", "rerank_score",
    "popularity", "quality", "freshness"
] if c in ag_results.columns]
ag_results[ag_show_cols].head(10)
