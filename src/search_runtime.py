# search_runtime.py
# Backend for the Streamlit demo and reproducible search outside the notebook.
# Mirrors main.ipynb: hybrid retrieval, cross-encoder rerank, query-aware metadata
# (target model / difficulty / language), MMR, and agentic rewrite + RRF.

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Repo root (one level up from src/); the dataset lives at the repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "LEAF-promptkaban-dataset" / "dataset.json"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RE_RANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMB_MAX_CHARS = 700

DEFAULT_METADATA_WEIGHTS = dict(w_sem=0.72, w_pop=0.14, w_quality=0.09, w_fresh=0.05)
METADATA_WEIGHTS = DEFAULT_METADATA_WEIGHTS.copy()

TARGET_MODEL_ALIASES = {
    "gpt-4o": ["gpt-4o", "gpt4o", "gpt 4o", "chatgpt-4o", "openai gpt-4o"],
    "gpt-4": ["gpt-4", "gpt4", "gpt 4", "chatgpt-4", "chatgpt"],
    "claude-3.5-sonnet": ["claude 3.5", "claude-3.5", "claude 3.5 sonnet", "claude sonnet"],
    "claude-4-sonnet": ["claude 4", "claude-4", "claude 4 sonnet", "claude opus"],
    "gemini-2.0-flash": ["gemini 2.0", "gemini-2.0", "gemini flash", "gemini 2.0 flash"],
    "gemini-2.5-pro": ["gemini 2.5", "gemini-2.5", "gemini pro", "gemini 2.5 pro"],
    "llama-3.3-70b": ["llama 3.3", "llama-3.3", "llama 70b", "llama 3"],
    "deepseek-r1": ["deepseek", "deepseek r1", "deepseek-r1"],
    "mistral-large": ["mistral large", "mistral-large", "mistral ai"],
}

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


def _safe_text(x) -> str:
    if isinstance(x, list):
        return " ".join(map(str, x))
    if pd.isna(x):
        return ""
    return str(x)


def _normalize(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return vals
    vmin, vmax = vals.min(), vals.max()
    if np.isclose(vmin, vmax):
        return np.ones_like(vals)
    return (vals - vmin) / (vmax - vmin)


def _emb_cache_path(model_name: str, max_chars: int) -> Path:
    safe = model_name.replace("/", "__").replace("-", "_")
    return PROJECT_ROOT / f"emb_cache_{safe}_{max_chars}.npy"


def detect_target_model(query: str) -> Optional[str]:
    if not query:
        return None
    q = query.lower().replace("_", " ")
    for canonical, aliases in TARGET_MODEL_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if alias in q:
                return canonical
    for slug in TARGET_MODEL_ALIASES:
        if slug.replace("-", " ") in q or slug in q.replace(" ", "-"):
            return slug
    if any(p in q for p in ["any model", "any llm", "model agnostic", "any provider"]):
        return "any"
    return None


def detect_difficulty_hint(query: str) -> Optional[str]:
    q = (query or "").lower()
    if any(k in q for k in ["beginner", "simple", "non-expert", "eli5", "5 year old", "kids", "novice"]):
        return "beginner"
    if any(k in q for k in ["advanced", "expert", "senior", "production-grade"]):
        return "advanced"
    return None


def detect_query_language(query: str) -> Optional[str]:
    q = (query or "").lower()
    if any(k in q for k in ["español", "spanish", " en espanol", " en español"]):
        return "es"
    if any(k in q for k in ["français", "french", " en français"]):
        return "fr"
    if any(k in q for k in ["deutsch", "german"]):
        return "de"
    return None


def target_model_compat_score(candidate_tm, requested_tm) -> float:
    if requested_tm is None:
        return 1.0
    c = str(candidate_tm or "any")
    if c == requested_tm:
        return 1.0
    if c == "any":
        return 0.90
    if requested_tm == "any":
        return 1.0
    return 0.50


def difficulty_compat_score(candidate_diff, hint) -> float:
    if hint is None:
        return 1.0
    c = str(candidate_diff or "")
    if c == hint:
        return 1.0
    if hint == "beginner" and c == "intermediate":
        return 0.85
    return 0.75


def language_compat_score(candidate_lang, hint) -> float:
    if hint is None:
        return 1.0
    c = str(candidate_lang or "en")
    return 1.0 if c == hint else 0.80


def detect_query_intent(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ["code", "python", "api", "debug", "bug", "sql", "javascript"]):
        return "coding"
    if any(k in q for k in ["email", "campaign", "ad", "brand", "linkedin", "seo", "marketing"]):
        return "marketing"
    if any(k in q for k in ["analysis", "dashboard", "kpi", "forecast", "trend", "data"]):
        return "data-analysis"
    if any(k in q for k in ["story", "poem", "creative", "script", "fiction", "bedtime"]):
        return "creative-writing"
    return "general"


def intent_adaptive_weights(intent: str) -> dict:
    if intent == "marketing":
        return dict(w_sem=0.64, w_pop=0.20, w_quality=0.10, w_fresh=0.06)
    if intent == "coding":
        return dict(w_sem=0.78, w_pop=0.10, w_quality=0.09, w_fresh=0.03)
    if intent == "data-analysis":
        return dict(w_sem=0.74, w_pop=0.12, w_quality=0.10, w_fresh=0.04)
    if intent == "creative-writing":
        return dict(w_sem=0.70, w_pop=0.14, w_quality=0.08, w_fresh=0.08)
    return dict(METADATA_WEIGHTS)


class QueryRewriter:
    def __init__(self, prefer: str = "auto", hf_model: str = "google/flan-t5-base", ollama_model: str = "llama3"):
        self.prefer = prefer
        self.hf_model_name = hf_model
        self.ollama_model = ollama_model
        self.backend: Optional[str] = None
        self._tokenizer = None
        self._llm = None
        self._init_backend()

    def _try_ollama(self) -> bool:
        try:
            import requests
            r = requests.get("http://localhost:11434/api/tags", timeout=1.5)
            if r.status_code == 200:
                self.backend = "ollama"
                return True
        except Exception:
            pass
        return False

    def _try_hf(self) -> bool:
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
            self._llm = AutoModelForSeq2SeqLM.from_pretrained(self.hf_model_name)
            self.backend = "hf"
            return True
        except Exception:
            return False

    def _init_backend(self) -> None:
        if self.prefer in ("auto", "ollama") and self._try_ollama():
            return
        if self.prefer in ("auto", "hf") and self._try_hf():
            return
        self.backend = "heuristic"

    def _clean(self, text: str) -> str:
        text = (text or "").strip().split("\n")[0].strip()
        text = re.sub(r"^(paraphrase|rewrite|query|intent|expanded|formal|alternative)\s*:\s*", "", text, flags=re.I)
        return text.strip(" \"'")

    def _ollama(self, prompt: str, max_tokens: int = 64) -> str:
        import requests
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": max_tokens},
        }
        r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
        r.raise_for_status()
        return self._clean(r.json().get("response", ""))

    def _hf(self, prompt: str, max_tokens: int = 64) -> str:
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        outputs = self._llm.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, top_p=0.92, temperature=0.9)
        return self._clean(self._tokenizer.decode(outputs[0], skip_special_tokens=True))

    def _heuristic(self, query: str, n: int) -> List[str]:
        q = query.lower()
        variants = [query]
        if "email" in q:
            variants += [f"professional email template for {query}", f"cold outreach message: {query}"]
        if any(k in q for k in ["code", "bug", "debug", "api"]):
            variants += [f"debugging help for {query}", f"software engineering prompt: {query}"]
        if "seo" in q or "marketing" in q:
            variants += [f"marketing copy for {query}", f"SEO content strategy: {query}"]
        if "story" in q or "creative" in q:
            variants += [f"creative writing prompt: {query}", f"narrative template for {query}"]
        seen = {v.lower() for v in variants}
        out = []
        for v in variants:
            if v.lower() not in seen:
                seen.add(v.lower())
                out.append(v)
            if len(out) >= n - 1:
                break
        return out[: max(0, n - 1)]

    def rewrite(self, query: str, n: int = 6) -> List[str]:
        cands = [query]
        seen = {query.lower()}
        for i in range(n - 1):
            prompt = PROMPT_VARIANTS[i % len(PROMPT_VARIANTS)].format(query=query)
            try:
                if self.backend == "ollama":
                    text = self._ollama(prompt)
                elif self.backend == "hf":
                    text = self._hf(prompt)
                else:
                    break
            except Exception:
                break
            text = self._clean(text)
            if text and text.lower() not in seen and len(text) > 3:
                seen.add(text.lower())
                cands.append(text)
        if len(cands) < n and self.backend == "heuristic":
            for h in self._heuristic(query, n):
                if h.lower() not in seen:
                    seen.add(h.lower())
                    cands.append(h)
                if len(cands) >= n:
                    break
        return cands[:n]


class SearchRuntime:
    """Loaded index + models; call search methods on an instance."""

    def __init__(self, ctx: dict, rewriter: QueryRewriter):
        self.ctx = ctx
        self.rewriter = rewriter

    @classmethod
    def load(cls, embed_model: str = EMBED_MODEL_NAME) -> "SearchRuntime":
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder, SentenceTransformer
        from sklearn.preprocessing import MinMaxScaler

        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

        df = pd.read_json(DATA_PATH)
        work = df.copy()
        work["semantic_text"] = (
            "Title: " + work["title"].map(_safe_text)
            + "\nCategory: " + work["category"].map(_safe_text)
            + "\nSubcategory: " + work["subcategory"].map(_safe_text)
            + "\nTags: " + work["tags"].map(_safe_text)
            + "\nPrompt: " + work["content"].map(_safe_text)
        )
        work["tokenized_for_bm25"] = work["semantic_text"].str.lower().str.findall(r"\b\w+\b")

        meta_cols = ["likes", "upvotes", "views", "uses", "author_reputation", "fork_count", "version"]
        meta_scaled = pd.DataFrame(
            MinMaxScaler().fit_transform(work[meta_cols].fillna(0)),
            columns=meta_cols,
        )
        work_meta = work.copy()
        for c in meta_cols:
            work_meta[f"{c}_n"] = meta_scaled[c]
        work_meta["created_at"] = pd.to_datetime(work_meta["created_at"], errors="coerce", utc=True)
        max_dt = work_meta["created_at"].max()
        work_meta["age_days"] = (max_dt - work_meta["created_at"]).dt.total_seconds() / 86400
        work_meta["age_days"] = work_meta["age_days"].fillna(work_meta["age_days"].median())
        work_meta["freshness"] = np.exp(-work_meta["age_days"] / 180.0)

        embedder = SentenceTransformer(embed_model)
        cache_path = _emb_cache_path(embed_model, EMB_MAX_CHARS)
        if cache_path.exists():
            emb = np.load(cache_path)
        else:
            corpus = [t[:EMB_MAX_CHARS] for t in work["semantic_text"].tolist()]
            emb = embedder.encode(corpus, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
            emb = np.asarray(emb, dtype=np.float32)
            np.save(cache_path, emb)

        use_faiss = True
        try:
            import faiss
            index = faiss.IndexFlatIP(emb.shape[1])
            index.add(emb)
        except Exception:
            from sklearn.neighbors import NearestNeighbors
            use_faiss = False
            index = NearestNeighbors(n_neighbors=200, metric="cosine")
            index.fit(emb)

        ctx = {
            "work": work,
            "work_meta": work_meta,
            "embedder": embedder,
            "embeddings": emb,
            "index": index,
            "use_faiss": use_faiss,
            "bm25": BM25Okapi(work["tokenized_for_bm25"].tolist()),
            "reranker": CrossEncoder(RE_RANKER_NAME),
            "embed_model": embed_model,
        }
        return cls(ctx, QueryRewriter(prefer="auto"))

    def _retrieve_dense(self, query: str, top_k: int) -> pd.DataFrame:
        qvec = np.asarray(self.ctx["embedder"].encode([query], normalize_embeddings=True), dtype=np.float32)
        if self.ctx["use_faiss"]:
            sims, idxs = self.ctx["index"].search(qvec, top_k)
            sims, idxs = sims[0], idxs[0]
        else:
            dists, idxs = self.ctx["index"].kneighbors(qvec, n_neighbors=top_k)
            sims, idxs = 1 - dists[0], idxs[0]
        out = self.ctx["work"].iloc[idxs].copy()
        out["dense_score"] = sims
        return out.reset_index(drop=True)

    def _retrieve_bm25(self, query: str, top_k: int) -> pd.DataFrame:
        toks = re.findall(r"\b\w+\b", query.lower())
        scores = self.ctx["bm25"].get_scores(toks)
        idxs = np.argsort(scores)[::-1][:top_k]
        out = self.ctx["work"].iloc[idxs].copy()
        out["bm25_score"] = scores[idxs]
        return out.reset_index(drop=True)

    def retrieve_hybrid(self, query: str, top_k: int = 20, alpha: float = 0.65) -> pd.DataFrame:
        d = self._retrieve_dense(query, top_k * 4)[["id", "dense_score"]]
        b = self._retrieve_bm25(query, top_k * 4)[["id", "bm25_score"]]
        m = d.merge(b, on="id", how="outer").fillna(0)
        m["dense_n"] = _normalize(m["dense_score"].values)
        m["bm25_n"] = _normalize(m["bm25_score"].values)
        m["hybrid_score"] = alpha * m["dense_n"] + (1 - alpha) * m["bm25_n"]
        m = m.sort_values("hybrid_score", ascending=False).head(top_k)
        return m.merge(self.ctx["work"], on="id", how="left").reset_index(drop=True)

    def rerank(self, query: str, candidates_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
        pairs = [(query, txt) for txt in candidates_df["semantic_text"].tolist()]
        scores = self.ctx["reranker"].predict(pairs)
        out = candidates_df.copy()
        out["rerank_score"] = scores
        return out.sort_values("rerank_score", ascending=False).head(top_k).reset_index(drop=True)

    def add_metadata_score(self, df_in: pd.DataFrame, query: str, **weight_kw) -> pd.DataFrame:
        weights = dict(METADATA_WEIGHTS)
        if weight_kw:
            weights.update(weight_kw)
        w_sem, w_pop, w_quality, w_fresh = (
            weights["w_sem"],
            weights["w_pop"],
            weights["w_quality"],
            weights["w_fresh"],
        )

        meta_cols = [
            "likes_n", "upvotes_n", "views_n", "uses_n",
            "author_reputation_n", "fork_count_n", "version_n", "freshness",
            "target_model", "difficulty", "language",
        ]
        merge_cols = ["id"] + [c for c in meta_cols if c not in df_in.columns]
        out = df_in.merge(self.ctx["work_meta"][merge_cols], on="id", how="left") if len(merge_cols) > 1 else df_in.copy()

        out["popularity"] = (
            0.45 * out["likes_n"] + 0.30 * out["upvotes_n"]
            + 0.15 * out["views_n"] + 0.10 * out["uses_n"]
        )
        out["quality"] = (
            0.6 * out["author_reputation_n"]
            + 0.25 * out["fork_count_n"]
            + 0.15 * out["version_n"]
        )
        sem_src = out["rerank_score"] if "rerank_score" in out.columns else out.get("dense_score", out["rrf_score"])
        out["semantic_n"] = _normalize(np.asarray(sem_src, dtype=float))

        tm_hint = detect_target_model(query)
        diff_hint = detect_difficulty_hint(query)
        lang_hint = detect_query_language(query)

        out["target_compat"] = [target_model_compat_score(tm, tm_hint) for tm in out["target_model"]]
        out["difficulty_compat"] = [difficulty_compat_score(d, diff_hint) for d in out["difficulty"]]
        out["language_compat"] = [language_compat_score(lang, lang_hint) for lang in out["language"]]
        out["constraint_mult"] = out["target_compat"] * out["difficulty_compat"] * out["language_compat"]

        out["final_score"] = (
            w_sem * out["semantic_n"]
            + w_pop * out["popularity"]
            + w_quality * out["quality"]
            + w_fresh * out["freshness"]
        ) * out["constraint_mult"]

        out["query_target_model"] = tm_hint
        out["query_intent"] = detect_query_intent(query)
        return out.sort_values("final_score", ascending=False).reset_index(drop=True)

    def mmr_diversify(self, df_in: pd.DataFrame, lambda_rel: float, top_k: int) -> pd.DataFrame:
        if len(df_in) <= top_k:
            return df_in.copy()
        cand = df_in.copy().reset_index(drop=True)
        vecs = np.asarray(
            self.ctx["embedder"].encode(
                cand["semantic_text"].tolist(), normalize_embeddings=True, show_progress_bar=False
            )
        )
        selected, remaining = [0], set(range(1, len(cand)))
        rel = _normalize(cand["final_score"].values)
        while len(selected) < min(top_k, len(cand)) and remaining:
            best_i, best_val = None, -1e9
            for i in list(remaining):
                sim = max(np.dot(vecs[i], vecs[j]) for j in selected)
                mmr = lambda_rel * rel[i] - (1 - lambda_rel) * sim
                if mmr > best_val:
                    best_i, best_val = i, mmr
            selected.append(best_i)
            remaining.remove(best_i)
        return cand.iloc[selected].reset_index(drop=True)

    @staticmethod
    def reciprocal_rank_fusion(rank_lists: List[pd.DataFrame], k_const: int = 60, top_n: int = 200) -> pd.DataFrame:
        scores: dict = {}
        for rl in rank_lists:
            if rl is None or len(rl) == 0:
                continue
            for rank, _id in enumerate(rl["id"].tolist()):
                scores[_id] = scores.get(_id, 0.0) + 1.0 / (k_const + rank + 1)
        if not scores:
            return pd.DataFrame(columns=["id", "rrf_score"])
        rows = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
        return pd.DataFrame(rows, columns=["id", "rrf_score"])

    def search_baseline(
        self, query: str, top_k: int = 10, adaptive: bool = True, diversify: bool = True
    ) -> Tuple[pd.DataFrame, float, dict]:
        t0 = time.perf_counter()
        stage1 = self.retrieve_hybrid(query, top_k=max(60, top_k * 6))
        stage2 = self.rerank(query, stage1, top_k=max(40, top_k * 4))
        intent = detect_query_intent(query)
        kw = intent_adaptive_weights(intent) if adaptive else dict(METADATA_WEIGHTS)
        stage3 = self.add_metadata_score(stage2, query, **kw)
        out = self.mmr_diversify(stage3, 0.82, top_k) if diversify else stage3.head(top_k)
        meta = {
            "intent": intent,
            "target_model": detect_target_model(query),
            "rewrites": None,
        }
        return out.reset_index(drop=True), (time.perf_counter() - t0) * 1000, meta

    def search_agentic(
        self,
        query: str,
        top_k: int = 10,
        n_rewrites: int = 6,
        candidate_pool: int = 120,
        lambda_rel: float = 0.65,
        use_reranker: bool = True,
        adaptive: bool = True,
        diversify: bool = True,
    ) -> Tuple[pd.DataFrame, float, dict]:
        t0 = time.perf_counter()
        rewrites = self.rewriter.rewrite(query, n=n_rewrites)
        rank_lists = [
            self.retrieve_hybrid(q, top_k=candidate_pool)[["id"]] for q in rewrites
        ]
        fused = self.reciprocal_rank_fusion(rank_lists, top_n=candidate_pool * 2)
        if len(fused) == 0:
            return self.ctx["work"].iloc[0:0].copy(), (time.perf_counter() - t0) * 1000, {"rewrites": rewrites}

        fused_full = fused.merge(self.ctx["work"], on="id", how="left")
        if use_reranker:
            rer = self.rerank(query, fused_full, top_k=max(60, top_k * 6))
        else:
            rer = fused_full.head(max(60, top_k * 6)).copy()
            rer["rerank_score"] = rer["rrf_score"]

        intent = detect_query_intent(query)
        kw = intent_adaptive_weights(intent) if adaptive else dict(METADATA_WEIGHTS)
        scored = self.add_metadata_score(rer, query, **kw)
        out = (
            self.mmr_diversify(scored, lambda_rel, top_k)
            if diversify and len(scored) > top_k
            else scored.head(top_k)
        )
        meta = {
            "intent": intent,
            "target_model": detect_target_model(query),
            "rewrites": rewrites,
        }
        return out.reset_index(drop=True), (time.perf_counter() - t0) * 1000, meta

    def run(self, pipeline: str, query: str, top_k: int, **kwargs) -> Tuple[pd.DataFrame, float, dict]:
        if pipeline == "baseline":
            return self.search_baseline(query, top_k=top_k)
        if pipeline == "agentic_full":
            return self.search_agentic(query, top_k=top_k, use_reranker=True, **kwargs)
        return self.search_agentic(query, top_k=top_k, use_reranker=False, **kwargs)
