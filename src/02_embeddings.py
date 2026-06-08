# embeddings.py
# Initialises the sentence-transformer embedding model and builds a FAISS
# IndexFlatIP (cosine-equivalent) over the full prompt corpus.
# Embedding matrices are cached to disk as .npy files so subsequent runs
# skip the expensive encode step. Falls back to sklearn NearestNeighbors
# if faiss-cpu is not available on the current machine.
# Also runs a head-to-head comparison of three candidate embedding models
# (MiniLM, BGE-small, E5-small) and activates the best-performing one
# for all downstream modules.
# Depends on: preprocessing.py (work["semantic_text"])

import torch
from sentence_transformers import SentenceTransformer

print("CUDA:", torch.cuda.is_available())
print("MPS:", torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False)

device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
print("Chosen device:", device)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME, device=device)

print("Model device:", embedder.device)

# ── Model catalogue and index builder ─────────────────────────────────────────

RE_RANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CANDIDATE_EMBED_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-small-v2",
]

import torch
from pathlib import Path


def _cache_name(model_name, max_chars=700):
    safe = model_name.replace('/', '__').replace('-', '_')
    return Path(f"emb_cache_{safe}_{max_chars}.npy")


def build_embedding_index(model_name, semantic_texts, max_chars=700, batch_size=64):
    emb_model = SentenceTransformer(model_name, device=device)
    cache_path = _cache_name(model_name, max_chars)

    if cache_path.exists():
        arr = np.load(cache_path)
    else:
        # Truncate to max_chars: the retrieval signal lives in the first ~700
        # characters (title, tags, category, opening content). Longer bodies add
        # noise and explode encode time on CPU.
        corpus_fast = [t[:max_chars] for t in semantic_texts]
        arr = emb_model.encode(
            corpus_fast,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        arr = np.asarray(arr, dtype=np.float32)
        np.save(cache_path, arr)

    use_faiss_local = True
    try:
        import faiss
        # IndexFlatIP with L2-normalised vectors is equivalent to cosine similarity.
        idx = faiss.IndexFlatIP(arr.shape[1])
        idx.add(arr)
    except Exception:
        use_faiss_local = False
        idx = NearestNeighbors(n_neighbors=100, metric="cosine")
        idx.fit(arr)

    return emb_model, arr, idx, use_faiss_local


embedder, emb, index, use_faiss = build_embedding_index(
    EMBED_MODEL_NAME,
    work["semantic_text"].tolist(),
    max_chars=700,
    batch_size=64,
)

print("Device:", device)
print("Embeddings shape:", emb.shape)
print("Using FAISS:", use_faiss)
print("Active embedding model:", EMBED_MODEL_NAME)

# ── Activate best model after evaluation (Cell 16) ────────────────────────────
# BEST_EMBED_MODEL is set by evaluation.py after the head-to-head benchmark.
# Re-run this block after evaluation.py to switch to the best model.

EMBED_MODEL_NAME = BEST_EMBED_MODEL
embedder, emb, index, use_faiss = build_embedding_index(
    EMBED_MODEL_NAME,
    work["semantic_text"].tolist(),
    max_chars=700,
    batch_size=64,
)

print("Active model switched to:", EMBED_MODEL_NAME)
print("Using FAISS:", use_faiss)
