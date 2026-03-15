import os
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from chunkers.ast_chunker import ast_chunker

dense_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def chunk_retrieve(root_dir: str, prefix: str, suffix: str, extension: str, min_lines: int = 10, top_k: int = 15) -> list[dict]:
    """
    Hybrid retrieval without reranking:
    chunk -> BM25 + Dense Retriever -> merge -> return top_k
    """

    def prepare_bm25_str(s: str) -> list[str]:
        return "".join(c if c.isalnum() else " " for c in s.lower()).split()

    corpus_bm25_tokens = []
    corpus_texts = []
    file_names = []

    # -----------------------
    # Build corpus
    # -----------------------
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extension):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if len(lines) >= min_lines:
                            content = "\n".join(lines)
                            chunks = ast_chunker(content)
                            for chunk in chunks:
                                corpus_texts.append(chunk)
                                corpus_bm25_tokens.append(prepare_bm25_str(chunk))
                                file_names.append(file_path)
                except Exception:
                    pass

    if not corpus_texts:
        return []

    query = (prefix + " " + suffix).lower()
    query_text = f"query: {prefix} {suffix}".lower()

    # -----------------------
    # BM25 retrieval
    # -----------------------
    bm25 = BM25Okapi(corpus_bm25_tokens)
    bm25_scores = bm25.get_scores(prepare_bm25_str(query))

    # -----------------------
    # Dense retrieval
    # -----------------------
    safe_corpus = [f"passage: {doc}" for doc in corpus_texts]
    query_emb = dense_model.encode(query_text, normalize_embeddings=True)
    doc_embs = dense_model.encode(safe_corpus, normalize_embeddings=True)
    dense_scores = np.dot(doc_embs, query_emb)

    # -----------------------
    # Merge scores (optional: sum or average)
    # -----------------------
    merged_scores = bm25_scores + dense_scores  # simple sum, can adjust weighting
    top_indices = np.argsort(merged_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "file": file_names[idx],
            "content": corpus_texts[idx]
        })

    return results