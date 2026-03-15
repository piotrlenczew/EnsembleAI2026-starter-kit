import os
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from chunkers.ast_chunker import ast_chunker
from filters.import_filter import search_imports

dense_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def filter_chunk_retrieve_with_rerank(root_dir: str, prefix: str, suffix: str, extension: str, min_lines: int = 10, top_k: int = 5) -> list[str]:
    """
    Hybrid retrieval:
    chunk -> BM25 + Dense Retriever -> merge -> Cross-Encoder reranker
    """

    def prepare_bm25_str(s: str) -> list[str]:
        return "".join(c if c.isalnum() else " " for c in s.lower()).split()

    candidate_files = search_imports(prefix + "\n" + suffix, root_dir, extension)
    if not candidate_files:
        print("No candidates found") 
    else:
        print("Candidates found")

    corpus_bm25_tokens = []
    corpus_texts = []
    file_names = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extension):
                file_path = os.path.join(dirpath, filename)
                if not candidate_files or file_path in candidate_files:
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

    # -----------------------
    # Build query
    # -----------------------

    query = (prefix + " " + suffix).lower()

    # -----------------------
    # BM25 retrieval
    # -----------------------

    bm25 = BM25Okapi(corpus_bm25_tokens)
    bm25_scores = bm25.get_scores(prepare_bm25_str(query))
    bm25_top_k = min(20, len(corpus_texts))
    bm25_idx = np.argsort(bm25_scores)[::-1][:bm25_top_k]

    # -----------------------
    # Dense retrieval
    # -----------------------

    query_emb = dense_model.encode(query, normalize_embeddings=True)
    doc_embs = dense_model.encode(corpus_texts, normalize_embeddings=True)
    dense_scores = np.dot(doc_embs, query_emb)
    dense_top_k = min(20, len(corpus_texts))
    dense_idx = np.argsort(dense_scores)[::-1][:dense_top_k]

    # -----------------------
    # Merge candidates
    # -----------------------

    candidate_idx = list(set(bm25_idx) | set(dense_idx))
    candidates = [corpus_texts[i] for i in candidate_idx]
    candidate_files = [file_names[i] for i in candidate_idx]

    # -----------------------
    # Rerank
    # -----------------------

    pairs = [(query, doc) for doc in candidates]
    rerank_scores = reranker.predict(pairs)
    top_indices = np.argsort(rerank_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "file": candidate_files[idx],
            "content": candidates[idx]
        })
    return results
