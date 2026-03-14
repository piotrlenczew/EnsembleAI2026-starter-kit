import os
from rank_bm25 import BM25Okapi
from chunkers.basic_chunker import basic_chunker


def find_bm25_chunk(root_dir: str, prefix: str, suffix: str, min_lines: int = 10):
    """
    Retrieve the most relevant code chunk using BM25.
    """

    def prepare_bm25_str(s: str) -> list[str]:
        return "".join(c if c.isalnum() else " " for c in s.lower()).split()

    corpus = []
    file_names = []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) >= min_lines:
                        file_names.append(file_path)
                        content = "\n".join(lines)
                        chunks = basic_chunker(content)
                        for chunk in chunks:
                            corpus.append(prepare_bm25_str(chunk))
            except Exception:
                pass

    if not corpus:
        return None, None

    query = (prefix + " " + suffix).lower()
    query = prepare_bm25_str(query)

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query)
    best_idx = scores.argmax()

    return file_names[best_idx] if file_names else None