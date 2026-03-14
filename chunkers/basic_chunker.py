def basic_chunker(content, chunk_size=80, overlap=20):

    lines = content.split("\n")
    chunks = []

    start = 0
    while start < len(lines):
        chunk = "\n".join(lines[start:start+chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks