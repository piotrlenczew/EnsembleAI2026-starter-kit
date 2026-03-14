def basic_chunker(
    text: str,
    chunk_size: int = 80,
    overlap: int = 20
):
    """
    Split code into overlapping chunks.

    Parameters
    ----------
    text : str
        File content.
    chunk_size : int
        Number of lines per chunk.
    overlap : int
        Overlap between chunks.

    Returns
    -------
    list[str]
        List of code chunks.
    """

    lines = text.split("\n")

    if len(lines) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(lines):

        end = start + chunk_size
        chunk_lines = lines[start:end]

        chunk = "\n".join(chunk_lines).strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks