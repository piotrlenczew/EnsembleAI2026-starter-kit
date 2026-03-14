import ast

def ast_chunker(content: str):
    """
    Chunk Python code by AST nodes:
    - functions
    - classes
    - methods
    """

    chunks = []
    lines = content.split("\n")

    try:
        tree = ast.parse(content)
    except SyntaxError:
        # fallback to whole file if parsing fails
        return [content]

    for node in ast.walk(tree):

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):

            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                start = node.lineno - 1
                end = node.end_lineno

                chunk = "\n".join(lines[start:end])
                chunks.append(chunk)

    # fallback if nothing detected
    if not chunks:
        return [content]

    return chunks
