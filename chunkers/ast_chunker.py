import ast

def ast_chunker(content: str):

    chunks = []
    lines = content.split("\n")

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return [content]

    # -------- HEADER CHUNK --------
    first_def_line = None

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first_def_line = node.lineno
            break

    if first_def_line:
        header = "\n".join(lines[:first_def_line-1])
        if header.strip():
            chunks.append(header)

    # -------- MAIN CHUNKS --------
    for node in tree.body:

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):

            start = node.lineno - 1
            end = node.end_lineno

            chunk = "\n".join(lines[start:end])
            chunks.append(chunk)

        elif isinstance(node, ast.ClassDef):

            # chunk the class itself
            start = node.lineno - 1
            end = node.end_lineno
            class_chunk = "\n".join(lines[start:end])
            chunks.append(class_chunk)

            # chunk methods separately
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = child.lineno - 1
                    end = child.end_lineno

                    method_chunk = "\n".join(lines[start:end])
                    chunks.append(method_chunk)

    if not chunks:
        return [content]

    return chunks