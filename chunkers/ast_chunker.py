import ast

def ast_chunker(content: str):
    chunks = []
    lines = content.split("\n")

    try:
        tree = ast.parse(content)
    except (SyntaxError, IndentationError):
        # Fallback to basic_chunker if AST fails
        from chunkers.basic_chunker import basic_chunker
        return basic_chunker(content)

    # Helper to get source from lines safely
    def get_lines(start_node, end_node):
        return "\n".join(lines[start_node.lineno - 1 : end_node.end_lineno])

    # 1. Handle Module-level "Header" (Imports, Constants, Module Docstrings)
    # Find where the first major block (Class/Function) starts
    first_major_node = next((n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))), None)
    
    if first_major_node and first_major_node.lineno > 1:
        header = "\n".join(lines[:first_major_node.lineno - 1]).strip()
        if header:
            chunks.append(header)
    elif not first_major_node:
        # File is just flat code, return as is (or let basic_chunker handle it)
        from chunkers.basic_chunker import basic_chunker
        return basic_chunker(content)

    # 2. Iterate through Top-Level Nodes
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Top-level function: Add as one chunk
            chunks.append(get_lines(node, node))

        elif isinstance(node, ast.ClassDef):
            # CLASS HANDLING: We want to avoid duplicating method code
            # First, extract just the "Header" of the class (Definition + Class variables)
            # We find the first method and take everything before it.
            first_method = next((n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
            
            if first_method:
                class_header = "\n".join(lines[node.lineno - 1 : first_method.lineno - 1]).strip()
                if class_header:
                    chunks.append(class_header)
                
                # Now add each method as its own chunk
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        chunks.append(get_lines(child, child))
                    elif isinstance(child, (ast.Assign, ast.AnnAssign)) and child.lineno >= first_method.lineno:
                        # Catch class variables that might appear after/between methods
                        chunks.append(get_lines(child, child))
            else:
                # Class has no methods (e.g., a simple Data Class or Exception)
                chunks.append(get_lines(node, node))

        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr)):
            # Logic at module level that isn't a function/class
            chunk = get_lines(node, node).strip()
            if chunk:
                chunks.append(chunk)

    # Final cleanup: Remove empty strings or very small fragments
    return [c for c in chunks if len(c.strip()) > 20]