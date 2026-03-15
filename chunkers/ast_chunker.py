import ast

def ast_chunker(content: str, max_chars: int = 1500):
    """
    AST-based chunker that:
    1. Prepends class signatures to methods for context.
    2. Merges small adjacent nodes to avoid 'micro-chunks'.
    3. Falls back to basic chunking on syntax errors.
    """
    chunks = []
    lines = content.split("\n")

    try:
        tree = ast.parse(content)
    except (SyntaxError, IndentationError):
        from chunkers.basic_chunker import basic_chunker
        return basic_chunker(content)

    def get_source(node):
        # end_lineno is 3.8+, using a safe fallback just in case
        end = getattr(node, 'end_lineno', node.lineno)
        return "\n".join(lines[node.lineno - 1 : end])

    def add_chunk(text):
        if text.strip():
            chunks.append(text.strip())

    # --- 1. Handle Module Header ---
    first_major = next((n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if first_major and first_major.lineno > 1:
        add_chunk("\n".join(lines[:first_major.lineno - 1]))

    # --- 2. Process Body with Grouping ---
    current_group = []
    current_group_len = 0

    for node in tree.body:
        node_text = ""
        
        if isinstance(node, ast.ClassDef):
            # Flush current group before starting a class
            if current_group:
                add_chunk("\n\n".join(current_group))
                current_group, current_group_len = [], 0
            
            # Get class signature (everything before first method/nested class)
            first_child = next((n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))), None)
            if first_child:
                class_sig = "\n".join(lines[node.lineno - 1 : first_child.lineno - 1]).strip()
            else:
                class_sig = get_source(node).strip()
            
            # Process children: prepend class signature to methods
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_src = get_source(child)
                    # Prepend class signature + '...' to show it's a snippet
                    add_chunk(f"{class_sig}\n    ...\n    {method_src}")
                else:
                    # Handle class-level variables/docstrings that aren't methods
                    child_src = get_source(child)
                    if len(child_src) > 20:
                         add_chunk(f"{class_sig}\n    {child_src}")
            continue

        # For Top-level functions or standalone logic
        node_text = get_source(node)
        
        # Grouping Logic: if the node is small, bundle it with others
        if len(node_text) + current_group_len < max_chars:
            current_group.append(node_text)
            current_group_len += len(node_text)
        else:
            # Flush existing group and start new one
            if current_group:
                add_chunk("\n\n".join(current_group))
            current_group = [node_text]
            current_group_len = len(node_text)

    # Final flush
    if current_group:
        add_chunk("\n\n".join(current_group))

    return chunks