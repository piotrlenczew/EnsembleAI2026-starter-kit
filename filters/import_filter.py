import os
import re

def extract_imports(text: str) -> set[str]:
    """Extract module paths from import statements."""
    modules = set()

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("import "):
            parts = line.replace("import", "").split(",")
            for p in parts:
                modules.add(p.strip())

        elif line.startswith("from "):
            match = re.match(r"from\s+([a-zA-Z0-9_.]+)\s+import", line)
            if match:
                modules.add(match.group(1))

    return modules


def module_to_paths(module: str, root_dir: str, extension: str):
    """
    Convert module path to possible filesystem paths.
    """
    parts = module.split(".")

    file_path = os.path.join(root_dir, *parts) + extension
    package_init = os.path.join(root_dir, *parts, "__init__" + extension)

    paths = []

    if os.path.exists(file_path):
        paths.append(file_path)

    if os.path.exists(package_init):
        paths.append(package_init)

    return paths


def search_imports(query: str, root_dir: str = ".", extension: str = ".py", max_depth: int = 2) -> set[str]:
    """
    Resolve imported modules from query and follow imports recursively.
    Supports packages.
    """

    modules = extract_imports(query)

    visited_modules = set()
    candidate_files = set()

    depth = 0

    while modules and depth < max_depth:
        new_modules = set()
        for module in modules:
            if module in visited_modules:
                continue

            visited_modules.add(module)
            paths = module_to_paths(module, root_dir, extension)

            for path in paths:
                candidate_files.add(path)

                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    new_modules |= extract_imports(content)
                except Exception:
                    pass

        modules = new_modules
        depth += 1

    return candidate_files