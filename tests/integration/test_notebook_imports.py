import ast
import json
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).resolve().parents[2] / "notebooks"
NOTEBOOK_FILES = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))


@pytest.mark.parametrize(
    "notebook_path",
    NOTEBOOK_FILES,
    ids=[nb.name for nb in NOTEBOOK_FILES],
)
def test_notebook_project_imports(notebook_path: Path) -> None:

    content = json.loads(notebook_path.read_text(encoding="utf-8"))

    import_statements: list[str] = []

    for cell in content.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source_lines = cell.get("source", [])
        code = "".join(source_lines)

        # Filter out jupyter magic commands like %matplotlib, !pip, etc.
        cleaned_lines = [line for line in code.splitlines() if not line.strip().startswith(("%", "!"))]
        cleaned_code = "\n".join(cleaned_lines)

        try:
            tree = ast.parse(cleaned_code)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("project"):
                        import_statements.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("project"):
                names = ", ".join(a.name for a in node.names)
                import_statements.append(f"from {node.module} import {names}")

    # Deduplicate while preserving order
    unique_imports = list(dict.fromkeys(import_statements))

    # Execute each import statement in an isolated execution namespace
    exec_scope: dict[str, object] = {}
    for stmt in unique_imports:
        try:
            exec(stmt, exec_scope)
        except Exception as exc:
            pytest.fail(f"Failed to execute notebook import in {notebook_path.name}: '{stmt}'. Error: {exc}")
