"""Validate notebook JSON and Python syntax without running training or downloads."""
import ast
import json
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell

root = Path(__file__).resolve().parents[1]
shell = InteractiveShell.instance()
checked = []
for path in sorted((root / 'notebooks').glob('*.ipynb')):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code_cells = 0
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code' and cell.source.strip():
            transformed = shell.input_transformer_manager.transform_cell(cell.source)
            ast.parse(transformed, filename=f'{path.name}:cell-{i}')
            code_cells += 1
    checked.append({'notebook': path.name, 'code_cells': code_cells})
print(json.dumps({'validated': checked}, indent=2))
