"""Выполнить ноутбук в IPython и сохранить outputs + HTML без запуска kernel-server.

Такой режим подходит для ограниченного окружения без доступа к портам. Ячейки
исполняются последовательно в одном свежем процессе. Код ноутбука доверенный.
"""

import argparse
import contextlib
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", nargs="?", default="notebooks/01_municipal_data_audit.ipynb")
    parser.add_argument("--html", default="artifacts/eda/municipal_data_audit.html")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    (root / ".cache").mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
    os.environ.setdefault("IPYTHONDIR", str(root / ".cache/ipython"))
    import nbformat
    from nbconvert import HTMLExporter
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output

    path = (root / args.notebook).resolve()
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    shell = InteractiveShell.instance()
    shell.history_manager.enabled = False
    previous = Path.cwd()
    try:
        os.chdir(root)
        count = 0
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            count += 1
            print(f"Ячейка {count}: {cell.id}", flush=True)
            with capture_output(stdout=True, stderr=True, display=True) as captured:
                result = shell.run_cell(cell.source, store_history=False)
            error = result.error_before_exec or result.error_in_exec
            if error:
                raise RuntimeError(f"Ошибка ячейки {cell.id}:\n{captured.stdout}\n{captured.stderr}") from error
            outputs = []
            for stream, content in [("stdout", captured.stdout), ("stderr", captured.stderr)]:
                if content:
                    outputs.append(nbformat.v4.new_output("stream", name=stream, text=content))
            for rich in captured.outputs:
                outputs.append(nbformat.v4.new_output("display_data", data=rich.data, metadata=rich.metadata))
            cell.execution_count = count
            cell.outputs = outputs
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        exporter = HTMLExporter()
        exporter.exclude_input_prompt = True
        exporter.exclude_output_prompt = True
        html, _ = exporter.from_notebook_node(notebook)
        report = root / args.html
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(html, encoding="utf-8")
        print(f"Готово: {path.relative_to(root)}")
        print(f"HTML: {report.relative_to(root)}")
    finally:
        os.chdir(previous)
        with contextlib.suppress(Exception):
            shell.history_manager.end_session()


if __name__ == "__main__":
    main()
