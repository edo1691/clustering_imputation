from pathlib import Path

import nbformat


def main():
    folder = Path(__file__).resolve().parent

    for path in sorted(folder.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in path.parts:
            continue

        notebook = nbformat.read(path, as_version=4)

        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None

            cell.metadata.pop("execution", None)

        notebook.metadata.pop("widgets", None)

        nbformat.write(notebook, path)
        print(f"Cleaned: {path.relative_to(folder)}")


if __name__ == "__main__":
    main()