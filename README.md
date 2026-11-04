## Installation

Run all commands from the repository root.

Use a Python version compatible with the project's `requires-python` setting in `pyproject.toml`.

### macOS / Linux

Create a project-local virtual environment:

```bash
python3 -m venv .venv
```

Install the package and development dependencies inside it:

```bash
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e ".[dev]"
```

Verify the interpreter and environment:

```bash
./.venv/bin/python -c "import sys; print('Interpreter:', sys.executable); print('Virtual environment:', sys.prefix != sys.base_prefix)"
```

The interpreter path should point inside this repository's `.venv`, and the virtual-environment check should print `True`.

### Windows PowerShell

Create and install into the project-local environment:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Verify it:

```powershell
.\.venv\Scripts\python.exe -c "import sys; print('Interpreter:', sys.executable); print('Virtual environment:', sys.prefix != sys.base_prefix)"
```

### Optional activation

Activation makes shorter commands available in the current terminal.

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, use `python -m pip`, `python -m pytest`, and `python -m cluster_representation`.

The explicit interpreter commands below work without activation. No global installation, `sudo pip`, or `pip install --user` is needed.

Keep `.venv/` in `.gitignore`. Recreate the environment on each machine rather than committing or copying it.

## Run the workflow

From the repository root, run the complete configured pipeline.

macOS / Linux:

```bash
./.venv/bin/python -m cluster_representation --config config/default.yml
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m cluster_representation --config config/default.yml
```

Run an individual stage after its prerequisites have been saved.

macOS / Linux:

```bash
./.venv/bin/python -m cluster_representation \
  --config config/default.yml \
  --step generate_dataset
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m cluster_representation --config config/default.yml --step generate_dataset
```

Supported stages:

- `generate_dataset`
- `split_dataset`
- `fit_preprocessor`
- `train_model`
- `build_prototypes`
- `transform_test`
- `evaluate`

Each stage validates its prerequisite artifacts and reports missing or incompatible inputs.

Existing artifacts are not overwritten unless `output.overwrite: true` is configured. Use a new `output.run_name` for a separate experiment.

All application implementation lives in `src/cluster_representation/`. The CLI and notebooks call the same stage functions.

## Notebooks

Install development dependencies into `.venv` before starting Jupyter. They must include JupyterLab and `ipykernel`.

Launch Jupyter using the virtual environment's Python, with `notebooks/` as the working directory.

macOS / Linux:

```bash
cd notebooks
../.venv/bin/python -m jupyterlab
```

Windows PowerShell:

```powershell
Set-Location notebooks
..\.venv\Scripts\python.exe -m jupyterlab
```

Select the Python kernel belonging to this repository's `.venv`.

If it is not listed, register it from the repository root:

macOS / Linux:

```bash
./.venv/bin/python -m ipykernel install \
  --prefix .venv \
  --name synthetic-cluster-representation \
  --display-name "Python (.venv — Synthetic Cluster Representation)"
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name synthetic-cluster-representation --display-name "Python (.venv - Synthetic Cluster Representation)"
```

Restart Jupyter and select that kernel. Registration is local to `.venv`.

To inspect the interpreter used by a notebook:

```python
import sys

print(sys.executable)
```

It should point inside the repository's `.venv`.

Notebooks load `../config/default.yml`, call package stage functions, and display returned summaries. All processing, model logic, validation, plotting, and artifact saving remain in `src/`.

Notebooks must not install dependencies, modify `sys.path`, duplicate implementation, or depend on variables from other notebooks.

Run stages in order. Each stage loads its saved prerequisites.

To choose another configuration, set `CLUSTER_REPRESENTATION_CONFIG` before launching Jupyter. An absolute path avoids working-directory ambiguity.

## Tests

Run tests from the repository root using the virtual environment.

macOS / Linux:

```bash
./.venv/bin/python -m pytest
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Tests cover configuration validation, reproducibility, exact scenario counts, split integrity, training-only preprocessing, constrained fitting, serialization, prototype construction, feature replacement, artifact compatibility, model-template errors, and a small end-to-end pipeline.