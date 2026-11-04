# Synthetic Cluster Representation

A reproducible Python package for independently generated numeric features,
minimum-size-constrained K-means, training-derived cluster prototypes, and
scenario-based replacement of test features. YAML can select
`constrained_kmeans`, `constrained_coclustering`, or
`robust_constrained_coclustering` through `model.config_path`.

> This is a software-validation example, not a realistic customer-behavior
> model. Independent synthetic features do not establish meaningful behavioral
> clusters. Replacing values alone does not establish anonymity or regulatory
> compliance.

## Installation

Python 3.9 or newer is supported. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
```

The required dependencies include NumPy, pandas, PyYAML, scikit-learn,
PyArrow/Parquet, joblib, and `k-means-constrained`. The constrained model uses
the latter package's `KMeansConstrained` estimator with explicit mappings:
`n_clusters`, `size_min`, `random_state`, `max_iter`, and `n_init`. Its fitted
training cluster sizes are checked against the configured minimum. Test
assignments use nearest fitted centroids and are not size constrained.

## Run the workflow

Run the complete configured pipeline:

```bash
python -m cluster_representation --config config/default.yml
# or, after installation:
cluster-representation --config config/default.yml
```

Run a single stage after its prerequisites have been saved:

```bash
python -m cluster_representation \
  --config config/default.yml \
  --step generate_dataset
```

Supported `--step` values are `generate_dataset`, `split_dataset`,
`fit_preprocessor`, `train_model`, `build_prototypes`, `transform_test`, and
`evaluate`. Each stage validates prerequisite artifacts and gives an actionable
error if an earlier stage is missing. Existing output artifacts are not
overwritten unless `output.overwrite: true` is set; a new `output.run_name` is
the recommended way to start a separate run.

All implementation is under `src/cluster_representation/`; tests are in
`tests/`. The package entry point and installed console command call the same
stage APIs used by the notebooks.

## Configuration

`config/default.yml` documents the complete configuration. Feature names must
be contiguous and ordered (`ft_1`, `ft_2`, ...). Supported distributions and
their parameters are:

| Distribution | Parameters | Meaning |
|---|---|---|
| `normal` | `mean`, `std` | Normal mean and positive standard deviation |
| `uniform` | `low`, `high` | Lower-inclusive and upper-exclusive bounds |
| `exponential` | `scale` | Positive scale (mean) |
| `poisson` | `lam` | Non-negative event rate |
| `binomial` | `n`, `p` | Number of trials and success probability |
| `lognormal` | `mean`, `sigma` | Mean and positive standard deviation of the underlying normal |

Relative paths are resolved against the YAML file that declares them. For
example, `model.config_path` in `config/default.yml` resolves within `config/`,
and `output.root: ../artifacts` resolves from that same directory.
Select the implemented K-means model or either co-clustering template by
changing `model.config_path` to `models/constrained_kmeans.yml`,
`models/constrained_coclustering.yml`, or
`models/robust_constrained_coclustering.yml`, respectively. The two
co-clustering templates validate and load through the shared pipeline, but
training stops with an explicit `NotImplementedError` until their algorithms
are implemented.

Configuration loading rejects unknown keys, invalid distribution parameters,
invalid percentage schedules, incompatible model settings, and stage sequences
that omit prerequisites. `gdpr=1` means that replacement is required; `gdpr=0`
means original values are retained. The non-consent indicators are simulation
scenarios, not supervised prediction targets. Counts use
`floor(n_rows * percentage / 100)` and are exact for the full dataset.

## Data and modeling assumptions

* Each feature receives a deterministic random stream keyed by the master seed
  and feature name. Adding another feature does not alter existing features.
  Python's built-in `hash()` and global random state are not used.
* Non-consent scenarios are saved separately from generated feature values.
  Nested scenarios use prefixes of one reproducible permutation; independent
  scenarios use separate reproducible streams.
* The deterministic train/test split does not stratify on non-consent status.
  Saved split tables retain original IDs and their original ID order. Actual
  overall, train, and test scenario rates are reported separately.
* Imputation and optional standard scaling are fitted only on training
  features. IDs and `gdpr_*` columns are excluded from model inputs.
* All original `ft_*` features are used for assignment. Training memberships
  from the constrained fit define cluster assignments and original-unit
  feature medians; test memberships are never used to build prototypes.
* Minimum cluster size applies to fitted training clusters and therefore to
  prototype support. Replacement medians remain fractional when appropriate.
* For test customers, assignment happens before replacement. Every feature is
  replaced with its training-cluster median when `gdpr=1`, and every original
  feature is retained when `gdpr=0`. Original test values are saved separately
  for evaluation.
* The assumption is: “Original transactional features are accessible for
  cluster assignment. For simulated non-consenting customers, their direct
  downstream representation is replaced with training-derived cluster
  medians.”

## Artifacts

Each run is written under `artifacts/<run_name>/`, including the resolved
configuration, manifest, environment versions, generated and split Parquet
data, fitted preprocessor and model, prototype assignments and medians,
predictions, transformed scenario outputs, and machine-readable metrics.
Manifest records include stage-specific fingerprints, artifact dependencies,
schemas/parameters where relevant, training-ID references, and timestamps.
Changing only the selected replacement scenario does not invalidate model
training, prototypes, or test cluster assignments.

The evaluation reports clustering metrics in preprocessed model-input space
and replacement errors in original feature units. Overall MAE aggregates
features on different scales and should be interpreted accordingly. Metrics
that cannot be defined for the available cluster/sample configuration are
reported as unavailable. Replaced-row MAE is unavailable when no test rows are
replaced. Silhouette sampling uses a deterministic stream.

## Notebooks

Install the package first, then launch Jupyter from the `notebooks/` directory:

```bash
cd notebooks
jupyter lab
```

The notebooks use `../config/default.yml`, call package stage functions, and
display their result summaries. They do not modify `sys.path`, contain model or
data-processing implementations, or depend on state from another notebook.
Run stages in order; each notebook stage loads saved prerequisites.
To select a separate YAML configuration for a notebook run, set
`CLUSTER_REPRESENTATION_CONFIG` to its path before starting Jupyter.

## Tests

```bash
pytest
```

The tests cover configuration validation, reproducibility, exact scenario
counts, split integrity, training-only preprocessing, constrained fitting,
serialization, prototypes, replacement behavior, artifact checks, the future
model's explicit errors, and a small end-to-end pipeline.

## Future work (not implemented)

The registered `constrained_coclustering` and
`robust_constrained_coclustering` models are configuration-only. Both `fit()`
and `predict()` raise `NotImplementedError`; no simulated results are
generated. The constrained template has no robustness or trimming options;
the robust template additionally includes row and column trimming fractions.
Their notebooks only display the template configurations. Before implementing
either algorithm, define:

* How a new observation receives a row-cluster prediction.
* For the robust variant, whether trimmed observations may receive clusters
  and contribute prototypes.
* For the robust variant, how trimmed features participate in inference and
  reports.
* For the robust variant, whether trimmed observations/features contribute to
  prototype support.

Other planned research extensions are cluster-count selection, correlated
features and latent customer groups, customer-feature block structure,
outlier-robustness experiments, and downstream predictive evaluation using a
separate outcome `y`. None are implemented or claimed by this package.
