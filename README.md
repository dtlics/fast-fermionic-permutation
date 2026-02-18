# fast-fermionic-permutation

Research code for analyzing and visualizing fermionic permutation circuits on 2D grids.
The repository compares:

1. Baseline OpenFermion snake-order permutation.
2. A custom ancilla-assisted row-CNOT construction.

## Start here (main content)

The primary content of this repo is the notebook: `FP.ipynb`.
If you are new to this project, go through the notebook first, top to bottom.

Use the Python files only when needed:

- `resource_estimation.py` and `visualize_fermionic.py` hold implementation details and helper APIs used by the notebook.
- You typically do not need to read these files unless you want to inspect internals or modify behavior.

## Repository layout

- `FP.ipynb`: interactive notebook for experiments and plots.
- `resource_estimation.py`: supporting module for circuit/resource counting and scaling studies.
- `visualize_fermionic.py`: supporting module for decomposition and visualization helpers.
- `resource_estimation_cache.csv`: cached resource-estimation rows to avoid expensive reruns.
- `visualization_demo_5x5/`: tracked demo visualization frames.
- `visualization_hall_demo_5x5/`: additional visualization assets.
- `Fermionic Permutation — Draft Writeup.md`: draft writeup notes.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart (recommended): open the notebook

```bash
jupyter lab FP.ipynb
```

or

```bash
jupyter notebook FP.ipynb
```

## Optional: script-level usage (only if needed)

```python
from resource_estimation import run_weighted_metric_study_with_cache, build_summary_table

df = run_weighted_metric_study_with_cache(
    L_values=[4, 6],
    permutation_kind="random",
    seed=0,
)
summary = build_summary_table(df)
print(summary[["method", "L", "weighted_count_metric", "success_probability"]])
```

## Optional: generate visualization frames directly

```python
from visualize_fermionic import visualize_permutation

L = 5
perm = list(range(L * L - 1, -1, -1))  # Reverse permutation example.
visualize_permutation(L, perm, output_dir="visualization_demo_5x5")
```

## Cache behavior

`run_weighted_metric_study_with_cache(...)` chooses cache path in this order:

1. Explicit `cache_csv_path=...` argument.
2. Environment variable `FFP_CACHE_CSV_PATH`.
3. Repo-local default: `resource_estimation_cache.csv` (next to `resource_estimation.py`).

Example override:

```bash
export FFP_CACHE_CSV_PATH=/tmp/resource_estimation_cache.csv
```

## Tracked artifacts policy

This repo intentionally tracks:

- `FP.ipynb`
- `resource_estimation_cache.csv`
- demo visualization PNG frames

OS metadata files such as `.DS_Store` are ignored.
