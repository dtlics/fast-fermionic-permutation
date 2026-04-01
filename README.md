# Ancilla-Free Fermionic Permutation on 2D Qubit Grids

Research code for a MICRO 2026 paper on ancilla-free fermionic permutation (FP)
with O(sqrt(N))-depth circuits on L x L nearest-neighbor grids.

## Four Baselines

All implement the **same unitary** -- they differ only in circuit structure.

| Baseline | Method | CNOT Depth | Ancillas |
|----------|--------|------------|----------|
| 1 | 1D snake OET sort | O(N) ~ 2L^2 | 0 |
| 2 | Row-Col-Row + Ancilla Gamma | O(sqrt(N)) ~ 20L | L |
| 3 | Row-Col-Row + Ancilla-free Gamma (primitives) | O(sqrt(N)) ~ 24L | 0 |
| 4 | Row-Col-Row + Ancilla-free Gamma (pipelined) | O(sqrt(N)) ~ 22L | **0** |

Baseline 4 is our best: O(sqrt(N)) depth with zero ancillas.

## Project Structure

```
common/                  # Core modules
  grid.py                # Grid topology, snake JW indexing
  fswap.py               # FSWAP gate definition
  hall_decomposition.py   # Hall 3-stage Row-Col-Row decomposition
  oet_sort.py            # Odd-even transposition sort
  gamma_ancilla.py       # Baseline 2: Gamma with ancillas (7L-3 depth)
  gamma_primitive.py     # Baseline 3: Ancilla-free Gamma (9L+12 depth)
  gamma_pipeline.py      # Baseline 4: Pipelined Gamma (8L+O(1) depth)
  fp_1d.py               # Baseline 1: 1D snake FSWAP sort
  fp_2d.py               # Baselines 2-4: unified 2D FP builder
  metrics.py             # Spacetime volume, union-bound fidelity, gate counting
  tests/                 # pytest test suite
exp1_fp/                 # Experiment 1: FP benchmarking
exp2_ffft/               # Experiment 2: 2D FFFT
exp3_syk/                # Experiment 3: Sparse SYK simulation
paper_figures/           # Generated figures
docs/                    # Documentation and writeups
archive/                 # Legacy code (notebooks, visualization)
```

## Quick Start

```bash
pip install -r requirements.txt
pytest common/tests/ -v
```

## Usage

```python
from common.fp_1d import build_fp_1d, build_benchmark_permutation
from common.fp_2d import build_fp_2d, GammaMethod
from common.metrics import count_resources, spacetime_volume, counting_union_bound_fidelity

L = 5
perm = build_benchmark_permutation(L, "reverse")

# Baseline 1: 1D
circ_1d, qubits_1d = build_fp_1d(L, perm)

# Baseline 4: Best (pipelined, 0 ancillas)
circ_4, sys_4, _ = build_fp_2d(L, perm, GammaMethod.PIPELINED)

# Compare resources
r1 = count_resources(circ_1d, L)
r4 = count_resources(circ_4, L)
print(f"1D CNOT depth: {r1['cnot_depth']}, 2D pipelined: {r4['cnot_depth']}")
```
