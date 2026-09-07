# Asymptotically Optimal Depth Fermionic Permutation on 2D Grids without Ancillas

Code, benchmark data, and figure pipeline for

> D. Li, S. Xu, and Y. Ding, *Asymptotically Optimal Depth Fermionic Permutation
> on 2D Grid Quantum Architecture without Ancillas*, submitted to PRX Quantum
> (2026). arXiv:2605.26041.

The paper gives a fermionic-permutation protocol for an $L\times L$
nearest-neighbor qubit grid with CNOT depth at most $10L+12$ for $N=L^2$
modes, using no ancillas, mid-circuit measurements, or feedforward. Its key
ingredient is a *folded* phase-polynomial circuit for the diagonal parity
correction $\Gamma$ of depth $2L+O(1)$. This repository builds every circuit
family the paper compares, contains the data behind every evaluation figure,
and regenerates those figures from that data.

## Circuit families

All implement the same unitary; they differ only in circuit structure.

| Method | CNOT depth | Ancillas | Module |
|---|---|---|---|
| 1D snake `FSWAP` network (baseline) | $2L^2$ | 0 | `common/fp_1d.py` |
| Row–Column–Row + ancilla $\Gamma$ (Jiang et al.) | $32L+8$ | $L$ | `common/gamma_ancilla.py` |
| Row–Column–Row + primitive ancilla-free $\Gamma$ (reference) | $30L+16$ | 0 | `common/gamma_primitive.py` |
| Row–Column–Row + pipelined ancilla-free $\Gamma$ (reference) | $22L+O(1)$ | 0 | `common/gamma_pipeline.py` |
| **Row–Column–Row + folded $\Gamma^\star$ (this paper)** | **$\le 10L+12$** | **0** | `common/gamma_folded.py` |

The folded $\Gamma^\star$ has per-application depth $2g(L)+8\le 2L+6$ for every
$L\ge 2$ (analytic family for $L\ge 7$; explicit certified tables for
$L=2,\dots,6$ in `common/gamma_small.py`). Every family is checked against the
full unitary in `common/tests/`.

## Layout

```
common/                 circuit constructions shared by all experiments
  grid.py               grid topology and snake Jordan–Wigner indexing
  fswap.py              the FSWAP gate
  hall_decomposition.py Hall's Row–Column–Row factorization
  oet_sort.py           odd–even transposition sorting networks
  gamma_*.py            the Gamma constructions listed above
  fp_1d.py, fp_2d.py    end-to-end 1D and 2D fermionic permutation circuits
  metrics.py            depth, gate count, spacetime volume, fidelity estimates
  noise_models.py       the circuit-level noise model
  stim_convert.py       Cirq -> Stim conversion and the process-fidelity estimator
exp1_fp/                Experiment 1: standalone fermionic permutation (Figs. 10, 11)
exp2_ffft/              Experiment 2: 2D fermionic fast Fourier transform (Fig. 12)
exp3_syk/               Experiment 3: sparse-SYK Trotter step (Fig. 13)
explorations/hilbert_bk_jw/   Hilbert-curve BK/Parity/JW conversion rounds (Fig. 9)
docs/                   gate accounting, experiment provenance, tree-gate dictionary
requirements.txt        exact package versions used for the reported data
```

Each `exp*/` directory holds `run_experiment.py` (the sweep), `results/data.csv`
(the data the paper plots, with seeds and library versions stamped on every
row), `plot*.py` (the figures), and `tests/`.

## Installation

Python 3.11 or later. The pins in `requirements.txt` are the versions the
reported data were produced with.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest common/tests exp1_fp/tests exp2_ffft/tests exp3_syk/tests -q
```

## Regenerating the paper's figures from the committed data

Run from the repository root.

```bash
export PYTHONPATH=$(pwd)           # Windows PowerShell: $env:PYTHONPATH = (Get-Location).Path
python -m exp1_fp.plot_paper --paper-fig-dir out --width column     # FP-exp1_depth_spacetime.pdf, FP-exp1_stim_fidelity.pdf
python -m exp2_ffft.plot_ft --output-dir exp2_ffft/figures          # FP-exp2_combined_ft.pdf
python -m exp3_syk.plot_ft  --output-dir exp3_syk/figures           # FP-exp3_combined_ft.pdf
python explorations/hilbert_bk_jw/hilbert_bk_jw.py                  # all_rounds_k6_N63.pdf and per-round SVGs
```

The Exp. 2 and Exp. 3 plot drivers validate complete publication coverage of
the CSV before drawing; do not pass `--allow-partial` for a paper figure.

## Regenerating the data

The full sweeps are deterministic given the recorded seeds. Write to a fresh
output directory; reusing `exp1_fp/results` or `exp3_syk/results` triggers
checkpoint resume.

```bash
python -m exp1_fp.run_experiment --workers 4 --output-dir out/exp1/results --fig-dir out/exp1/figures
python -m exp1_fp.validate_results out/exp1/results/data.csv
python -m exp2_ffft.run_experiment --output-dir out/exp2/results --fig-dir out/exp2/figures
python -m exp3_syk.run_experiment  --output-dir out/exp3/results --fig-dir out/exp3/figures
```

Experiment 1 draws $10^6$ Stim shots per noisy row and is the longest run
(roughly 7–8 hours on one worker, 2–3 hours on four). Experiments 2 and 3 take
tens of minutes. Settings, schemas, seeds, validators, and the mapping from
outputs to paper figures are recorded in
[`docs/publication_experiment_provenance.md`](docs/publication_experiment_provenance.md);
the CNOT-equivalent gate accounting is in
[`docs/ft_gate_accounting.md`](docs/ft_gate_accounting.md).

## Using the constructions directly

```python
from common.fp_1d import build_fp_1d, build_benchmark_permutation
from common.fp_2d import build_fp_2d, GammaMethod
from common.metrics import count_resources

L = 5
perm = build_benchmark_permutation(L, "reverse")
fp_1d = build_fp_1d(L, perm)                                   # 1D FSWAP network
fp_2d = build_fp_2d(L, perm, gamma_method=GammaMethod.FOLDED)  # this paper
for r in (fp_1d, fp_2d):                                       # FPResult: .circuit, .sys_qubits, .anc_qubits, ...
    print(count_resources(r.circuit, L, len(r.anc_qubits))["cnot_depth"])   # 50, then 40
```

## Citing

Please cite the paper (arXiv:2605.26041). The code and data archived for the
submitted version are on Zenodo, DOI [10.5281/zenodo.22646475](https://doi.org/10.5281/zenodo.22646475),
matching tag `v1.0-prxq`. See [`CITATION.cff`](CITATION.cff).

## License

MIT; see [`LICENSE`](LICENSE).
