# Publication experiment provenance

This document records the exact numerical sweeps, provenance checks, figure
generators, and paper-asset mappings for the journal manuscript. Run every
command from the repository root in the local `.run-env` environment populated
from the exact versions in `requirements.txt`. The environment directory is
intentionally Git-ignored; recreate it from the pinned file rather than using
ambient site packages.

The Experiment 1 and 3 collectors support provenance-checked
checkpoint/resume, so a command aimed at an existing valid CSV can skip
already-complete work. Experiment 2 checkpoints after each size but currently
recollects all requested sizes. For an auditable clean rerun, always use a new
staging directory whose `data.csv` does not exist. Do not use
`--allow-partial` for publication artifacts.

## Environment and accounting contract

The reported artifacts use Cirq 1.7.0, OpenFermion 1.8.1, Stim 1.16.0,
NumPy 2.5.2, pandas 3.0.5, Matplotlib 3.11.1, and NetworkX 3.6.1. The complete
set of pinned dependencies is in `requirements.txt`.

The development rerun environment uses Python 3.12.13. On Windows, create and
select it with:

```powershell
python -m venv .run-env
$py = ".\.run-env\Scripts\python.exe"
& $py -m pip install -r requirements.txt
& $py --version  # must report Python 3.12.13 for an exact rerun
```

All commands below deliberately invoke `$py`; this prevents an activated or
system Python from silently changing dependency versions.

Across all experiments, an FSWAP is charged as two CNOT-equivalent gates and
two CNOT-equivalent layers. Layered idle slots satisfy

```text
I = Q D - 2 G,
```

where `Q` is the total number of system and ancilla qubits, `D` is
CNOT-equivalent entangling depth (not a sum of serialized one-qubit layers),
and `G` is the total CNOT-equivalent gate count. The publication no-fault
model uses the same logical rate for entangling and non-entangling locations.
Because `I` includes one-qubit-active slots, explicit one-qubit inventories
are not charged again. Phase-frame operations are free; non-Clifford
magic-state costs are stated below.

Every resource row also records `gamma_schedule_model`. Folded-Gamma rows use
branch-specific provenance: `folded_zero_extra_ge7_v1` for the unchanged
closed-form `L>=7` schedule (and the unchanged direct `L=2` circuit), and
`folded_boundary_l3_l6_v1` for the promoted record-depth schedules at
`L=3,4,5,6`. All other rows record `not_applicable`. Resume and publication
validation reject a missing or size-incompatible value, so the affected
boundary rows must be regenerated without invalidating unchanged asymptotic
data.

## Clean staged rerun

The following PowerShell commands use a fresh directory and make every
publication sweep explicit.

```powershell
$stage = "tmp/publication-rerun-gamma-zero-extra-v1"
if (Test-Path -LiteralPath $stage) {
  throw "Refusing to reuse existing staging directory: $stage"
}
$py = ".\.run-env\Scripts\python.exe"

# Experiment 1: 3,740 rows and one million Stim shots per noisy row.
& $py -m exp1_fp.run_experiment `
  --L 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 `
  --perms reverse transpose random `
  --n-random 20 `
  --p-values 0.0001 0.00001 `
  --p-idle-factor 1 `
  --shots 1000000 `
  --workers 4 `
  --output-dir "$stage/exp1/results" `
  --fig-dir "$stage/exp1/figures"
& $py -m exp1_fp.validate_results "$stage/exp1/results/data.csv"
& $py -m exp1_fp.summarize_results "$stage/exp1/results/data.csv"

# Experiment 2: 204 rows; the canonical driver makes the two-panel early-FT
# figure used in the paper.
& $py -m exp2_ffft.run_experiment `
  --L 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 `
  --output-dir "$stage/exp2/results" `
  --fig-dir "$stage/exp2/figures"

# Experiment 3: 3,060 resource rows plus 680 colors-only rows. The driver
# makes the four standard SVGs and the two-panel FT PDF/SVG.
& $py -m exp3_syk.run_experiment `
  --L 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 `
  --k 1 `
  --n-instances 10 `
  --p-values 0.001 0.0001 0.00001 `
  --p-idle-factor 1 `
  --colors-k 0.5 1 2 3 `
  --output-dir "$stage/exp3/results" `
  --fig-dir "$stage/exp3/figures"
```

Experiment 1 parallelizes independent permutations. Four workers is a
conservative setting for the 24-logical-CPU development machine because the
largest Cirq/Stim circuits are memory intensive. Worker count does not change
row ordering or seeded Stim results. Experiment 3 derives every random
instance from its recorded `instance_idx`; neither experiment depends on an
unseeded stochastic input. Experiments 2 and 3 are sequential.

## Validated plot-only rerun

After collection, these commands regenerate every staged experiment figure
without repeating the numerical sweeps. Experiment 1's exhaustive validator
is explicit; the Experiment 2 and 3 plot drivers enforce their own complete
publication schemas and coverage.

```powershell
& $py -m exp1_fp.validate_results "$stage/exp1/results/data.csv"
& $py -m exp1_fp.run_experiment --plot-only `
  --output-dir "$stage/exp1/results" --fig-dir "$stage/exp1/figures"
& $py -m exp1_fp.compose_paper_figures `
  --fig-dir "$stage/exp1/figures" --paper-fig-dir "paper/figures"

& $py -m exp2_ffft.run_experiment --plot-only `
  --output-dir "$stage/exp2/results" --fig-dir "$stage/exp2/figures"

& $py -m exp3_syk.run_experiment --plot-only `
  --output-dir "$stage/exp3/results" --fig-dir "$stage/exp3/figures"
```

## Paper-asset mapping

The manuscript checkout in `paper/` is a separate nested Git repository. It
must already exist. Generate or copy the paper-facing PDFs only after the
staged CSVs pass strict validation and the rendered figures pass visual QA.

| Experiment source | Paper destination | Generator or operation |
|---|---|---|
| `exp1/figures/depth_vs_L.pdf` + `spacetime_vs_N.pdf` | `paper/figures/FP-exp1_depth_spacetime.pdf` | `exp1_fp.compose_paper_figures` stacks the two vector pages |
| `exp1/figures/stim_fidelity.pdf` | `paper/figures/FP-exp1_stim_fidelity.pdf` | `exp1_fp.compose_paper_figures` copies it atomically |
| `exp2/figures/FP-exp2_combined_ft.pdf` | `paper/figures/FP-exp2_combined_ft.pdf` | validated file copy |
| `exp3/figures/FP-exp3_combined_ft.pdf` | `paper/figures/FP-exp3_combined_ft.pdf` | validated file copy |

With `$stage` from the clean-rerun commands:

```powershell
& $py -m exp1_fp.compose_paper_figures `
  --fig-dir "$stage/exp1/figures" `
  --paper-fig-dir "paper/figures"

& $py -c "from pathlib import Path; import shutil; s=Path(r'$stage/exp2/figures/FP-exp2_combined_ft.pdf'); d=Path('paper/figures/FP-exp2_combined_ft.pdf'); d.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(s,d)"

& $py -c "from pathlib import Path; import shutil; s=Path(r'$stage/exp3/figures/FP-exp3_combined_ft.pdf'); d=Path('paper/figures/FP-exp3_combined_ft.pdf'); d.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(s,d)"
```

The currently referenced experiment PDFs in `paper/main.tex` are the two
Experiment 1 files and the combined early-FT files for Experiments 2 and 3.
The standalone Experiment 2 depth/no-fault files and the older combined files
are legacy, unreferenced assets.

## Experiment 1: fermionic permutations

### Sweep and seeds

- Grid sides: every integer `L` from 4 through 20; `N=L^2`.
- Permutation families: reversal, transpose (labeled 2D reflection), and 20
  random permutations per size.
- Random permutation seed: `perm_idx`, from 0 through 19. Structured families
  record `-1` as the deterministic seed sentinel.
- Baselines: `1d`, `ancilla`, `primitive`, `pipelined`, and `folded`.
- Noise rates: `p_2q` in `{1e-4,1e-5}` and
  `p_idle=p_2q`.
- Stim samples: 1,000,000 per `(L, permutation, baseline, p_2q)` row.
- Stim statistic: the process fidelity `F_e = Pr(final data Pauli = I)`,
  estimated by propagating the sampled Pauli errors through the noisy forward
  circuit with `stim.FlipSimulator` (`disable_stabilizer_randomization=True`)
  and counting a shot as a success when both the X and the Z flip masks vanish
  on every data qubit; errors confined to working ancillas are ignored after
  the ideal circuit is checked to disentangle them. Each row also carries a
  Wilson 95% interval (`stim_fidelity_ci_lo/hi`).
- Stim seed: a stable 63-bit SHA-256 derivation of the full scientific row
  key, including accounting, Hall-decomposition, and Gamma-schedule models.
  The corresponding sampling provenance is
  `stim_flipsim_process_fidelity_sha256_row_key_v3` (v2 was the all-zero
  return probability after a noiseless inverse, which is blind to phase-only
  faults).

The exact coverage is
`17 sizes x 22 permutations x 5 baselines x 2 rates = 3,740 rows`.
`exp1_fp.validate_results` verifies exact coverage, unique keys and Stim seeds,
dependency and permutation fingerprints, FSWAP accounting, the layered-idle
identity, no-fault products, and shot-count quantization.

The 38-column schema is:

```text
L,N,perm_kind,perm_idx,baseline,accounting_model,
hall_decomposition_model,gamma_schedule_model,sampling_model,stim_seed,
permutation_model,permutation_seed,permutation_sha256,numpy_version,
cirq_version,openfermion_version,stim_version,n_ancillas,total_qubits,
cnot_depth,total_cnots,n_fswap,n_2q_cnot_cz,total_idle_slots_layered,
total_2q_gates_legacy,total_idle_slots_legacy,n_1q_hadamard,n_1q_other,
n_1q_s_sdag,n_1q_rz,n_1q_pauli_z,spacetime_volume,p_2q,p_idle,
mult_fidelity,mult_fidelity_legacy,stim_fidelity,stim_shots
```

### Figures

`exp1_fp.run_experiment --plot-only` validates native accounting and generates
`depth_vs_L`, `spacetime_vs_N`, and `stim_fidelity` as PDF and SVG. The first
two use the smallest stored noise rate only to select one copy of
noise-independent resource counts. Random depth and volume curves show mean
and sample standard deviation over 20 instances. The Stim grid orders rates
as `1e-5`, `1e-4`; zero-return points are omitted on the logarithmic
axis, and random curves show means without error bars.

`exp1_fp.compose_paper_figures` is now the source of truth for paper layout.
It stacks depth above spacetime with 30/8-point side margins, 24/8-point
vertical margins, a 30-point row gap, and 14-point descriptive row labels.

### Runtime and repeatability

A historical four-baseline, one-worker run took 5.91 hours. The current
five-baseline sweep is expected to take roughly 7-8 hours with one worker or
2-3 hours with four workers, depending on memory pressure. With the pinned
runtime, serial and parallel schedules produce identical seeded row values.
The new folded schedule changes resource values, and the `v2` seed contract
changes Stim seeds for every row. The pre-scheduler CSV is intentionally
incompatible and must not be backfilled. Matplotlib PDF/SVG files also contain
timestamps and randomized SVG identifiers; compare validated canonicalized
data and rendered appearance, not file hashes.

## Experiment 2: fermionic FFT

### Sweep and schema

- Grid sides: every integer `L` from 4 through 20, for 17 sizes.
- Methods: CT-FFFT (`1d_baseline`), relaxed-layout arithmetic core
  (`gamma_2d_core`), mode-preserving ancilla-Gamma
  (`gamma_2d_ancilla`), and mode-preserving folded-Gamma proper
  (`gamma_2d_proper`).
- The arithmetic core remains in the table for audit but is not an equivalent
  end-to-end transform and is excluded from the early-FT comparison.
- Noise rates: `p_2q` in `{1e-3,1e-4,1e-5}` and
  `p_idle=p_1q=p_2q`. The `n_1q_nonz` inventory is retained, but these gates
  are already covered by the non-entangling exponent `I=QD-2G` and are not
  charged a second time.
- Small system-only transforms are numerically verified in complex128 where
  `N<=16`; publication validation requires a Frobenius residual at most
  `1e-10`. Dedicated tests cover the ancilla construction.

The exact new coverage is
`17 sizes x 4 methods x 3 rates = 204 rows`. The expected 31-column schema
after the clean rerun is:

```text
L,N,method,gamma_method,gamma_schedule_model,cirq_version,openfermion_version,numpy_version,
pandas_version,matplotlib_version,networkx_version,accounting_model,
rotation_inventory_model,hall_decomposition_model,n_ancillas,total_qubits,
cnot_depth,total_2q_gates,total_cnot_equiv,total_idle_slots,
total_idle_slots_layered,n_1q_nonz,n_exact_t,n_synth_rz,
spacetime_volume,p_2q,p_idle,p_1q,mult_fidelity,
mult_fidelity_legacy,verification_error
```

`exp2_ffft.run_experiment --plot-only` is the strict publication validator and
the canonical FT-figure driver. It requires the full grid
and all four methods, the current folded-Gamma schedule, folded/ancilla Gamma
provenance, the current Hall and rotation inventory models, exact dependency
versions, `Q=N+n_ancillas`, FSWAP and idle identities, and consistent no-fault
estimates.

### Early-FT figure

The canonical driver generates `FP-exp2_combined_ft` as PDF and SVG with the
publication defaults. The older standalone depth and no-fault plots remain
available from `exp2_ffft.plot` for diagnostic use but are not paper assets.
The compatibility inventory `n_1q_nonz` is not a second error exponent.

The early-FT figure applies the same model as Experiment 3 with
`alpha=1`, synthesis cost `c_T=70`, surface-code distance `d=11`, physical
error `p_phys=1e-3`, threshold `p_th=1e-2`, and logical-error prefactor `0.03`.
The FFFT inventory counts each F0 as
two exact T-angle rotations. Each general Givens gate contributes two
commuting Pauli rotations that are classified by angle, while each diagonal
twiddle is classified as Clifford-free, exact T/T-dagger, or arbitrary-angle
synthesis. Thus it records exact states separately from synthesized
rotations:

```text
n_T = n_exact_t + 70 n_synth_rz
S = Q D
V_T = n_T
F_FT = (1-p_B)^(S+n_T+V_T)
p_cyc = 0.03 (1e-3 / 1e-2)^((11+1)/2) = 3e-8
p_B = 1-(1-p_cyc)^11 = 3.30e-7
```

Here `p_cyc` is the logical error per surface-code cycle and one logical
block contains `d=11` cycles. The exponent charges one state-consumption block
and one preparation-service block per T state. It omits a separate cultivated
state-output factor so the factory failures represented by `V_T` are not
counted twice. Both FT figures use a 3:1
resource-to-fidelity-panel ratio. In Experiment 2, the main resource axis shows
`S=QD`; a logarithmic inset shows the raw magic inventory `n_T` for all
three methods, and the percentage annotations compare the plotted `S`. Fidelity values
below `1e-2`
are replaced pointwise by NaN for
display, so a line stops rather than acquiring a misleading near-zero tail;
later values above the floor remain visible.

The two panels compare only the equivalent, mode-preserving CT-FFFT,
ancilla-Gamma, and folded-Gamma circuits. A full run historically takes on the
order of 10-20 minutes on the development machine; the added ancilla method
may move it toward the upper end of that range. The pre-scheduler CSV is stale;
after a fresh collection, repeated clean runs in the same environment should
be byte-identical. Figure bytes are not expected to match for the same
Matplotlib metadata reasons as Experiment 1.

The magic series is intentionally not smoothed. OpenFermion uses
a factorization-dependent line-FFFT decomposition and falls back to a dense Givens
decomposition when a remaining factor is prime. The resulting synthesized
rotation inventory is therefore compiler- and factorization-dependent and can
decrease at a larger, more FFT-friendly `L`. This is a property of the literal
compiled circuit, not a Gamma-dependent cost or a random-instance fluctuation.

## Experiment 3: sparse SYK

### Sweep and seeds

- Grid sides: every integer `L` from 4 through 20.
- Resource table: `k=1`, 10 instances, six baselines
  (`naive_pauli`, `1d`, `ancilla`, `primitive`, `pipelined`, `folded`), and
  three stored noise rates.
- Colors table: `k` in `{0.5,1,2,3}` and the same 10 instances.
- `instance_idx` from 0 through 9 is the NumPy generator seed at every
  `(L,k)` pair. Quartet and coloring SHA-256 fingerprints are stored.

The resource table has
`17 sizes x 10 instances x 6 baselines x 3 rates = 3,060 rows` and 43
columns. The colors table has `17 x 4 x 10 = 680 rows` and 15 columns.
`generate_all_plots` validates exact publication coverage, dependency,
generator, coloring, Hall, Gamma-schedule, and Pauli-compiler provenance;
FSWAP, depth-split, idle, and spacetime identities; noise-independent
resources (including `n_1q_rz`); and uniform non-entangling-location
no-fault estimates.

The resource-table schema is:

```text
L,N,k,instance_idx,baseline,accounting_model,hall_decomposition_model,
gamma_schedule_model,
rng_seed,syk_generator_model,quartet_sampling_model,coupling_model,
parent_coloring_model,numpy_version,networkx_version,quartets_sha256,
coloring_sha256,cirq_version,openfermion_version,
pauli_rotation_compiler_model,n_colors,n_quartets,
n_ancillas,total_qubits,cnot_depth,fp_cnot_depth,interaction_cnot_depth,
total_cnots,n_fswap,n_2q_cnot_cz,total_idle_slots_layered,
total_2q_gates_legacy,total_idle_slots_legacy,n_1q_hadamard,n_1q_other,
n_1q_s_sdag,n_1q_rz,n_1q_pauli_z,spacetime_volume,p_2q,p_idle,
mult_fidelity,mult_fidelity_legacy
```

The colors-only schema is:

```text
L,N,k,instance_idx,n_colors,n_quartets,rng_seed,syk_generator_model,
quartet_sampling_model,coupling_model,parent_coloring_model,numpy_version,
networkx_version,quartets_sha256,coloring_sha256
```

### Nearest-neighbor Pauli compiler and validation

Every resource row records
`pauli_rotation_compiler_model=snake_nn_dirty_bridge_sweep_v1`. The compiler
orders support on the physical Jordan--Wigner snake and reduces its parity by
an exact nearest-neighbor dirty-bridge sweep. If a string has `s` supported
sites and `b` unsupported sites inside its minimal snake interval, compute and
uncompute use `2(s-1+2b)` CNOTs. The bridge qubits may hold arbitrary states
and are restored exactly.

The publication collection path rejects a generated circuit unless every
two-qubit operation acts on Manhattan-adjacent square-grid qubits. Before
moment-wise packing of same-color rotations, it also requires each compiled
footprint to equal the complete snake interval between its support endpoints
and requires those full intervals, not only the quartet supports, to be
pairwise disjoint. These circuit-level guards run before a checkpoint is
written.

Checkpoint/resume accepts an instance only when it has complete baseline and
noise-rate coverage and the current Gamma-schedule and Pauli-compiler
provenance; rows from either former implementation are therefore discarded
and recomputed. Resume and publication plotting both enforce
`D=fp_cnot_depth+interaction_cnot_depth`, noise-invariance of all resource
fields including `n_1q_rz`, the FSWAP and layered-idle identities, and the
uniform no-fault value implied by the recorded counts. Resume checks
`p_idle=p_2q*p_idle_factor` against the factor requested for that run; the
publication validator fixes the reported setting to `p_idle=p_2q`.

### Figures and FT model

The standard plot driver writes SVG files for spacetime volume, the
hardware-oriented no-fault estimate, depth breakdown, and coloring. The paper
uses `FP-exp3_combined_ft.pdf`, generated as PDF and SVG by
`exp3_syk.plot_ft`.

For `k=1`, the FT loader selects one copy of the noise-independent resources,
aggregates mean and sample standard deviation over ten instances, and uses

```text
n_T = 70 n_Rz
S = Q D
V_T = n_T
F_FT = (1-p_B)^(S+n_T+V_T),
p_B = 1-(1-3e-8)^11 = 3.30e-7.
```

Panel (a) shows logical spacetime volume `S`; its annotated percentages compare
`S` for the folded method and 1D baseline. The inset shows raw `n_T` for all three
routing methods, whose inventories coincide because their interaction workload
is common. Panel (b) also
includes the naive-Pauli baseline. The historical 1.47-hour five-baseline
timing predates the exact nearest-neighbor Pauli compiler and is not a runtime
estimate for the current six-baseline sweep. The seeded structures and
resource counts reproduce under the pinned
NumPy/NetworkX/Cirq/OpenFermion environment and sequential row order. A clean
rerun can differ from the prior official CSV by last-bit float serialization
in the no-fault columns (observed maximum relative difference
`6.93e-16`), so validate schemas, provenance, exact integer resources, and
floating values at numerical tolerance rather than requiring a file hash.
Experiment 2 should be accepted only after the clean four-method sweep passes
the schema, provenance, exact-resource, and numerical-tolerance validators;
no byte-level CSV hash is treated as authoritative.

## Historical visual lineage

- The earliest Experiment 1 composites were manually assembled in a
  diagrams.net file (kept with the manuscript sources) from Matplotlib 3.9.4 SVGs. The diagrams.net file
  added descriptive row headers. The current pypdf composer captures the
  plot sizes and vector stacking and restores the descriptive row headers.
- The current Exp1 plotting style incorporates the later large-font,
  compact-grid branch changes. Older `results_1M` and branch figures are not
  canonical because their baseline sets, labels, and sampling code differ.
- Visual QA should specifically check the dense Experiment 1 Stim legend: the
  pre-rerun paper PDF has a folded-Gamma label intruding into a neighboring
  panel. This is an observed artifact, not a hidden generation parameter to
  preserve.
- The MICRO-era Exp3 FT plot used a 3.15:1 panel ratio, pipelined Gamma, a
  different magic-volume presentation, a logarithmic fidelity axis, and curve
  truncation. The current additive-volume, folded-Gamma, 3:1 design is
  fully specified by `exp3_syk.plot_ft` and supersedes it.
- The new Experiment 2 FT design is fully specified by
  `exp2_ffft.ft_accounting` and `exp2_ffft.plot_ft`. No external branch or
  diagrams.net configuration is required.

## Promotion policy

Experiment 1 provides `exp1_fp.promote_publication`, which validates a staged
CSV and atomically replaces its six official standalone figures before
committing the official CSV last. Experiments 2 and 3 currently have no
equivalent atomic promoter. Keep their staged tables and figures together,
validate and render them, then copy the complete accepted artifact set. Never
promote a reduced `--allow-partial` smoke run.
