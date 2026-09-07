# CNOT-equivalent gate accounting

The numerical experiments use one CNOT-equivalent two-qubit operation as
their gate-error and time unit. Under this convention,

- CNOT and CZ cost one CNOT-equivalent operation and one layer;
- FSWAP costs two CNOT-equivalent operations and two layers; and
- logical Pauli corrections are frame tracked and add no physical layer.

This is deliberately conservative for FSWAP-heavy circuits. Although FSWAP
may be compiled efficiently on an iSWAP-like device, the early
fault-tolerant comparison uses its exact two-CNOT synthesis (up to adjacent
one-qubit Clifford basis changes). Treating FSWAP as a single generic
two-qubit error source would therefore undercount both its gate-error
opportunities and its duration relative to a bare CNOT or CZ.

## Gate and non-entangling exponents

For a two-qubit moment \(m\), let

- \(Q\) be the number of data and explicit ancilla qubits;
- \(n_F(m)\) be its number of FSWAPs;
- \(n_C(m)\) be its number of bare CNOT/CZ gates;
- \(G_m = 2n_F(m)+n_C(m)\); and
- \(D_m=2\) if the moment contains an FSWAP, and \(D_m=1\) otherwise.

The non-entangling qubit-layer count is

\[
I_m = QD_m-2G_m.
\]

Thus, for the complete circuit,

\[
G=\sum_m G_m,\qquad D=\sum_m D_m,\qquad
I=\sum_m I_m=QD-2G.
\]

Here (D) is CNOT-equivalent entangling depth; serial one-qubit gates are not
added to it.  Accordingly, (S=QD) below is a logical-spacetime/CNOT-schedule proxy,
not a claim about the complete wall-clock duration of syndrome extraction and
magic-state delivery.

The mixed-moment case is important. If an FSWAP and a CNOT run concurrently,
both pairs are active in the first layer, but only the FSWAP pair is active
in the second. The completed CNOT pair is therefore idle in layer two. The
formula above accounts for this without treating every gate in the moment as
if it had FSWAP duration.

The multiplicative estimate is

\[
P_{\mathrm{nf}}=(1-p_{2q})^G(1-p_{\mathrm{idle}})^I,
\]

with `total_cnots` used for \(G\) and `total_idle_slots_layered` used for
\(I\). In the publication early-FT setting,
\(p_{2q}=p_{\mathrm{nonent}}=p\). The exponent \(I=QD-2G\) already includes
qubits undergoing one-qubit operations within the CNOT schedule, so the
explicit one-qubit inventories are not multiplied into the probability a
second time. They remain stored for auditing and, where applicable, for
separate magic-state accounting.

## Stim schedule

`common/stim_convert.py` implements the same convention by default. The exact
FSWAP unitary is emitted as `SWAP` followed by `CZ`. Its FSWAP pair receives
two independent `DEPOLARIZE2` channels. In the second layer, every non-FSWAP
qubit, including a bare CNOT/CZ pair from a mixed moment, receives
`DEPOLARIZE1(p_idle)`.

Passing `layered=False` reproduces the original per-Cirq-moment convention
for audit comparisons only.

## Stored columns

Experiments 1 and 3 use the following canonical columns:

| Quantity | Column |
|---|---|
| CNOT-equivalent two-qubit count \(G\) | `total_cnots` |
| CNOT-equivalent depth \(D\) | `cnot_depth` |
| layered non-entangling count \(I\) | `total_idle_slots_layered` |
| corrected independent-location no-fault estimate | `mult_fidelity` |

Experiment 2 keeps its established FFFT schema: `total_cnot_equiv` is \(G\),
`cnot_depth` is \(D\), and `total_idle_slots_layered` is \(I\). Its
`n_1q_nonz` and `p_1q` columns are compatibility/audit metadata and do not add
another exponent in the uniform model. Its audit-only
raw columns are `total_2q_gates` and `total_idle_slots`; the legacy probability
is `mult_fidelity_legacy`.

In Experiments 1 and 3, raw counts from the original convention instead carry
explicit `_legacy` suffixes: `total_2q_gates_legacy`,
`total_idle_slots_legacy`, and `mult_fidelity_legacy`. The historical
`mult_fidelity` column name denotes a no-modeled-fault product, not a state or
channel fidelity. Publication plotting requires the canonical columns and
rejects missing or provenance-incompatible accounting; legacy columns are
retained for audit comparisons only.

All three resource tables also carry `gamma_schedule_model`. Folded rows use
`folded_boundary_l3_l6_v1` at `L=3,4,5,6` and
`folded_zero_extra_ge7_v1` for the unchanged direct `L=2` circuit and analytic
`L>=7` schedule; non-folded rows use `not_applicable`. This is
circuit-schedule provenance rather than another gate count, and strict
resume/publication checks reject rows whose value is missing or incompatible
with their size.
