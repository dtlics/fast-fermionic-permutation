# Archive

Legacy code from the initial development phase. Preserved for reference.

## Files

- **FP.ipynb**: Original main notebook containing all code (grid utilities, Hall
  decomposition, Gamma constructions 1-3, FSWAP sort, circuit builders,
  verification, resource counting, and scaling plots). All code has been
  extracted into `common/`.

- **Gamma.ipynb**: Gamma operator verification notebook with constructions 1-3,
  depth analysis, and pipelined gamma verification.

- **pipelined_gamma.py**: Original standalone pipelined Gamma implementation.
  Migrated to `common/gamma_pipeline.py` with a bugfix to `pipeline_skip_only_ops`
  (forward pass `max_fwd` was too short by 1, causing bit corruption on the
  last column's skip gadget).

- **visualize_fermionic.py**: Circuit visualization and `CompressedFermionicPermutation`
  (old ancilla-based method with ancilla qubits between data columns). Contains
  `GridVisualizer` for drawing qubit grids and gates -- potentially useful for
  future circuit diagrams.

- **resource_estimation.py**: Old resource estimation comparing only 2 methods
  (baseline_openfermion_snake vs custom_ancilla_row_cnot). Contains useful
  gate-lowering code (FSWAP decomposition, SWAP/CZ lowering) that was partially
  extracted into `common/metrics.py` and `common/fswap.py`.

## Visualization Patterns (for future reference)

The `visualize_fermionic.py` module had:
- `GridVisualizer` class for drawing L x L qubit grids with matplotlib
- Gate synthesis showing FSWAP decomposition to 2 CNOTs
- CZ decomposition to H-CNOT-H
- Frame-by-frame circuit visualization (used to generate `visualization_demo_5x5/`)
- Color-coded stages (RowA, Col, RowB) in circuit diagrams
