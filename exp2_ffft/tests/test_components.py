"""Tests for FFFT building blocks: twiddle, column FFTs, row FFTs."""

import numpy as np
import pytest
import cirq

from common.grid import make_system_qubits
from common.fp_2d import GammaMethod
from common.gamma_folded import (
    FOLDED_GAMMA_SCHEDULE_MODEL,
    GAMMA_SCHEDULE_NOT_APPLICABLE,
)
from common.hall_decomposition import HALL_DECOMPOSITION_MODEL
from exp2_ffft.collect import HALL_NOT_APPLICABLE
from exp2_ffft.twiddle import build_twiddle_circuit
from exp2_ffft.col_ffft_bare import build_bare_column_fffts, build_row_fffts
from exp2_ffft.collect import ACCOUNTING_MODEL, collect_for_L, count_cnot_resources
from exp2_ffft.plot import _validate_folded_provenance
from exp2_ffft.ft_accounting import ROTATION_INVENTORY_MODEL


# ---------------------------------------------------------------------------
# Twiddle tests
# ---------------------------------------------------------------------------

class TestTwiddle:
    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_twiddle_phases(self, L):
        """Twiddle applies correct ZPowGate exponents."""
        sq = make_system_qubits(L)
        circ = build_twiddle_circuit(L, sq)
        N = L * L

        # Collect all ZPowGate exponents by qubit position
        exponents = {}
        for moment in circ:
            for op in moment:
                assert len(op.qubits) == 1, "Twiddle should only have 1q gates"
                q = op.qubits[0]
                r, c = q.row, q.col
                assert isinstance(op.gate, cirq.ZPowGate)
                exponents[(r, c)] = op.gate.exponent

        # Check values
        for r in range(L):
            for c in range(L):
                expected = -2.0 * c * r / N
                if c * r == 0:
                    assert (r, c) not in exponents, \
                        f"Identity twiddle at ({r},{c}) should be skipped"
                else:
                    assert (r, c) in exponents, \
                        f"Missing twiddle at ({r},{c})"
                    assert abs(exponents[(r, c)] - expected) < 1e-14, \
                        f"Wrong exponent at ({r},{c})"

    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_twiddle_depth(self, L):
        """Twiddle circuit has depth 1 (all single-qubit, one moment)."""
        sq = make_system_qubits(L)
        circ = build_twiddle_circuit(L, sq)
        # All single-qubit gates can be packed into 1 moment
        assert len(circ) <= 1


# ---------------------------------------------------------------------------
# Column FFFT tests
# ---------------------------------------------------------------------------

class TestColumnFFT:
    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_nn_compliance(self, L):
        """All 2-qubit gates in bare column FFTs are between vertical NN."""
        sq = make_system_qubits(L)
        circ = build_bare_column_fffts(L, sq)
        # Decompose composite gates (SwapPermutationGate etc) to 2q primitives
        decomposed = cirq.Circuit(cirq.decompose(circ))

        for moment in decomposed:
            for op in moment:
                if len(op.qubits) == 2:
                    q1, q2 = op.qubits
                    dist = abs(q1.row - q2.row) + abs(q1.col - q2.col)
                    assert dist == 1, (
                        f"Non-NN gate between {q1} and {q2} "
                        f"(dist={dist})"
                    )

    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_columns_same_column(self, L):
        """Each 2-qubit gate acts within a single column."""
        sq = make_system_qubits(L)
        circ = build_bare_column_fffts(L, sq)

        for moment in circ:
            for op in moment:
                if len(op.qubits) >= 2:
                    cols = set(q.col for q in op.qubits)
                    assert len(cols) == 1, (
                        f"Gate spans columns {cols}: {op}"
                    )

    @pytest.mark.parametrize("L", [2, 4, 8, 16])
    def test_power_of_two_line_depth_under_paper_metric(self, L):
        """Rows and columns realize the audited OpenFermion line depth."""
        sq = make_system_qubits(L)
        column = build_bare_column_fffts(L, sq)
        row = build_row_fffts(L, sq)
        expected = 2 if L == 2 else 5 * L - 2 * int(np.log2(L)) - 4
        assert count_cnot_resources(column, L, 0)["cnot_depth"] == expected
        assert count_cnot_resources(row, L, 0)["cnot_depth"] == expected


# ---------------------------------------------------------------------------
# Row FFFT tests
# ---------------------------------------------------------------------------

class TestRowFFT:
    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_nn_compliance(self, L):
        """All 2-qubit gates in row FFTs are between horizontal NN."""
        sq = make_system_qubits(L)
        circ = build_row_fffts(L, sq)
        decomposed = cirq.Circuit(cirq.decompose(circ))

        for moment in decomposed:
            for op in moment:
                if len(op.qubits) == 2:
                    q1, q2 = op.qubits
                    dist = abs(q1.row - q2.row) + abs(q1.col - q2.col)
                    assert dist == 1, (
                        f"Non-NN gate between {q1} and {q2}"
                    )

    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_rows_same_row(self, L):
        """Each 2-qubit gate acts within a single row."""
        sq = make_system_qubits(L)
        circ = build_row_fffts(L, sq)

        for moment in circ:
            for op in moment:
                if len(op.qubits) >= 2:
                    rows = set(q.row for q in op.qubits)
                    assert len(rows) == 1, (
                        f"Gate spans rows {rows}: {op}"
                    )


# ---------------------------------------------------------------------------
# Benchmark provenance guards
# ---------------------------------------------------------------------------

class TestBenchmarkProvenance:
    def test_collection_rejects_ancilla_method(self):
        with pytest.raises(ValueError, match="ancilla-Gamma proper comparator is collected automatically"):
            collect_for_L(
                2,
                p_values=(1e-3,),
                p_idle_factor=1.0,
                verify=False,
                gamma_method=GammaMethod.ANCILLA,
            )

    def test_plotting_rejects_legacy_data(self):
        import pandas as pd

        legacy = pd.DataFrame({
            "method": ["gamma_2d_proper"],
            "cnot_depth": [190],
        })
        with pytest.raises(ValueError, match="provenance columns"):
            _validate_folded_provenance(legacy)

    def test_folded_provenance_is_accepted(self):
        import pandas as pd

        p_2q = np.array([1e-3] * 4)
        p_idle = p_2q.copy()
        p_1q = p_2q.copy()
        total_qubits = np.array([4, 4, 6, 4])
        cnot_depth = np.array([6, 5, 7, 8])
        total_cnot_equiv = np.array([10, 8, 10, 12])
        total_idle = total_qubits * cnot_depth - 2 * total_cnot_equiv
        n_1q_nonz = np.array([2, 3, 3, 4])
        fidelity = (
            (1 - p_2q) ** total_cnot_equiv
            * (1 - p_idle) ** total_idle
        )
        current = pd.DataFrame({
            "L": [2, 2, 2, 2],
            "N": [4, 4, 4, 4],
            "method": [
                "1d_baseline", "gamma_2d_core", "gamma_2d_ancilla",
                "gamma_2d_proper",
            ],
            "gamma_method": [None, "folded", "ancilla", "folded"],
            "gamma_schedule_model": [
                GAMMA_SCHEDULE_NOT_APPLICABLE,
                FOLDED_GAMMA_SCHEDULE_MODEL,
                GAMMA_SCHEDULE_NOT_APPLICABLE,
                FOLDED_GAMMA_SCHEDULE_MODEL,
            ],
            "cirq_version": ["1.7.0"] * 4,
            "openfermion_version": ["1.8.1"] * 4,
            "numpy_version": ["2.5.2"] * 4,
            "pandas_version": ["3.0.5"] * 4,
            "matplotlib_version": ["3.11.1"] * 4,
            "networkx_version": ["3.6.1"] * 4,
            "accounting_model": [ACCOUNTING_MODEL] * 4,
            "rotation_inventory_model": [ROTATION_INVENTORY_MODEL] * 4,
            "hall_decomposition_model": [
                HALL_NOT_APPLICABLE,
                HALL_NOT_APPLICABLE,
                HALL_DECOMPOSITION_MODEL,
                HALL_DECOMPOSITION_MODEL,
            ],
            "n_ancillas": [0, 0, 2, 0],
            "total_qubits": total_qubits,
            "cnot_depth": cnot_depth,
            "total_cnot_equiv": total_cnot_equiv,
            "total_idle_slots_layered": total_idle,
            "n_1q_nonz": n_1q_nonz,
            "n_exact_t": [8, 8, 8, 8],
            "n_synth_rz": [0, 0, 0, 0],
            "spacetime_volume": total_qubits * cnot_depth,
            "p_2q": p_2q,
            "p_idle": p_idle,
            "p_1q": p_1q,
            "mult_fidelity": fidelity,
            "verification_error": [0.0, np.nan, np.nan, 0.0],
        })
        _validate_folded_provenance(current)

    def test_plotting_rejects_stale_raw_gate_accounting(self):
        import pandas as pd

        stale = pd.DataFrame({
            "L": [2, 2, 2, 2],
            "N": [4, 4, 4, 4],
            "method": [
                "1d_baseline", "gamma_2d_core", "gamma_2d_ancilla",
                "gamma_2d_proper",
            ],
            "gamma_method": [None, "folded", "ancilla", "folded"],
            "gamma_schedule_model": [
                GAMMA_SCHEDULE_NOT_APPLICABLE,
                FOLDED_GAMMA_SCHEDULE_MODEL,
                GAMMA_SCHEDULE_NOT_APPLICABLE,
                FOLDED_GAMMA_SCHEDULE_MODEL,
            ],
            "cirq_version": ["1.7.0"] * 4,
            "openfermion_version": ["1.8.1"] * 4,
            "numpy_version": ["2.5.2"] * 4,
            "pandas_version": ["3.0.5"] * 4,
            "matplotlib_version": ["3.11.1"] * 4,
            "networkx_version": ["3.6.1"] * 4,
            "accounting_model": ["raw_two_qubit_gate_count"] * 4,
            "rotation_inventory_model": [ROTATION_INVENTORY_MODEL] * 4,
            "hall_decomposition_model": [
                HALL_NOT_APPLICABLE,
                HALL_NOT_APPLICABLE,
                HALL_DECOMPOSITION_MODEL,
                HALL_DECOMPOSITION_MODEL,
            ],
            "n_ancillas": [0, 0, 2, 0],
            "total_qubits": [4, 4, 6, 4],
            "cnot_depth": [6, 5, 7, 8],
            "total_cnot_equiv": [10, 8, 10, 12],
            "total_idle_slots_layered": [4, 4, 22, 8],
            "n_1q_nonz": [2, 3, 3, 4],
            "n_exact_t": [8, 8, 8, 8],
            "n_synth_rz": [0, 0, 0, 0],
            "spacetime_volume": [24, 20, 42, 32],
            "p_2q": [1e-3] * 4,
            "p_idle": [1e-4] * 4,
            "p_1q": [1e-4] * 4,
            "mult_fidelity": [0.98, 0.98, 0.97, 0.97],
            "verification_error": [0.0, np.nan, np.nan, 0.0],
        })
        with pytest.raises(ValueError, match="CNOT-equivalent layered"):
            _validate_folded_provenance(stale)
