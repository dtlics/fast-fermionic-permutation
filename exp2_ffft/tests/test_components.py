"""Tests for FFFT building blocks: twiddle, column FFTs, row FFTs."""

import numpy as np
import pytest
import cirq

from common.grid import make_system_qubits
from exp2_ffft.twiddle import build_twiddle_circuit
from exp2_ffft.col_ffft_bare import build_bare_column_fffts, build_row_fffts


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
