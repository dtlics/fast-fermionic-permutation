import pytest

import pennylane as qp

from common.scheduler import greedy_schedule


@pytest.mark.parametrize("circuit, expected", [
    (
        qp.tape.qscript.QuantumScript(
            [
                qp.CNOT([0, 1]),
                qp.H(0),
                qp.X(2),
                qp.CNOT([1, 2]),
                qp.H(0),
                qp.CNOT([2, 3]),
                qp.X(1),
                qp.CNOT([1, 2])
            ]
        ),
        [[qp.CNOT(wires=[0, 1]), qp.X(2)], [qp.H(0), qp.CNOT(wires=[1, 2])], [qp.H(0), qp.CNOT(wires=[2, 3]), qp.X(1)], [qp.CNOT(wires=[1, 2])]]
    ),
])
def test_gamma_primitive_depth(circuit, expected):
    """Verify that the scheduler works as expected."""

    moments, _ = greedy_schedule(circuit)
    assert moments == expected