"""Publication-table completeness and atomic-write guards for Experiment 2."""

import pandas as pd
import pytest

from common.hall_decomposition import HALL_DECOMPOSITION_MODEL
from common.gamma_folded import (
    FOLDED_GAMMA_SCHEDULE_MODEL,
    GAMMA_SCHEDULE_NOT_APPLICABLE,
    folded_gamma_schedule_model,
)
from exp2_ffft.collect import (
    ACCOUNTING_MODEL,
    HALL_NOT_APPLICABLE,
    _atomic_write_csv,
)
from exp2_ffft.plot import _validate_folded_provenance
from exp2_ffft.ft_accounting import ROTATION_INVENTORY_MODEL


def test_publication_validator_rejects_partial_coverage():
    rows = []
    for method, gamma, hall in (
        ("1d_baseline", None, HALL_NOT_APPLICABLE),
        ("gamma_2d_core", "folded", HALL_NOT_APPLICABLE),
        ("gamma_2d_ancilla", "ancilla", HALL_DECOMPOSITION_MODEL),
        ("gamma_2d_proper", "folded", HALL_DECOMPOSITION_MODEL),
    ):
        p_2q = 1e-3
        p_idle = p_2q
        rows.append({
            "L": 2, "N": 4, "method": method, "gamma_method": gamma,
            "gamma_schedule_model": (
                FOLDED_GAMMA_SCHEDULE_MODEL
                if gamma == "folded"
                else GAMMA_SCHEDULE_NOT_APPLICABLE
            ),
            "cirq_version": "1.7.0", "openfermion_version": "1.8.1",
            "numpy_version": "2.5.2", "pandas_version": "3.0.5",
            "matplotlib_version": "3.11.1", "networkx_version": "3.6.1",
            "accounting_model": ACCOUNTING_MODEL,
            "rotation_inventory_model": ROTATION_INVENTORY_MODEL,
            "hall_decomposition_model": hall,
            "n_ancillas": 2 if method == "gamma_2d_ancilla" else 0,
            "total_qubits": 6 if method == "gamma_2d_ancilla" else 4,
            "cnot_depth": 2,
            "total_cnot_equiv": 3,
            "total_idle_slots_layered": (
                (6 if method == "gamma_2d_ancilla" else 4) * 2 - 6
            ),
            "n_1q_nonz": 1,
            "spacetime_volume": (
                (6 if method == "gamma_2d_ancilla" else 4) * 2
            ),
            "verification_error": (
                0.0
                if method in {"1d_baseline", "gamma_2d_proper"}
                else float("nan")
            ),
            "n_exact_t": 8, "n_synth_rz": 0,
            "p_2q": p_2q, "p_idle": p_idle, "p_1q": p_2q,
            "mult_fidelity": (
                (1 - p_2q) ** 3
                * (1 - p_idle) ** (
                    (6 if method == "gamma_2d_ancilla" else 4) * 2 - 6
                )
            ),
        })
    partial = pd.DataFrame(rows)
    _validate_folded_provenance(partial)

    excessive = partial.copy()
    excessive.loc[
        excessive["method"] == "gamma_2d_proper", "verification_error"
    ] = 1e-6
    with pytest.raises(ValueError, match="verification error"):
        _validate_folded_provenance(excessive)

    with pytest.raises(ValueError, match="exact publication coverage"):
        _validate_folded_provenance(
            partial, require_publication_coverage=True,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda df: df.drop(columns=["gamma_schedule_model"]),
        lambda df: df.assign(
            gamma_schedule_model=df["gamma_schedule_model"].mask(
                df["gamma_method"] == "folded",
                FOLDED_GAMMA_SCHEDULE_MODEL,
            )
        ),
    ],
)
def test_publication_validator_rejects_stale_gamma_schedule(mutate):
    L = 4
    N = L * L
    rows = []
    for method, gamma, hall in (
        ("1d_baseline", None, HALL_NOT_APPLICABLE),
        ("gamma_2d_core", "folded", HALL_NOT_APPLICABLE),
        ("gamma_2d_ancilla", "ancilla", HALL_DECOMPOSITION_MODEL),
        ("gamma_2d_proper", "folded", HALL_DECOMPOSITION_MODEL),
    ):
        p_2q = 1e-3
        qubits = N + L if method == "gamma_2d_ancilla" else N
        idle = qubits * 2 - 6
        rows.append({
            "L": L, "N": N, "method": method, "gamma_method": gamma,
            "gamma_schedule_model": (
                folded_gamma_schedule_model(L)
                if gamma == "folded"
                else GAMMA_SCHEDULE_NOT_APPLICABLE
            ),
            "cirq_version": "1.7.0", "openfermion_version": "1.8.1",
            "numpy_version": "2.5.2", "pandas_version": "3.0.5",
            "matplotlib_version": "3.11.1", "networkx_version": "3.6.1",
            "accounting_model": ACCOUNTING_MODEL,
            "rotation_inventory_model": ROTATION_INVENTORY_MODEL,
            "hall_decomposition_model": hall,
            "n_ancillas": L if method == "gamma_2d_ancilla" else 0,
            "total_qubits": qubits, "cnot_depth": 2,
            "total_cnot_equiv": 3, "total_idle_slots_layered": idle,
            "n_1q_nonz": 1, "n_exact_t": 8, "n_synth_rz": 0,
            "verification_error": (
                0.0
                if method in {"1d_baseline", "gamma_2d_proper"}
                else float("nan")
            ),
            "spacetime_volume": qubits * 2,
            "p_2q": p_2q, "p_idle": p_2q, "p_1q": p_2q,
            "mult_fidelity": (
                (1 - p_2q) ** 3
                * (1 - p_2q) ** idle
            ),
        })

    with pytest.raises(ValueError, match="provenance|stale"):
        _validate_folded_provenance(mutate(pd.DataFrame(rows)))


def test_atomic_write_preserves_previous_csv_on_failure(tmp_path, monkeypatch):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("previous\n", encoding="utf-8")

    def interrupted_to_csv(self, stream, *args, **kwargs):
        stream.write("partial\n")
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(pd.DataFrame, "to_csv", interrupted_to_csv)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _atomic_write_csv(pd.DataFrame({"x": [1]}), str(csv_path))

    assert csv_path.read_text(encoding="utf-8") == "previous\n"
    assert not list(tmp_path.glob(".data.csv.*.tmp"))
