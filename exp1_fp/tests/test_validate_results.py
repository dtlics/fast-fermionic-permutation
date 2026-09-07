"""Regression tests for the exhaustive Experiment 1 result validator."""

import pandas as pd
import pytest

from common.fp_1d import build_benchmark_permutation
from common.gamma_folded import FOLDED_GAMMA_SCHEDULE_MODEL
from exp1_fp.collect import collect_instance
from exp1_fp.validate_results import validate_results


def _small_complete_result():
    L = 2
    return pd.DataFrame(
        collect_instance(
            L,
            build_benchmark_permutation(L, "reverse"),
            "reverse",
            0,
            p_values=(1e-3,),
            p_idle_factor=1.0,
            shots=10,
        )
    )


def test_exhaustive_validator_accepts_complete_small_sweep():
    summary = validate_results(
        _small_complete_result(),
        L_values=(2,),
        perm_kinds=("reverse",),
        n_random=0,
        p_values=(1e-3,),
        shots=10,
    )
    assert summary["rows"] == 5
    assert summary["fswap_cnot_equivalent_cost"] == 2
    assert summary["gamma_schedule_model_folded_by_L"] == {
        "2": FOLDED_GAMMA_SCHEDULE_MODEL
    }


def test_exhaustive_validator_rejects_duplicate_or_nonquantized_rows():
    df = _small_complete_result()
    with pytest.raises(ValueError):
        validate_results(
            pd.concat([df, df.iloc[[0]]], ignore_index=True),
            L_values=(2,), perm_kinds=("reverse",), n_random=0,
            p_values=(1e-3,), shots=10,
        )

    df.loc[0, "stim_fidelity"] = 0.123
    with pytest.raises(ValueError):
        validate_results(
            df,
            L_values=(2,), perm_kinds=("reverse",), n_random=0,
            p_values=(1e-3,), shots=10,
        )


def test_exhaustive_validator_rejects_reused_stim_seed():
    df = _small_complete_result()
    df.loc[1, "stim_seed"] = df.loc[0, "stim_seed"]

    with pytest.raises(ValueError):
        validate_results(
            df,
            L_values=(2,), perm_kinds=("reverse",), n_random=0,
            p_values=(1e-3,), shots=10,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda df: df.drop(columns=["gamma_schedule_model"]),
        lambda df: df.assign(gamma_schedule_model="stale"),
    ],
)
def test_exhaustive_validator_rejects_missing_or_stale_gamma_provenance(mutate):
    with pytest.raises(ValueError):
        validate_results(
            mutate(_small_complete_result()),
            L_values=(2,), perm_kinds=("reverse",), n_random=0,
            p_values=(1e-3,), shots=10,
        )
