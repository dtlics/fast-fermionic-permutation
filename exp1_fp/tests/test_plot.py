"""Regression tests for strict Experiment 1 plot provenance checks."""

import pandas as pd
import pytest

from common.fp_1d import build_benchmark_permutation
from exp1_fp.collect import collect_instance
from exp1_fp.plot import _validate_native_data


def _valid_plot_data():
    L = 2
    rows = collect_instance(
        L,
        build_benchmark_permutation(L, "reverse"),
        "reverse",
        0,
        p_values=(1e-3,),
        p_idle_factor=1.0,
        shots=0,
    )
    df = pd.DataFrame(rows)
    df["stim_shots"] = 1_000_000
    df["stim_fidelity"] = 0.5
    return df


def test_native_folded_data_pass_plot_validation():
    _validate_native_data(_valid_plot_data())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda df: df.drop(columns=["accounting_model"]),
        lambda df: df.drop(columns=["gamma_schedule_model"]),
        lambda df: df.assign(accounting_model="legacy"),
        lambda df: df.assign(hall_decomposition_model="legacy"),
        lambda df: df.assign(gamma_schedule_model="legacy"),
        lambda df: df.assign(sampling_model="unseeded"),
        lambda df: df.assign(stim_seed=df.stim_seed + 1),
        lambda df: df.assign(permutation_sha256="0" * 64),
        lambda df: df.assign(cirq_version="0.0"),
        lambda df: df[df.baseline != "folded"],
        lambda df: pd.concat([df, df.iloc[[0]]], ignore_index=True),
        lambda df: df.assign(p_idle=df.p_idle / 10),
        lambda df: df.assign(total_cnots=df.total_cnots + 1),
        lambda df: df.assign(total_idle_slots_layered=df.total_idle_slots_layered + 1),
        lambda df: df.assign(stim_shots=0),
    ],
)
def test_plot_validation_rejects_stale_or_inconsistent_data(mutate):
    with pytest.raises(ValueError):
        _validate_native_data(mutate(_valid_plot_data()))
