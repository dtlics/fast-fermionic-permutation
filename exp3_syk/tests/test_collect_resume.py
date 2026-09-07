"""Regression tests for Experiment 3 CSV provenance-aware resume logic."""

from types import SimpleNamespace

import pandas as pd
import pytest

from exp3_syk import collect
from exp3_syk.collect import (
    ACCOUNTING_MODEL,
    BASELINE_CONFIGS,
    _gamma_schedule_model_for_baseline,
    _hall_model_for_baseline,
    _validated_instance_resume_rows,
    _validated_resume_rows,
)


def _complete_group(n_instances=2, p_values=(1e-3, 1e-4), L=4):
    rows = []
    for instance_idx in range(n_instances):
        identity = collect._expected_instance_identity(L, 1.0, instance_idx)
        for baseline, _ in BASELINE_CONFIGS:
            for p_2q in p_values:
                rows.append({
                    "L": L,
                    "N": L * L,
                    "k": 1.0,
                    "instance_idx": instance_idx,
                    "baseline": baseline,
                    "p_2q": p_2q,
                    "accounting_model": ACCOUNTING_MODEL,
                    "hall_decomposition_model": _hall_model_for_baseline(
                        baseline
                    ),
                    "gamma_schedule_model": _gamma_schedule_model_for_baseline(
                        baseline, L
                    ),
                    **identity,
                    **collect.RESOURCE_DEPENDENCY_VERSIONS,
                    "n_ancillas": 0,
                    "total_qubits": L * L,
                    "cnot_depth": 20,
                    "fp_cnot_depth": 0 if baseline == "naive_pauli" else 12,
                    "interaction_cnot_depth": (
                        20 if baseline == "naive_pauli" else 8
                    ),
                    "total_cnots": 100,
                    "n_fswap": 40,
                    "n_2q_cnot_cz": 20,
                    "total_idle_slots_layered": L * L * 20 - 200,
                    "spacetime_volume": L * L * 20,
                    "n_1q_hadamard": 1,
                    "n_1q_other": 2,
                    "n_1q_s_sdag": 3,
                    "n_1q_rz": 4,
                    "p_idle": p_2q,
                    "mult_fidelity": (
                        (1 - p_2q) ** 100
                        * (1 - p_2q) ** (L * L * 20 - 200)
                    ),
                })
    return pd.DataFrame(rows)


def test_resume_accepts_only_complete_native_group():
    df = _complete_group()
    keys, kept = _validated_resume_rows(df, 2, (1e-3, 1e-4))
    assert keys == {(4, 1.0)}
    assert len(kept) == len(df)


def test_resume_rejects_legacy_or_incomplete_group():
    df = _complete_group()

    legacy = df.drop(columns=["accounting_model"])
    keys, kept = _validated_resume_rows(legacy, 2, (1e-3, 1e-4))
    assert not keys
    assert kept.empty

    incomplete = df.iloc[:-1]
    keys, kept = _validated_resume_rows(incomplete, 2, (1e-3, 1e-4))
    assert not keys
    assert kept.empty


def test_resume_rejects_broken_native_idle_invariant():
    df = _complete_group()
    df.loc[0, "total_idle_slots_layered"] += 1
    keys, kept = _validated_resume_rows(df, 2, (1e-3, 1e-4))
    assert not keys
    assert kept.empty


def test_resume_rejects_broken_depth_breakdown():
    df = _complete_group()
    df.loc[0, "interaction_cnot_depth"] += 1
    keys, kept = _validated_resume_rows(df, 2, (1e-3, 1e-4))
    assert not keys
    assert kept.empty


def test_resume_rejects_changed_idle_noise_factor():
    df = _complete_group()
    keys, kept = _validated_resume_rows(
        df, 2, (1e-3, 1e-4), p_idle_factor=0.2
    )
    assert not keys
    assert kept.empty


def test_resume_rejects_inconsistent_fidelity():
    df = _complete_group()
    df.loc[0, "mult_fidelity"] *= 0.99
    keys, kept = _validated_resume_rows(df, 2, (1e-3, 1e-4))
    assert not keys
    assert kept.empty


def test_resume_rejects_noise_dependent_rz_count():
    df = _complete_group()
    df.loc[1, "n_1q_rz"] += 1
    keys, kept = _validated_resume_rows(df, 2, (1e-3, 1e-4))
    assert not keys
    assert kept.empty


def test_resume_rejects_stale_hall_decomposition():
    df = _complete_group()
    df.loc[df["baseline"] == "folded", "hall_decomposition_model"] = (
        "networkx_unseeded"
    )
    keys, kept = _validated_resume_rows(df, 2, (1e-3, 1e-4))
    assert not keys
    assert kept.empty


@pytest.mark.parametrize(
    "mutate",
    [
        lambda df: df.drop(columns=["gamma_schedule_model"]),
        lambda df: df.assign(
            gamma_schedule_model="folded_midpoint_fallback_v0"
        ),
    ],
)
def test_resume_rejects_stale_gamma_schedule(mutate):
    keys, kept = _validated_resume_rows(
        mutate(_complete_group()), 2, (1e-3, 1e-4)
    )
    assert not keys
    assert kept.empty


def test_resume_invalidates_only_changed_boundary_schedule_rows():
    """Stale L=3..6 rows are dropped while unchanged L>=7 rows survive."""
    from common.gamma_folded import FOLDED_GAMMA_SCHEDULE_MODEL

    boundary = _complete_group(L=6)
    asymptotic = _complete_group(L=8)
    boundary.loc[
        boundary["baseline"] == "folded", "gamma_schedule_model"
    ] = FOLDED_GAMMA_SCHEDULE_MODEL

    keys, kept = _validated_resume_rows(
        pd.concat([boundary, asymptotic]), 2, (1e-3, 1e-4)
    )

    assert keys == {(8, 1.0)}
    assert set(kept["L"]) == {8}
    assert len(kept) == len(asymptotic)


def test_resume_rejects_stale_instance_fingerprint():
    df = _complete_group()
    df.loc[df["instance_idx"] == 0, "quartets_sha256"] = "0" * 64
    keys, kept = _validated_resume_rows(df, 2, (1e-3, 1e-4))
    assert not keys
    assert kept.empty


@pytest.mark.parametrize("column", ["cirq_version", "openfermion_version"])
def test_resume_rejects_stale_circuit_dependency(column):
    df = _complete_group()
    df.loc[df["instance_idx"] == 0, column] = "stale-version"
    keys, kept = _validated_resume_rows(df, 2, (1e-3, 1e-4))
    assert not keys
    assert kept.empty


def test_resume_rejects_stale_pauli_rotation_compiler():
    df = _complete_group()
    df.loc[
        df["instance_idx"] == 0, "pauli_rotation_compiler_model"
    ] = "lexicographic_nonlocal_legacy"
    keys, kept = _validated_resume_rows(df, 2, (1e-3, 1e-4))
    assert not keys
    assert kept.empty


def test_resume_keeps_valid_groups_and_discards_invalid_groups():
    complete = _complete_group()
    incomplete = _complete_group()
    incomplete["L"] = 6
    incomplete = incomplete.iloc[:-1]

    keys, kept = _validated_resume_rows(
        pd.concat([complete, incomplete]), 2, (1e-3, 1e-4)
    )

    assert keys == {(4, 1.0)}
    assert set(kept["L"]) == {4}
    assert len(kept) == len(complete)


def test_instance_resume_retains_complete_part_of_partial_L():
    complete_instance = _complete_group(n_instances=1)
    interrupted_instance = _complete_group(n_instances=1)
    interrupted_instance["instance_idx"] = 1
    interrupted_instance = interrupted_instance.iloc[:-1]

    keys, kept = _validated_instance_resume_rows(
        pd.concat([complete_instance, interrupted_instance]),
        (1e-3, 1e-4),
    )

    assert keys == {(4, 1.0, 0)}
    assert set(kept["instance_idx"]) == {0}
    assert len(kept) == len(complete_instance)


def test_partial_L_resume_skips_valid_instance_and_checkpoints_each_new_one(
    tmp_path, monkeypatch
):
    p_values = (1e-3, 1e-4)
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    csv_path = output_dir / "data.csv"
    complete_instance = _complete_group(n_instances=1, p_values=p_values)
    interrupted_instance = _complete_group(n_instances=1, p_values=p_values)
    interrupted_instance["instance_idx"] = 1
    interrupted_instance = interrupted_instance.iloc[:-1]
    pd.concat([complete_instance, interrupted_instance]).to_csv(
        csv_path, index=False,
    )

    collected_indices = []
    expected_identities = {
        idx: collect._expected_instance_identity(4, 1.0, idx)
        for idx in range(3)
    }

    def fake_generate_sparse_syk(L, k, rng):
        return SimpleNamespace(L=L, k=k, quartets=[object()], n_colors=3)

    def fake_collect_instance(instance, instance_idx, requested_p, idle_factor):
        collected_indices.append(instance_idx)
        rows = _complete_group(n_instances=1, p_values=requested_p)
        rows["L"] = instance.L
        rows["N"] = instance.L * instance.L
        rows["k"] = instance.k
        rows["instance_idx"] = instance_idx
        identity = expected_identities[instance_idx]
        for column, value in identity.items():
            rows[column] = value
        return rows.to_dict("records")

    checkpoint_sizes = []
    real_atomic_write = collect._atomic_write_csv

    def recording_atomic_write(df, path):
        checkpoint_sizes.append(len(df))
        real_atomic_write(df, path)

    monkeypatch.setattr(collect, "generate_sparse_syk", fake_generate_sparse_syk)
    monkeypatch.setattr(collect, "collect_instance", fake_collect_instance)
    monkeypatch.setattr(collect, "_atomic_write_csv", recording_atomic_write)

    result = collect.run_experiment(
        L_values=(4,),
        k_values=(1.0,),
        n_instances=3,
        p_values=p_values,
        output_dir=str(output_dir),
    )

    rows_per_instance = len(BASELINE_CONFIGS) * len(p_values)
    assert collected_indices == [1, 2]
    assert checkpoint_sizes == [2 * rows_per_instance, 3 * rows_per_instance]
    assert len(result) == 3 * rows_per_instance

    saved = pd.read_csv(csv_path)
    keys, kept = _validated_instance_resume_rows(saved, p_values)
    assert keys == {(4, 1.0, 0), (4, 1.0, 1), (4, 1.0, 2)}
    assert len(kept) == 3 * rows_per_instance
    assert not list(output_dir.glob(".data.csv.*.tmp"))


def test_atomic_checkpoint_preserves_previous_csv_on_write_failure(
    tmp_path, monkeypatch
):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("previous checkpoint\n", encoding="utf-8")

    def interrupted_to_csv(self, stream, *args, **kwargs):
        stream.write("partial replacement\n")
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(pd.DataFrame, "to_csv", interrupted_to_csv)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        collect._atomic_write_csv(pd.DataFrame({"x": [1]}), str(csv_path))

    assert csv_path.read_text(encoding="utf-8") == "previous checkpoint\n"
    assert not list(tmp_path.glob(".data.csv.*.tmp"))


def test_atomic_checkpoint_retries_transient_windows_replace_lock(
    tmp_path, monkeypatch
):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("old\n", encoding="utf-8")
    real_replace = collect.os.replace
    attempts = []

    def flaky_replace(source, destination):
        attempts.append((source, destination))
        if len(attempts) < 3:
            raise PermissionError("simulated transient reader lock")
        real_replace(source, destination)

    monkeypatch.setattr(collect.os, "replace", flaky_replace)
    monkeypatch.setattr(collect.time, "sleep", lambda _: None)
    collect._atomic_write_csv(pd.DataFrame({"x": [1]}), str(csv_path))

    assert len(attempts) == 3
    assert pd.read_csv(csv_path).to_dict("records") == [{"x": 1}]
    assert not list(tmp_path.glob(".data.csv.*.tmp"))
