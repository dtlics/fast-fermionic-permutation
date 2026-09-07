"""Emit manuscript-ready summary statistics for Experiment 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PLOT_BASELINES = ("1d", "ancilla", "folded")
PERM_KINDS = ("reverse", "transpose", "random")


def _mean_by_family(df: pd.DataFrame, value: str) -> pd.DataFrame:
    return (
        df.groupby(["p_2q", "perm_kind", "L", "baseline"], as_index=False)[value]
        .mean()
    )


def _first_folded_win(group: pd.DataFrame, metric: str, *, lower_is_better: bool) -> int:
    pivot = group.pivot(index="L", columns="baseline", values=metric).sort_index()
    wins = pivot["folded"] < pivot["1d"] if lower_is_better else pivot["folded"] > pivot["1d"]
    return int(wins[wins].index.min())


def summarize(df: pd.DataFrame) -> dict:
    resource_rows = df[df["p_2q"] == df["p_2q"].min()]
    resources = (
        resource_rows.groupby(["perm_kind", "L", "baseline"], as_index=False)[
            ["cnot_depth", "spacetime_volume", "total_cnots", "n_fswap"]
        ]
        .mean()
    )
    max_L = int(resources["L"].max())
    max_N = max_L * max_L

    depth_crossovers = {}
    for kind in PERM_KINDS:
        sub = resources[(resources["perm_kind"] == kind) & resources["baseline"].isin(("1d", "folded"))]
        L = _first_folded_win(sub, "cnot_depth", lower_is_better=True)
        depth_crossovers[kind] = {"L": L, "N": L * L}

    at_largest_N = {}
    for kind in PERM_KINDS:
        rows = resources[
            (resources["perm_kind"] == kind)
            & (resources["L"] == max_L)
            & resources["baseline"].isin(PLOT_BASELINES)
        ].set_index("baseline")
        folded_depth = float(rows.loc["folded", "cnot_depth"])
        folded_volume = float(rows.loc["folded", "spacetime_volume"])
        methods = {}
        for baseline in PLOT_BASELINES:
            methods[baseline] = {
                "cnot_equivalent_depth": float(rows.loc[baseline, "cnot_depth"]),
                "spacetime_volume": float(rows.loc[baseline, "spacetime_volume"]),
                "total_cnot_equivalents": float(rows.loc[baseline, "total_cnots"]),
                "fswaps": float(rows.loc[baseline, "n_fswap"]),
            }
        at_largest_N[kind] = {
            "methods": methods,
            "depth_ratio_1d_over_folded": float(rows.loc["1d", "cnot_depth"] / folded_depth),
            "depth_ratio_ancilla_over_folded": float(rows.loc["ancilla", "cnot_depth"] / folded_depth),
            "depth_reduction_vs_1d_percent": float(100 * (1 - folded_depth / rows.loc["1d", "cnot_depth"])),
            "depth_reduction_vs_ancilla_percent": float(100 * (1 - folded_depth / rows.loc["ancilla", "cnot_depth"])),
            "volume_ratio_1d_over_folded": float(rows.loc["1d", "spacetime_volume"] / folded_volume),
            "volume_ratio_ancilla_over_folded": float(rows.loc["ancilla", "spacetime_volume"] / folded_volume),
            "volume_reduction_vs_1d_percent": float(100 * (1 - folded_volume / rows.loc["1d", "spacetime_volume"])),
            "volume_reduction_vs_ancilla_percent": float(100 * (1 - folded_volume / rows.loc["ancilla", "spacetime_volume"])),
        }

    returns = _mean_by_family(df[df["baseline"].isin(PLOT_BASELINES)], "stim_fidelity")
    return_crossovers = {}
    half_thresholds = {}
    return_at_largest_N = {}
    for p_2q in sorted(returns["p_2q"].unique()):
        p_key = f"{p_2q:.0e}"
        return_crossovers[p_key] = {}
        half_thresholds[p_key] = {}
        return_at_largest_N[p_key] = {}
        for kind in PERM_KINDS:
            sub = returns[(returns["p_2q"] == p_2q) & (returns["perm_kind"] == kind)]
            L = _first_folded_win(
                sub[sub["baseline"].isin(("1d", "folded"))],
                "stim_fidelity",
                lower_is_better=False,
            )
            return_crossovers[p_key][kind] = {"L": L, "N": L * L}
            half_thresholds[p_key][kind] = {}
            return_at_largest_N[p_key][kind] = {}
            for baseline in PLOT_BASELINES:
                curve = sub[sub["baseline"] == baseline].sort_values("L")
                passing = curve[curve["stim_fidelity"] >= 0.5]
                last_L = int(passing["L"].max()) if not passing.empty else None
                half_thresholds[p_key][kind][baseline] = {
                    "L": last_L,
                    "N": last_L * last_L if last_L is not None else None,
                }
                value = curve.loc[
                    curve["L"] == max_L, "stim_fidelity"
                ].iloc[0]
                return_at_largest_N[p_key][kind][baseline] = float(value)

    strict_audit = {}
    strict_rows = df[
        (df["L"] == max_L)
        & (df["p_2q"] == 1e-5)
        & (df["baseline"] == "folded")
    ]
    for kind in PERM_KINDS:
        rows = strict_rows[strict_rows["perm_kind"] == kind]
        strict_audit[kind] = {
            "strict_no_fault_probability": float(rows["mult_fidelity"].mean()),
            "legacy_no_fault_probability": float(rows["mult_fidelity_legacy"].mean()),
            "strict_over_legacy_probability_ratio": float(
                rows["mult_fidelity"].mean() / rows["mult_fidelity_legacy"].mean()
            ),
            "strict_cnot_equivalents": float(rows["total_cnots"].mean()),
            "legacy_raw_two_qubit_gates": float(rows["total_2q_gates_legacy"].mean()),
        }

    return {
        "largest_sampled_size": {"L": max_L, "N": max_N},
        "depth_first_folded_win_vs_1d": depth_crossovers,
        "first_size_where_depth_win_holds_for_all_families": {
            "L": max(item["L"] for item in depth_crossovers.values()),
            "N": max(item["L"] for item in depth_crossovers.values()) ** 2,
        },
        "resources_at_largest_sampled_N": at_largest_N,
        "return_probability_first_folded_win_vs_1d": return_crossovers,
        "last_sampled_size_with_return_probability_at_least_half": half_thresholds,
        "return_probability_at_largest_sampled_N": return_at_largest_N,
        "strict_vs_legacy_folded_accounting_at_largest_sampled_N_p_1e_5": strict_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=Path("exp1_fp/results/data.csv"))
    args = parser.parse_args()
    print(json.dumps(summarize(pd.read_csv(args.csv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
