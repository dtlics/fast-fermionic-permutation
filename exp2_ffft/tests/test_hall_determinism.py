"""Cross-process reproducibility checks for Hall-routed FFFT resources."""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pandas as pd

from exp2_ffft.collect import PUBLICATION_L_VALUES


ROOT = Path(__file__).resolve().parents[2]
L_VALUES = PUBLICATION_L_VALUES


RESOURCE_SCRIPT = textwrap.dedent(
    """
    import json

    from common.fp_2d import GammaMethod, build_fp_2d
    from common.gamma_folded import build_gamma_folded
    from common.grid import make_system_qubits
    from exp2_ffft.collect import PUBLICATION_L_VALUES, count_cnot_resources
    from exp2_ffft.col_ffft_bare import (
        build_bare_column_fffts,
        build_row_fffts,
    )
    from exp2_ffft.ffft_proper import (
        _build_odd_row_reversal,
        _col_major_raster_to_row_major_snake_perm,
        build_ffft_proper,
    )
    from exp2_ffft.twiddle import build_twiddle_circuit

    output = {}
    for L in PUBLICATION_L_VALUES:
        sq = make_system_qubits(L)
        gamma, _ = build_gamma_folded(L, sq=sq)
        fp = build_fp_2d(
            L,
            _col_major_raster_to_row_major_snake_perm(L),
            GammaMethod.FOLDED,
        ).circuit
        stages = [
            _build_odd_row_reversal(L, sq),
            gamma,
            build_bare_column_fffts(L, sq),
            gamma,
            build_twiddle_circuit(L, sq),
            build_row_fffts(L, sq),
            fp,
        ]
        stage_depths = [
            count_cnot_resources(stage, L, 0)["cnot_depth"]
            for stage in stages
        ]
        resources = count_cnot_resources(
            build_ffft_proper(L, GammaMethod.FOLDED).circuit,
            L,
            0,
        )
        output[str(L)] = {
            "resources": resources,
            "stage_depths": stage_depths,
        }
    print(json.dumps(output, sort_keys=True))
    """
)


def _resources_from_fresh_process(hash_seed: int) -> dict:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(hash_seed)
    prior_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + prior_path if prior_path else "")
    result = subprocess.run(
        [sys.executable, "-c", RESOURCE_SCRIPT],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_proper_resources_are_hash_seed_independent_and_match_csv():
    """Every official proper-FFFT row and stage total is reproducible."""
    fresh = [_resources_from_fresh_process(seed) for seed in (0, 1, 8675309)]
    assert fresh[1:] == fresh[:-1]

    df = pd.read_csv(ROOT / "exp2_ffft" / "results" / "data.csv")
    proper = df[df["method"] == "gamma_2d_proper"]
    resource_columns = [
        "L",
        "N",
        "n_ancillas",
        "total_qubits",
        "cnot_depth",
        "total_2q_gates",
        "total_cnot_equiv",
        "total_idle_slots",
        "total_idle_slots_layered",
        "n_1q_nonz",
        "n_exact_t",
        "n_synth_rz",
    ]

    for L in L_VALUES:
        rows = proper[proper["L"] == L]
        assert len(rows) == 3
        assert all(rows[column].nunique() == 1 for column in resource_columns)
        csv_resources = {
            column: int(rows.iloc[0][column])
            for column in resource_columns
        }
        built = fresh[0][str(L)]
        assert built["resources"] == csv_resources
        assert sum(built["stage_depths"]) == csv_resources["cnot_depth"]
