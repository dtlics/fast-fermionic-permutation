"""Regression tests for Experiment 1 reproducibility provenance."""

from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from exp1_fp.backfill_provenance import backfill
from exp1_fp.collect import (
    DEPENDENCY_VERSIONS,
    DETERMINISTIC_PERMUTATION_SEED,
    PERMUTATION_MODEL,
    REPRODUCIBILITY_COLUMNS,
    _permutation_provenance,
    add_reproducibility_provenance,
    has_valid_reproducibility_provenance,
)


def _rows_without_provenance() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"L": 4, "perm_kind": "reverse", "perm_idx": 0, "value": 1.5},
            {"L": 4, "perm_kind": "random", "perm_idx": 0, "value": 2.5},
            {"L": 4, "perm_kind": "random", "perm_idx": 1, "value": 3.5},
        ]
    )


def test_permutation_provenance_uses_seed_sentinel_and_exact_digest():
    structured = _permutation_provenance(4, "reverse", 0)
    random_0 = _permutation_provenance(4, "random", 0)
    random_1 = _permutation_provenance(4, "random", 1)

    assert structured["permutation_model"] == PERMUTATION_MODEL
    assert structured["permutation_seed"] == DETERMINISTIC_PERMUTATION_SEED
    assert random_0["permutation_seed"] == 0
    assert random_1["permutation_seed"] == 1
    assert len(structured["permutation_sha256"]) == 64
    assert len({
        structured["permutation_sha256"],
        random_0["permutation_sha256"],
        random_1["permutation_sha256"],
    }) == 3


def test_add_provenance_preserves_every_existing_value_and_rejects_tampering():
    original = _rows_without_provenance()
    enriched = add_reproducibility_provenance(original)

    assert_frame_equal(enriched[original.columns], original, check_exact=True)
    assert REPRODUCIBILITY_COLUMNS.issubset(enriched.columns)
    assert has_valid_reproducibility_provenance(enriched)
    for column, value in DEPENDENCY_VERSIONS.items():
        assert set(enriched[column]) == {value}

    tampered = enriched.copy()
    tampered.loc[0, "permutation_sha256"] = "0" * 64
    assert not has_valid_reproducibility_provenance(tampered)


def test_backfill_writes_atomically_without_changing_existing_columns(tmp_path):
    source = tmp_path / "data.csv"
    original = _rows_without_provenance()
    original.to_csv(source, index=False)

    enriched = backfill(source)
    reread = pd.read_csv(source)

    assert_frame_equal(reread[original.columns], original, check_exact=True)
    assert has_valid_reproducibility_provenance(reread)
    assert_frame_equal(reread, enriched, check_exact=True)
    assert not (tmp_path / "data.csv.tmp").exists()

    first_bytes = source.read_bytes()
    backfill(source)
    assert source.read_bytes() == first_bytes
