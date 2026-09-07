"""Scientific-structure regressions for sparse-SYK collection."""

from dataclasses import replace

import numpy as np
import pandas as pd

from common.gamma_folded import (
    GAMMA_SCHEDULE_NOT_APPLICABLE,
    folded_gamma_schedule_model,
)
from exp3_syk.collect import (
    REPRODUCIBILITY_COLUMNS,
    RESOURCE_DEPENDENCY_VERSIONS,
    add_reproducibility_provenance,
    collect_instance,
    validate_colors_data,
    validate_reproducibility_provenance,
    validate_resource_dependency_provenance,
)
from exp3_syk.syk_instance import (
    generate_sparse_syk,
    instance_reproducibility_provenance,
)


def test_coupling_rescale_does_not_change_coloring_or_resource_rows():
    """Coupling normalization changes angles, never compiled structure."""
    instance = generate_sparse_syk(
        L=4,
        k=0.5,
        rng=np.random.default_rng(seed=7),
    )
    rescaled = replace(instance, couplings=instance.couplings * 7.0)

    assert rescaled.quartets == instance.quartets
    assert rescaled.color_groups == instance.color_groups
    assert rescaled.packing_perms == instance.packing_perms
    assert rescaled.n_colors == instance.n_colors
    assert (
        instance_reproducibility_provenance(rescaled, 7)
        == instance_reproducibility_provenance(instance, 7)
    )

    original_rows = collect_instance(instance, 7, (1e-4,), 0.1)
    rescaled_rows = collect_instance(rescaled, 7, (1e-4,), 0.1)
    assert original_rows == rescaled_rows
    assert {
        row["gamma_schedule_model"]
        for row in original_rows
        if row["baseline"] == "folded"
    } == {folded_gamma_schedule_model(instance.L)}
    assert {
        row["gamma_schedule_model"]
        for row in original_rows
        if row["baseline"] != "folded"
    } == {GAMMA_SCHEDULE_NOT_APPLICABLE}

    legacy = pd.DataFrame(original_rows).drop(
        columns=[*REPRODUCIBILITY_COLUMNS, *RESOURCE_DEPENDENCY_VERSIONS]
    )
    upgraded = add_reproducibility_provenance(legacy)
    validate_reproducibility_provenance(upgraded)
    validate_resource_dependency_provenance(upgraded)
    for column, expected in RESOURCE_DEPENDENCY_VERSIONS.items():
        assert set(upgraded[column]) == {expected}

    identity = instance_reproducibility_provenance(instance, 7)
    colors = pd.DataFrame([{
        "L": 4,
        "N": 16,
        "k": 0.5,
        "instance_idx": 7,
        "n_colors": instance.n_colors,
        "n_quartets": len(instance.quartets),
        **identity,
    }])
    # The validation API expects instance indices 0..n_instances-1; use the
    # matching seed-indexed row to exercise exact key/fingerprint validation.
    seed_zero = generate_sparse_syk(4, 0.5, np.random.default_rng(seed=0))
    colors.loc[0, "instance_idx"] = 0
    colors.loc[0, "n_colors"] = seed_zero.n_colors
    colors.loc[0, "n_quartets"] = len(seed_zero.quartets)
    for key, value in instance_reproducibility_provenance(seed_zero, 0).items():
        colors.loc[0, key] = value
    validate_colors_data(colors, (4,), (0.5,), 1)
