"""Deterministic record-depth Gamma schedules for ``3 <= L <= 6``.

The asymptotic folded construction has a closed-form schedule for ``L >= 7``.
These four boundary witnesses close the finite gap without invoking a solver at
runtime.  Qubit indices are row-major.  Every returned two-qubit layer is a
nearest-neighbour matching, and the final Z correction is parallel and free in
the repository's CNOT-equivalent two-qubit-depth metric.

The schedules are certified by ``common/tests/test_gamma.py`` against the full
unitary.  They implement
members of the complete admissible Gamma coset, not necessarily the canonical
representative, and therefore have the same Gamma-sandwich semantics.
"""

from __future__ import annotations

from typing import List, Tuple


Gate = Tuple[str, int, int]
Layer = List[Gate]


def _copy_layers(layers: tuple[tuple[Gate, ...], ...]) -> List[Layer]:
    return [list(layer) for layer in layers]


_L3_GATHER: tuple[tuple[Gate, ...], ...] = (
    (
        ("cx", 1, 0),
        ("cx", 5, 8),
        ("cx", 6, 3),
        ("cx", 7, 4),
    ),
    (
        ("cx", 3, 4),
        ("cx", 2, 5),
        ("cx", 0, 1),
        ("cx", 7, 6),
    ),
)

_L3_PHASE: tuple[tuple[Gate, ...], ...] = (
    (("cz", 1, 4), ("cz", 3, 6)),
    (("cz", 0, 1), ("cz", 4, 5)),
)


_L4_GATHER: tuple[tuple[Gate, ...], ...] = (
    (
        ("cx", 0, 4),
        ("cx", 1, 5),
        ("cx", 6, 2),
        ("cx", 7, 3),
        ("cx", 8, 12),
        ("cx", 9, 13),
        ("cx", 14, 10),
        ("cx", 15, 11),
    ),
    (
        ("cx", 3, 2),
        ("cx", 4, 8),
        ("cx", 5, 9),
        ("cx", 10, 6),
        ("cx", 11, 7),
        ("cx", 14, 15),
    ),
    (
        ("cx", 0, 1),
        ("cx", 5, 4),
        ("cx", 6, 7),
        ("cx", 8, 9),
        ("cx", 11, 10),
        ("cx", 13, 12),
    ),
)

_L4_PHASE: tuple[tuple[Gate, ...], ...] = (
    (
        ("cz", 11, 15),
        ("cz", 0, 4),
        ("cz", 1, 2),
        ("cz", 3, 7),
        ("cz", 8, 12),
        ("cz", 9, 10),
    ),
)


_L5_GATHER: tuple[tuple[Gate, ...], ...] = (
    (
        ("cx", 0, 1),
        ("cx", 2, 7),
        ("cx", 8, 3),
        ("cx", 4, 9),
        ("cx", 5, 6),
        ("cx", 10, 11),
        ("cx", 14, 13),
        ("cx", 15, 16),
        ("cx", 18, 17),
        ("cx", 19, 24),
        ("cx", 20, 21),
    ),
    (
        ("cx", 5, 0),
        ("cx", 1, 6),
        ("cx", 4, 3),
        ("cx", 7, 12),
        ("cx", 13, 8),
        ("cx", 9, 14),
        ("cx", 11, 10),
        ("cx", 20, 15),
        ("cx", 16, 21),
        ("cx", 17, 22),
        ("cx", 23, 18),
        ("cx", 24, 19),
    ),
    (
        ("cx", 9, 4),
        ("cx", 6, 11),
        ("cx", 12, 17),
        ("cx", 18, 13),
        ("cx", 14, 19),
        ("cx", 16, 15),
        ("cx", 21, 22),
    ),
    (
        ("cx", 1, 2),
        ("cx", 4, 3),
        ("cx", 6, 5),
        ("cx", 15, 10),
        ("cx", 11, 12),
        ("cx", 14, 13),
        ("cx", 21, 16),
        ("cx", 17, 18),
        ("cx", 19, 24),
        ("cx", 22, 23),
    ),
)

_L5_PHASE: tuple[tuple[Gate, ...], ...] = (
    (
        ("cz", 2, 7),
        ("cz", 23, 24),
        ("cz", 16, 21),
        ("cz", 12, 13),
        ("cz", 17, 18),
        ("cz", 10, 11),
    ),
    (
        ("cz", 7, 12),
        ("cz", 20, 21),
        ("cz", 18, 23),
        ("cz", 2, 3),
        ("cz", 9, 14),
        ("cz", 1, 6),
        ("cz", 11, 16),
    ),
)


_L6_GATHER: tuple[tuple[Gate, ...], ...] = (
    (
        ("cx", 0, 6),
        ("cx", 1, 7),
        ("cx", 2, 8),
        ("cx", 3, 4),
        ("cx", 10, 11),
        ("cx", 13, 12),
        ("cx", 14, 15),
        ("cx", 17, 16),
        ("cx", 18, 24),
        ("cx", 19, 20),
        ("cx", 23, 22),
        ("cx", 26, 32),
        ("cx", 33, 27),
        ("cx", 34, 28),
        ("cx", 35, 29),
        ("cx", 30, 31),
    ),
    (
        ("cx", 0, 1),
        ("cx", 8, 2),
        ("cx", 3, 9),
        ("cx", 4, 10),
        ("cx", 11, 5),
        ("cx", 7, 6),
        ("cx", 18, 12),
        ("cx", 14, 13),
        ("cx", 16, 15),
        ("cx", 20, 26),
        ("cx", 22, 23),
        ("cx", 24, 25),
        ("cx", 29, 28),
        ("cx", 32, 31),
        ("cx", 34, 35),
    ),
    (
        ("cx", 1, 2),
        ("cx", 9, 3),
        ("cx", 5, 4),
        ("cx", 6, 12),
        ("cx", 8, 7),
        ("cx", 15, 14),
        ("cx", 17, 23),
        ("cx", 19, 18),
        ("cx", 27, 21),
        ("cx", 28, 22),
        ("cx", 25, 26),
        ("cx", 29, 35),
        ("cx", 31, 30),
    ),
    (
        ("cx", 0, 6),
        ("cx", 7, 1),
        ("cx", 4, 3),
        ("cx", 8, 9),
        ("cx", 11, 17),
        ("cx", 12, 18),
        ("cx", 14, 20),
        ("cx", 16, 15),
        ("cx", 22, 21),
        ("cx", 29, 23),
        ("cx", 30, 24),
        ("cx", 28, 27),
    ),
    (
        ("cx", 1, 7),
        ("cx", 4, 10),
        ("cx", 6, 12),
        ("cx", 14, 8),
        ("cx", 9, 15),
        ("cx", 22, 16),
        ("cx", 18, 19),
        ("cx", 21, 20),
        ("cx", 29, 23),
        ("cx", 25, 24),
        ("cx", 27, 28),
    ),
)

_L6_PHASE_0: tuple[Gate, ...] = (
    ("cz", 19, 20),
    ("cz", 10, 11),
    ("cz", 2, 3),
    ("cz", 14, 15),
    ("cz", 27, 33),
    ("cz", 22, 28),
    ("cz", 29, 35),
    ("cz", 1, 7),
    ("cz", 24, 25),
)

_L6_PHASE_1: tuple[Gate, ...] = (
    ("cz", 13, 19),
    ("cz", 3, 9),
    ("cz", 22, 23),
    ("cz", 15, 21),
    ("cz", 26, 27),
    ("cz", 10, 16),
    ("cz", 18, 24),
)

_L6_PHASE_2: tuple[Gate, ...] = (("cz", 3, 4), ("cz", 21, 27))


SMALL_GAMMA_Z_QUBITS = {
    3: (0, 6),
    4: (0, 7, 8, 15),
    5: (1, 2, 4, 9, 11, 12, 21, 23),
    6: (0, 4, 5, 9, 10, 13, 15, 22, 25, 27, 28, 34, 35),
}


def small_gamma_layers(L: int) -> List[Layer]:
    """Return a fresh explicit two-qubit schedule for ``3 <= L <= 6``."""

    if L == 3:
        return _copy_layers(_L3_GATHER + _L3_PHASE + tuple(reversed(_L3_GATHER)))
    if L == 4:
        return _copy_layers(_L4_GATHER + _L4_PHASE + tuple(reversed(_L4_GATHER)))
    if L == 5:
        layers = _copy_layers(_L5_GATHER)
        layers[3].append(("cz", 8, 9))
        layers.extend(_copy_layers(_L5_PHASE))
        layers.extend(_copy_layers(tuple(reversed(_L5_GATHER))))
        layers[7].append(("cz", 0, 1))
        return layers
    if L == 6:
        layers = _copy_layers(_L6_GATHER)
        layers.extend([list(_L6_PHASE_0), list(_L6_PHASE_1)])
        layers.append(list(_L6_GATHER[4]))
        layers.append(list(_L6_PHASE_2))
        layers.extend(_copy_layers(tuple(reversed(_L6_GATHER[:4]))))
        return layers
    raise ValueError("the explicit boundary schedules are defined for 3 <= L <= 6")


def small_gamma_z_qubits(L: int) -> List[int]:
    """Return the fixed parallel-Z repair for a boundary schedule."""

    try:
        return list(SMALL_GAMMA_Z_QUBITS[L])
    except KeyError as exc:
        raise ValueError(
            "the explicit boundary schedules are defined for 3 <= L <= 6"
        ) from exc
