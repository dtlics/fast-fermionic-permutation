"""Sparse SYK instance generation with parent-level coloring.

Generates random sparse SYK Hamiltonians:
    H = sum_{i<j<k<l} J_{ijkl} chi_i chi_j chi_k chi_l

Uses the standard sparse-SYK parameterization: the Hamiltonian contains
approximately k * N_Maj nonzero 4-Majorana terms, where each possible
quartet is included independently with probability
    p = k * N_Maj / C(N_Maj, 4).

Parent-level coloring ensures fermionic permutations move Majorana pairs
(chi_{2t}, chi_{2t+1}) together without splitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import List, Sequence, Tuple

import networkx as nx
import numpy as np

from common.grid import snake_order_indices, validate_permutation


@dataclass
class SYKInstance:
    """A sparse SYK Hamiltonian instance with coloring and packing permutations.

    Attributes:
        L: grid side length, N = L^2 fermionic modes
        k: sparsity parameter (expected terms ~ k * 2N)
        quartets: list of (i,j,k,l) Majorana 4-subsets, i<j<k<l
        couplings: J_{ijkl} array, one per quartet
        color_groups: color_groups[alpha] = list of quartet indices in color alpha
        packing_perms: packing_perms[alpha] = raster-order perm for color alpha
        n_colors: number of colors used
    """
    L: int
    k: float
    quartets: List[Tuple[int, int, int, int]]
    couplings: np.ndarray
    color_groups: List[List[int]]
    packing_perms: List[List[int]]
    n_colors: int


def generate_sparse_syk(
    L: int, k: float, rng: np.random.Generator,
) -> SYKInstance:
    """Generate a sparse SYK instance.

    Each possible 4-subset of {0, ..., 2N-1} is included independently
    with probability p = k * 2N / C(2N, 4), giving ~k*2N terms in
    expectation.

    Args:
        L: grid side length (N = L^2 modes, 2N Majoranas)
        k: sparsity parameter (expected terms ~ k * 2N)
        rng: numpy random generator

    Returns:
        SYKInstance with quartets, coloring, and packing permutations.
    """
    N = L * L
    M = 2 * N  # number of Majoranas

    quartets = _sample_quartets(M, k, rng)
    couplings = rng.normal(0, np.sqrt(6.0 / N**3), size=len(quartets))

    color_groups, n_colors = _build_parent_coloring(quartets, N)
    packing_perms = [
        _compute_packing_perm(
            [quartets[qi] for qi in group], L
        )
        for group in color_groups
    ]

    return SYKInstance(
        L=L, k=k,
        quartets=quartets,
        couplings=couplings,
        color_groups=color_groups,
        packing_perms=packing_perms,
        n_colors=n_colors,
    )


def _sample_quartets(
    M: int, k: float, rng: np.random.Generator,
) -> List[Tuple[int, int, int, int]]:
    """Sample 4-subsets of {0..M-1} with inclusion probability p = k*M / C(M,4).

    Expected number of terms: k * M.
    For small M where enumeration is feasible, we enumerate all C(M,4) subsets
    and accept each with probability p.  For large M, we draw the expected
    count from Binomial and sample that many distinct 4-subsets.
    """
    total_possible = comb(M, 4)
    expected_terms = k * M
    p_include = min(expected_terms / total_possible, 1.0)

    # For large M, direct enumeration of C(M,4) is too expensive.
    # Threshold: enumerate if C(M,4) < 2e6, else sample.
    if total_possible < 2_000_000:
        return _sample_by_enumeration(M, p_include, rng)
    else:
        n_terms = rng.binomial(total_possible, p_include)
        return _sample_n_distinct(M, n_terms, rng)


def _sample_by_enumeration(
    M: int, p: float, rng: np.random.Generator,
) -> List[Tuple[int, int, int, int]]:
    """Enumerate all C(M,4) and include each with probability p."""
    from itertools import combinations
    quartets = []
    for combo in combinations(range(M), 4):
        if rng.random() < p:
            quartets.append(combo)
    return quartets


def _sample_n_distinct(
    M: int, n: int, rng: np.random.Generator,
) -> List[Tuple[int, int, int, int]]:
    """Sample n distinct 4-subsets of {0..M-1} by rejection."""
    seen: set = set()
    quartets: List[Tuple[int, int, int, int]] = []
    max_attempts = n * 20

    for _ in range(max_attempts):
        if len(quartets) >= n:
            break
        idx = rng.choice(M, size=4, replace=False)
        idx.sort()
        key = tuple(idx)
        if key not in seen:
            seen.add(key)
            quartets.append(key)

    return quartets


def _build_parent_coloring(
    quartets: Sequence[Tuple[int, int, int, int]], N: int,
) -> Tuple[List[List[int]], int]:
    """Build parent-level conflict graph and greedy color it.

    Two quartets conflict if their parent mode sets overlap.

    Returns:
        (color_groups, n_colors) where color_groups[c] = list of quartet indices
    """
    n_q = len(quartets)
    if n_q == 0:
        return [], 0

    parent_sets = []
    for q in quartets:
        parent_sets.append(frozenset(m // 2 for m in q))

    # Build conflict graph
    G = nx.Graph()
    G.add_nodes_from(range(n_q))

    # Index quartets by parent for efficient conflict detection
    parent_to_quartets: dict = {}
    for qi, ps in enumerate(parent_sets):
        for p in ps:
            parent_to_quartets.setdefault(p, []).append(qi)

    for _, qi_list in parent_to_quartets.items():
        for i in range(len(qi_list)):
            for j in range(i + 1, len(qi_list)):
                G.add_edge(qi_list[i], qi_list[j])

    coloring = nx.greedy_color(G, strategy="largest_first")
    n_colors = max(coloring.values()) + 1 if coloring else 0

    color_groups: List[List[int]] = [[] for _ in range(n_colors)]
    for qi, c in coloring.items():
        color_groups[c].append(qi)

    return color_groups, n_colors


def _compute_packing_perm(
    quartets_in_group: Sequence[Tuple[int, int, int, int]], L: int,
) -> List[int]:
    """Compute raster-order permutation that packs parents to consecutive snake/JW positions.

    After applying FP with this permutation, each quartet's parent modes
    occupy consecutive positions in the JW (snake) ordering.

    Args:
        quartets_in_group: list of (i,j,k,l) Majorana quartets (disjoint parent sets)
        L: grid side length

    Returns:
        perm: raster-order permutation, perm[src_raster] = dst_raster
    """
    N = L * L
    snake = snake_order_indices(L)  # snake[snake_pos] = raster_idx

    perm = [0] * N
    assigned_sources: set = set()
    cursor = 0

    for quartet in quartets_in_group:
        parents = sorted(set(m // 2 for m in quartet))
        for i, p in enumerate(parents):
            perm[p] = snake[cursor + i]
            assigned_sources.add(p)
        cursor += len(parents)

    remaining_src = [m for m in range(N) if m not in assigned_sources]
    for src in remaining_src:
        perm[src] = snake[cursor]
        cursor += 1

    validate_permutation(perm, N)
    return perm


def invert_permutation(perm: List[int]) -> List[int]:
    """Compute the inverse of a permutation."""
    inv = [0] * len(perm)
    for i, v in enumerate(perm):
        inv[v] = i
    return inv
