"""Hall's deterministic 3-stage Row-Col-Row decomposition.

Decomposes any permutation on an L x L grid into three stages:
    RowA -> ColSort -> RowB

where RowA/RowB are within-row permutations (horizontal FSWAPs, JWT-adjacent)
and ColSort is a within-column permutation (vertical bare FSWAPs).

Key property: only the column stage requires Gamma correction, so we need
just 2 applications of Gamma (before and after ColSort) rather than 4.
"""

from typing import Dict, List, Sequence, Tuple

from common.grid import validate_permutation


HALL_DECOMPOSITION_MODEL = "canonical_augmenting_v1"


def _deterministic_perfect_matching(
    multiplicity: List[List[int]],
) -> List[int]:
    """Match every source row to a destination row in canonical order.

    ``multiplicity[sr][dr] > 0`` defines the support graph.  The residual
    multigraph in the Hall decomposition is regular, so its support always
    has a perfect matching.  A standard augmenting-path algorithm with both
    sides visited in ascending integer order makes the chosen matching
    independent of Python's hash seed and NetworkX's internal set order.

    Returns:
        ``dst_for_src`` with one distinct destination row for each source row.
    """
    L = len(multiplicity)
    dst_to_src = [-1] * L

    def augment(src: int, seen_dst: List[bool]) -> bool:
        for dst in range(L):
            if multiplicity[src][dst] <= 0 or seen_dst[dst]:
                continue
            seen_dst[dst] = True
            incumbent = dst_to_src[dst]
            if incumbent == -1 or augment(incumbent, seen_dst):
                dst_to_src[dst] = src
                return True
        return False

    for src in range(L):
        if not augment(src, [False] * L):
            raise AssertionError(
                "regular bipartite residual graph has no perfect matching"
            )

    dst_for_src = [-1] * L
    for dst, src in enumerate(dst_to_src):
        if src < 0:
            raise AssertionError("perfect matching left a destination unmatched")
        dst_for_src[src] = dst
    return dst_for_src


def decompose_permutation_rcr(
    L: int, perm: Sequence[int]
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]]]:
    """Decompose permutation into Row-Col-Row stages.

    Args:
        L: grid side length
        perm: permutation of 0..N-1 in raster order (perm[src] = dst)

    Returns:
        (s1, s2, s3) where:
          s1[r][c] = target column for item at (r,c) in RowA stage
          s2[c][r] = target row for item at (r,c) after RowA in ColSort stage
          s3[r][c] = target column for item at (r,c) after ColSort in RowB stage
    """
    N = L * L
    validate_permutation(list(perm), N)

    # Build a source-row/destination-row multiplicity matrix.  Parallel edges
    # retain raster (source-column) order so choosing a concrete item from a
    # matched row pair is deterministic as well.
    multiplicity = [[0] * L for _ in range(L)]
    items: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for r in range(L):
        for c in range(L):
            idx = r * L + c
            dst = perm[idx]
            dst_r, dst_c = dst // L, dst % L
            multiplicity[r][dst_r] += 1
            items.setdefault((r, dst_r), []).append((c, dst_c))
    next_item = {(sr, dr): 0 for sr, dr in items}

    # RowA: item at (sr, sc) goes to transit column k
    s1 = {r: [0] * L for r in range(L)}
    post_rowA: Dict[Tuple[int, int], Tuple[int, int]] = {}

    # Repeatedly remove one canonical perfect matching.  Each matching is one
    # transit column k in the RCR decomposition.
    for k in range(L):
        dst_for_src = _deterministic_perfect_matching(multiplicity)
        for sr, dr in enumerate(dst_for_src):
            item_index = next_item[(sr, dr)]
            sc, dc = items[(sr, dr)][item_index]
            next_item[(sr, dr)] = item_index + 1
            multiplicity[sr][dr] -= 1
            s1[sr][sc] = k
            post_rowA[(sr, k)] = (dr, dc)

    if any(count for row in multiplicity for count in row):
        raise AssertionError("Hall decomposition did not consume every item")

    # ColSort: item at (sr, k) needs to go to row dr
    s2 = {c: [0] * L for c in range(L)}
    post_colsort = {}

    for (sr, k), (dr, dc) in post_rowA.items():
        s2[k][sr] = dr
        post_colsort[(dr, k)] = dc

    # RowB: item at (dr, k) needs to go to column dc
    s3 = {r: [0] * L for r in range(L)}
    for (dr, k), dc in post_colsort.items():
        s3[dr][k] = dc

    return s1, s2, s3
