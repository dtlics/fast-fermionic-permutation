"""Hall's 3-stage Row-Col-Row decomposition for 2D permutations.

Decomposes any permutation on an L x L grid into three stages:
    RowA -> ColSort -> RowB

where RowA/RowB are within-row permutations (horizontal FSWAPs, JWT-adjacent)
and ColSort is a within-column permutation (vertical bare FSWAPs).

Key property: only the column stage requires Gamma correction, so we need
just 2 applications of Gamma (before and after ColSort) rather than 4.
"""

from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np

from common.grid import validate_permutation


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

    # Build bipartite multigraph: source rows -> destination rows
    G = nx.MultiGraph()
    G.add_nodes_from([f"R{i}" for i in range(L)], bipartite=0)
    G.add_nodes_from([f"D{i}" for i in range(L)], bipartite=1)

    edge_data = []
    for r in range(L):
        for c in range(L):
            idx = r * L + c
            dst = perm[idx]
            dst_r, dst_c = dst // L, dst % L
            key = G.add_edge(f"R{r}", f"D{dst_r}")
            edge_data.append((f"R{r}", f"D{dst_r}", key, r, c, dst_r, dst_c))

    # Decompose L-regular bipartite graph into L perfect matchings
    matchings = []
    H = G.copy()
    for _ in range(L):
        simp_H = nx.Graph(H.edges())
        matching_dict = nx.bipartite.maximum_matching(
            simp_H, top_nodes=[f"R{i}" for i in range(L)]
        )
        matched_edges = []
        seen = set()
        for u, v in matching_dict.items():
            if (u, v) in seen or (v, u) in seen:
                continue
            seen.add((u, v))
            key = list(H[u][v].keys())[0]
            matched_edges.append((u, v, key))
            H.remove_edge(u, v, key=key)
        matchings.append(matched_edges)

    # Build lookup: edge key -> (src_r, src_c, dst_r, dst_c)
    edge_lookup = {}
    for Rn, Dn, key, sr, sc, dr, dc in edge_data:
        edge_lookup[(Rn, Dn, key)] = (sr, sc, dr, dc)
        edge_lookup[(Dn, Rn, key)] = (sr, sc, dr, dc)

    # RowA: item at (sr, sc) goes to transit column k
    s1 = {r: [0] * L for r in range(L)}
    post_rowA = {}

    for k, matching in enumerate(matchings):
        for u, v, key in matching:
            info_key = (u, v, key) if u.startswith("R") else (v, u, key)
            sr, sc, dr, dc = edge_lookup[info_key]
            s1[sr][sc] = k
            post_rowA[(sr, k)] = (dr, dc)

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
