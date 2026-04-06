"""
Visualize BK→JW CNOT intervals on a Hilbert curve.

For N = 2^k - 1 qubits mapped to an L×L grid (L = 2^(k/2)) via a Hilbert
curve, this script:
  1. Generates the BK→JW CNOT rounds via balanced-BST right rotations
  2. Maps qubit indices to Hilbert-curve grid positions
  3. Plots each round, coloring the CNOT intervals on the grid
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

FIGURES_DIR = Path(__file__).parent / "figures"

# Vibrant-but-mild qualitative palette (no grays — gray is reserved for unused cells).
INTERVAL_PALETTE = [
    (0.55, 0.83, 0.78, 1.0),  # teal
    (1.00, 0.73, 0.47, 1.0),  # peach
    (0.75, 0.73, 0.85, 1.0),  # lavender
    (0.98, 0.50, 0.45, 1.0),  # coral
    (0.50, 0.69, 0.83, 1.0),  # steel blue
    (0.99, 0.71, 0.80, 1.0),  # rose
    (0.70, 0.87, 0.54, 1.0),  # soft green
    (0.99, 0.85, 0.47, 1.0),  # golden
    (0.85, 0.60, 0.70, 1.0),  # mauve
    (0.58, 0.77, 0.64, 1.0),  # sage
    (0.80, 0.58, 0.46, 1.0),  # sienna
    (0.55, 0.63, 0.80, 1.0),  # periwinkle
    (0.90, 0.56, 0.67, 1.0),  # dusty pink
    (0.65, 0.85, 0.73, 1.0),  # mint
    (0.85, 0.75, 0.55, 1.0),  # tan
    (0.60, 0.75, 0.90, 1.0),  # sky blue
]

# ---------------------------------------------------------------------------
# Section A: Balanced BST and BK→JW round generation
# ---------------------------------------------------------------------------


@dataclass
class Node:
    val: int
    left: Optional["Node"] = None
    right: Optional["Node"] = None


def build_balanced_bst(lo: int, hi: int) -> Optional[Node]:
    """Build a balanced BST with inorder labels lo..hi."""
    if lo > hi:
        return None
    mid = (lo + hi) // 2
    return Node(
        val=mid,
        left=build_balanced_bst(lo, mid - 1),
        right=build_balanced_bst(mid + 1, hi),
    )


def _is_leaf(node: Optional[Node]) -> bool:
    return node is not None and node.left is None and node.right is None


def _right_spine(root: Node) -> list[Node]:
    """Return the list of nodes on the right spine (root → rightmost)."""
    spine = []
    cur = root
    while cur is not None:
        spine.append(cur)
        cur = cur.right
    return spine


def generate_bk_jw_rounds(k: int) -> list[list[tuple[int, int]]]:
    """Generate BK→JW CNOT rounds for N = 2^k - 1 (1-based labels).

    Returns list of rounds, each round is a list of (j, parent) pairs.
    """
    N = 2**k - 1
    root = build_balanced_bst(1, N)
    assert root is not None

    # We need parent pointers for rotation. Use a dict node.val -> parent Node.
    # Rebuild after each round of rotations.
    def build_parent_map(r: Node) -> dict[int, Optional[Node]]:
        pmap: dict[int, Optional[Node]] = {r.val: None}
        stack = [r]
        while stack:
            n = stack.pop()
            if n.left:
                pmap[n.left.val] = n
                stack.append(n.left)
            if n.right:
                pmap[n.right.val] = n
                stack.append(n.right)
        return pmap

    rounds: list[list[tuple[int, int]]] = []

    while True:
        spine = _right_spine(root)
        parent_map = build_parent_map(root)

        # Collect rotation candidates: right-spine nodes that have a left
        # child (rotate it away to eventually make the tree a pure right chain).
        pairs: list[tuple[int, int]] = []
        rotations: list[tuple[Node, Node]] = []  # (spine_node, left_child)
        for sn in spine:
            j = sn.left
            if j is not None:
                pairs.append((j.val, sn.val))
                rotations.append((sn, j))

        if not pairs:
            break

        # Perform all right rotations for this round.
        for sn, j in rotations:
            # Right-rotate j about sn:
            #   j takes sn's position
            #   sn becomes j's right child
            #   j's old right subtree becomes sn's new left subtree
            sn.left = j.right
            j.right = sn

            # Update parent of sn to point to j instead.
            p = parent_map[sn.val]
            if p is None:
                root = j
            elif p.left is sn:
                p.left = j
            else:
                p.right = j

        rounds.append(pairs)

    return rounds


# ---------------------------------------------------------------------------
# Section B: Hilbert curve
# ---------------------------------------------------------------------------


def hilbert_d2xy(n: int, d: int) -> tuple[int, int]:
    """Map Hilbert-curve index d to (x, y) on an n×n grid (n = power of 2)."""
    x = y = 0
    s = 1
    while s < n:
        rx = 1 & (d // 2)
        ry = 1 & (d ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        d //= 4
        s *= 2
    return x, y


def precompute_hilbert(L: int) -> np.ndarray:
    """Return array of shape (L*L, 2) mapping index -> (col, row)."""
    coords = np.empty((L * L, 2), dtype=int)
    for d in range(L * L):
        coords[d] = hilbert_d2xy(L, d)
    return coords


# ---------------------------------------------------------------------------
# Section C: Plotting
# ---------------------------------------------------------------------------


def _render_round_on_ax(
    ax,
    round_pairs: list[tuple[int, int]],
    round_idx: int,
    total_rounds: int,
    N: int,
    L: int,
    coords: np.ndarray,
) -> int:
    """Render one round of CNOT intervals onto *ax*. Returns max_interval."""
    num_intervals = len(round_pairs)
    colors = [INTERVAL_PALETTE[i % len(INTERVAL_PALETTE)] for i in range(num_intervals)]

    # Build RGBA grid: start white.
    grid = np.ones((L, L, 4), dtype=float)

    # Mark unused cells (indices N .. L*L-1) as light gray.
    for u in range(N, L * L):
        ux, uy = coords[u]
        grid[uy, ux] = [0.85, 0.85, 0.85, 1.0]

    # Color intervals and collect edge segments.
    max_interval = 0
    all_segments: list[list[tuple[float, float]]] = []
    all_seg_colors: list[tuple[float, ...]] = []

    for idx, (a, b) in enumerate(round_pairs):
        interval_len = b - a
        max_interval = max(max_interval, interval_len)
        color = colors[idx]

        for i in range(a, b + 1):
            cx, cy = coords[i]
            grid[cy, cx] = color

        for i in range(a, b):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            all_segments.append(
                [(x1 + 0.5, L - y1 - 0.5), (x2 + 0.5, L - y2 - 0.5)]
            )
            all_seg_colors.append(color)

    ax.imshow(grid, origin="upper", extent=[0, L, 0, L], interpolation="nearest")

    # Background Hilbert curve trace.
    bg_segments = []
    for i in range(L * L - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        bg_segments.append(
            [(x1 + 0.5, L - y1 - 0.5), (x2 + 0.5, L - y2 - 0.5)]
        )
    bg_lc = LineCollection(
        bg_segments, colors="#a0a0a0", linewidths=0.7, alpha=0.7, zorder=1
    )
    ax.add_collection(bg_lc)

    # Draw colored interval edge lines on top.
    if all_segments:
        lc = LineCollection(
            all_segments,
            colors=all_seg_colors,
            linewidths=1.2 if L <= 16 else 0.8,
            alpha=0.85,
            zorder=2,
        )
        ax.add_collection(lc)

    # Grid lines.
    for i in range(L + 1):
        ax.axhline(i, color="gray", linewidth=0.3, alpha=0.5)
        ax.axvline(i, color="gray", linewidth=0.3, alpha=0.5)

    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Round {round_idx + 1}/{total_rounds}  |  "
        f"max interval = {max_interval}",
        fontsize=12,
    )

    return max_interval


def plot_round(
    round_pairs: list[tuple[int, int]],
    round_idx: int,
    total_rounds: int,
    N: int,
    L: int,
    coords: np.ndarray,
    output_dir: Path,
):
    """Plot one round of CNOT intervals on the Hilbert-curve grid."""
    figsize = (7, 7) if L <= 16 else (9, 9)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    max_interval = _render_round_on_ax(
        ax, round_pairs, round_idx, total_rounds, N, L, coords,
    )

    fig.tight_layout()
    stem = f"round_{round_idx + 1:02d}_N{N}"
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {stem} (max interval {max_interval})")


def plot_all_rounds(
    all_rounds: list[list[tuple[int, int]]],
    N: int,
    L: int,
    coords: np.ndarray,
    output_dir: Path,
    k: int,
):
    """Plot all rounds side-by-side in a single figure."""
    num_rounds = len(all_rounds)
    fig, axes = plt.subplots(1, num_rounds, figsize=(5 * num_rounds, 5))
    if num_rounds == 1:
        axes = [axes]

    for r_idx, rnd in enumerate(all_rounds):
        _render_round_on_ax(axes[r_idx], rnd, r_idx, num_rounds, N, L, coords)

    fig.suptitle(
        f"BK→JW CNOT rounds:  k = {k},  N = {N},  {L}×{L} grid",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    stem = f"all_rounds_k{k}_N{N}"
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {stem}.svg ({num_rounds} rounds)")


# ---------------------------------------------------------------------------
# Section D: Verification & main
# ---------------------------------------------------------------------------

REFERENCE_ROUNDS = {
    4: [  # k=4, N=15
        [(4, 8), (10, 12), (13, 14)],
        [(2, 4), (6, 8), (9, 10), (11, 12)],
        [(1, 2), (3, 4), (5, 6), (7, 8)],
    ],
    6: [  # k=6, N=63
        [(16, 32), (40, 48), (52, 56), (58, 60), (61, 62)],
        [
            (8, 16), (24, 32), (36, 40), (44, 48),
            (50, 52), (54, 56), (57, 58), (59, 60),
        ],
        [
            (4, 8), (12, 16), (20, 24), (28, 32),
            (34, 36), (38, 40), (42, 44), (46, 48),
            (49, 50), (51, 52), (53, 54), (55, 56),
        ],
        [
            (2, 4), (6, 8), (10, 12), (14, 16),
            (18, 20), (22, 24), (26, 28), (30, 32),
            (33, 34), (35, 36), (37, 38), (39, 40),
            (41, 42), (43, 44), (45, 46), (47, 48),
        ],
        [
            (1, 2), (3, 4), (5, 6), (7, 8),
            (9, 10), (11, 12), (13, 14), (15, 16),
            (17, 18), (19, 20), (21, 22), (23, 24),
            (25, 26), (27, 28), (29, 30), (31, 32),
        ],
    ],
}


def verify_against_reference():
    """Check generated rounds match known reference data."""
    for k, expected in REFERENCE_ROUNDS.items():
        rounds = generate_bk_jw_rounds(k)
        assert len(rounds) == len(expected), (
            f"k={k}: expected {len(expected)} rounds, got {len(rounds)}"
        )
        for r_idx, (got, exp) in enumerate(zip(rounds, expected)):
            assert got == exp, (
                f"k={k}, round {r_idx + 1}: expected {exp}, got {got}"
            )
    print("Verification passed for k=4 and k=6.")


def main():
    verify_against_reference()

    os.makedirs(FIGURES_DIR, exist_ok=True)

    import math

    # Combined subfigures for even k values (L² = N+1, tight fit).
    for k_small in [4, 6]:
        N_s = 2**k_small - 1
        L_s = 2 ** (k_small // 2)
        print(f"\nGenerating combined plot for k={k_small}, N={N_s}, grid={L_s}×{L_s}")
        rounds_1based_s = generate_bk_jw_rounds(k_small)
        rounds_s = [[(a - 1, b - 1) for a, b in rnd] for rnd in rounds_1based_s]
        coords_s = precompute_hilbert(L_s)
        plot_all_rounds(rounds_s, N_s, L_s, coords_s, FIGURES_DIR, k_small)

    # Individual round plots for k=10.
    for k in [10]:
        N = 2**k - 1
        L = 2 ** (k // 2)
        print(f"\nGenerating plots for k={k}, N={N}, grid={L}×{L}")

        rounds_1based = generate_bk_jw_rounds(k)
        rounds = [[(a - 1, b - 1) for a, b in rnd] for rnd in rounds_1based]

        coords = precompute_hilbert(L)

        for r_idx, rnd in enumerate(rounds):
            plot_round(rnd, r_idx, len(rounds), N, L, coords, FIGURES_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
