# Fermionic Permutation — Draft Writeup

## Task

Let $n = L^2$ be the number of qubits arranged on an $L \times L$ grid, and $\pi \in S_n$ be a permutation of $\{0, \dots, n-1\}$. We use **raster order** for indexing: position $(r, c) \mapsto r \cdot L + c$.

The **Fermionic Permutation operator** $\mathcal{F}_\pi$ is defined as the composition of a phase operator and a permutation operator:

$$\mathcal{F}_\pi = \hat{P}_\pi \, \hat{V}_\pi$$

1. **Phase Operator** $\hat{V}_\pi$: Applies a $CZ$ gate to every inversion pair. Let $\text{Inv}(\pi) = \{ (i, j) \mid i < j \text{ and } \pi(i) > \pi(j) \}$. Then $\hat{V}_\pi = \prod_{(i,j) \in \text{Inv}(\pi)} CZ_{i, j}$.

2. **Permutation Operator** $\hat{P}_\pi$: Physically rearranges quantum states, moving the qubit at index $i$ to position $\pi(i)$.

---

## Motivation

- [ ]  The Fermionic Fast Fourier Transform (FFFT) - Quantum Chemistry in Plane Wave Basis
- [ ]  talk
- [ ]  The Sachdev-Ye-Kitaev (SYK) model and BLACK HOLE?
- [x]  For a Centrosymmetric Matching on a 2D grid, a permutation saves a lot

---

## Existing Literature

In the https://arxiv.org/abs/1711.04789 paper they constructed a fermionic swap network that every qubit i and j would be adjacent at some point of the network. Exactly how many downstream tasks need that is TBD. However, http://arxiv.org/abs/2509.08898 explicitly argued for the importance of an efficient implementation of a Fermionic Permutation gadget.

---

## Our Contribution (3-Stage Routing + Measurement-Based Phase Tracking)

### Grid Layout

Data qubits sit at even columns: position $(r, c)$ maps to physical qubit $(r, 2c)$. Between every pair of adjacent data qubits in a row, there is an **ancilla qubit** at $(r, 2c+1)$. Each row has $L$ data qubits and $L-1$ ancillas.

### Top-Level Algorithm

```
Algorithm FermionicPermutation2D(π)

Input: permutation π on L×L grid in raster order
Output: quantum circuit implementing F_π

# ---- Classical pre-processing: 3-stage routing plan ----
(S₁, S₂, S₃) ← HallDecomposition(π)
    # S₁[c] : within-column-c permutation such that after stage 1,
    #          each row contains items destined for all L distinct columns
    # S₂[r] : within-row-r permutation sending each item to its destination column
    # S₃[c] : within-column-c permutation sending each item to its destination row

# ---- Quantum execution with fermionic phases ----
ExecuteColumnStage(S₁)
ExecuteRowStage(S₂)
ExecuteColumnStage(S₃)
```

### Hall's Decomposition

Build a bipartite multigraph $G$ with left nodes $\{C_0, \dots, C_{L-1}\}$ (source columns) and right nodes $\{T_0, \dots, T_{L-1}\}$ (target columns). For each grid position $(r, c)$, add an edge from $C_c$ to $T_{\pi(r \cdot L + c) \bmod L}$.

Since every left and right node has degree exactly $L$, by Hall's theorem $G$ can be decomposed into $L$ perfect matchings. Each matching $k$ assigns one item per column to "transit row" $k$, which defines the within-column permutation $S_1[c]$. The row permutations $S_2$ and final column permutations $S_3$ follow directly from the intermediate positions.

### Row Stage (FSWAP Network)

```
Procedure ExecuteRowStage(S₂)

For each row r in parallel:
    Apply a standard 1D fermionic swap (FSWAP) network to the L data
    qubits in row r, implementing the within-row permutation S₂[r].

    # Each FSWAP gate simultaneously swaps two qubits and applies a CZ,
    # which is exactly the fermionic exchange operator for adjacent modes.
    # The network uses odd-even transposition sorting in O(L) rounds.
```

### Column Stage (Core Contribution)

Each column stage performs odd-even transposition sorting on all columns simultaneously, with measurement-based feedforward for O(1)-depth fermionic phase tracking.

```
Procedure ExecuteColumnStage(S)

# cur[c] = current within-column permutation for column c (mutable copy)
cur[c] ← copy of S[c]   for all columns c

For round t = 0, 1, …, L-1:

    # ── Determine which vertical pairs to swap this round ──
    round_parity ← t mod 2
    start ← 1 if round_parity = 1, else 0
    active_swaps ← {}
    For each column c:
        For i = start, start+2, …, L-2:
            if cur[c][i] > cur[c][i+1]:
                active_swaps[c].add(i)

    if no active swaps anywhere:
        if all columns sorted: break
        else: continue

    # ── Four sequential steps per round ──
    Step 1.  BuildPrefixXOR(round_parity, t)
    Step 2.  PhaseCorrections(active_swaps, round_parity)
    Step 3.  UnbuildPrefixXOR(round_parity, t)
    Step 4.  VerticalSWAPs(active_swaps)

    # ── Classical bookkeeping ──
    For each column c, for each active row i in active_swaps[c]:
        swap cur[c][i] ↔ cur[c][i+1]
```

**Key structural point:** The SWAP gates happen *after* unbuilding the prefix XOR, not inside it. The four steps are cleanly separated.

### Step 1: BuildPrefixXOR

Encodes row-wise prefix parity information into the data qubits using ancillas, mid-circuit measurement, and classical feedforward. Direction alternates each round.

```
Procedure BuildPrefixXOR(round_parity, t)

# Direction: determines the sweep order along each row
#   round_parity = 0  →  R-to-L (reverse qubit order)
#   round_parity = 1  →  L-to-R (natural qubit order)
For each row r:
    (Q[0..L-1], A[0..L-2]) ← GetDirection(round_parity, r)
        # Q = data qubits, A = ancillas, in sweep order

# Moment 1: Reset all ancillas to |0⟩
RESET(a) for all ancillas a

# Moment 2: Prepare ancillas in |+⟩
H(a) for all ancillas a

# Moment 3: Entangle ancillas into data (CNOT anc → data)
For each row r, for k = 0 to L-2:
    CNOT(A[k], Q[k+1])         # ancilla controls, data is target

# Moment 4: Encode data parities into ancillas (CNOT data → anc)
For each row r, for k = 0 to L-2:
    CNOT(Q[k], A[k])           # data controls, ancilla is target

# Moment 5: Measure all ancillas in Z basis (one key per row)
For each row r:
    m[r] ← Measure(A[0], A[1], …, A[L-2])

# Moment 6: Classically-controlled prefix-X corrections
For each row r, for k = 0 to L-2:
    prefix ← m[r][0] ⊕ m[r][1] ⊕ … ⊕ m[r][k]
    if prefix = 1:
        X(Q[k+1])
```

### Step 2: PhaseCorrections

Applies fermionic CZ and Z phases for all active swaps in this round. Gates are accumulated globally, duplicates cancelled, and scheduled into at most 2 non-conflicting rounds.

```
Procedure PhaseCorrections(active_swaps, round_parity)

# ── Accumulate CZ and Z gates across all active swap positions ──
cz_counts ← {}    # maps (qubit, qubit) → count
z_counts  ← {}    # maps qubit → count

For each column c, for each active row r in active_swaps[c]:

    TL ← data(r, c)              # top qubit of the swap pair
    BL ← data(r+1, c)            # bottom qubit of the swap pair

    # Determine horizontal neighbor based on round parity
    if round_parity = 0 and c < L-1:
        TR ← data(r, c+1)        # right neighbor on top row
        BR ← data(r+1, c+1)      # right neighbor on bottom row
    elif round_parity = 1 and c > 0:
        TR ← data(r, c-1)        # left neighbor on top row
        BR ← data(r+1, c-1)      # left neighbor on bottom row
    else:
        TR ← None; BR ← None     # corner edge: no neighbor

    add_cz(TL, BL)
    if TR ≠ None:
        add_cz(TL, TR);  add_z(TR)
    if BR ≠ None:
        add_cz(BL, BR);  add_z(BR)
    if TR ≠ None and BR ≠ None:
        add_cz(TR, BR)

# ── Cancel even-count duplicates ──
z_ops  ← { Z(q)       | z_counts[q] mod 2 = 1 }
cz_ops ← { CZ(q1,q2)  | cz_counts[(q1,q2)] mod 2 = 1 }

# ── Schedule CZ into ≤ 2 rounds via graph 2-coloring ──
(round_A, round_B) ← TwoColorSchedule(cz_ops)
    # Build conflict graph: two CZ edges conflict if they share a qubit.
    # 2-color this graph (always feasible for our topology).

# ── Compile horizontal CZ through ancilla ──
# For a horizontal CZ between adjacent data qubits (r,c) and (r,c±1),
# use the ancilla between them:
#   RESET(anc)
#   CNOT(left_data, anc) → CZ(anc, right_data) → CNOT(left_data, anc)
# Vertical CZ gates are applied directly (qubits are adjacent).

# ── Emit moments ──
Apply z_ops
For each of round_A, round_B:
    Apply vertical CZ directly + horizontal CZ via ancilla compilation
```

### Step 3: UnbuildPrefixXOR

Reverses the prefix-XOR encoding and applies suffix-Z corrections based on measurement outcomes.

```
Procedure UnbuildPrefixXOR(round_parity, t)

# Same direction convention as BuildPrefixXOR
For each row r:
    (Q[0..L-1], A[0..L-2]) ← GetDirection(round_parity, r)

# Moment 1: Reset all ancillas to |0⟩
RESET(a) for all ancillas a

# Moment 2: Encode data parities into ancillas (CNOT data → anc)
For each row r, for k = 0 to L-2:
    CNOT(Q[k], A[k])

# Moment 3: Propagate ancilla info back to data (CNOT anc → data)
For each row r, for k = 0 to L-2:
    CNOT(A[k], Q[k+1])

# Moment 4: Rotate ancillas to X basis
H(a) for all ancillas a

# Moment 5: Measure all ancillas (X-basis measurement via H + Z-measure)
For each row r:
    m[r] ← Measure(A[0], A[1], …, A[L-2])

# Moment 6: Classically-controlled suffix-Z corrections
For each row r, for k = L-2 down to 0:
    suffix ← m[r][k] ⊕ m[r][k+1] ⊕ … ⊕ m[r][L-2]
    if suffix = 1:
        Z(Q[k])
```

### Step 4: VerticalSWAPs

```
Procedure VerticalSWAPs(active_swaps)

# All swaps are independent and applied in a single moment.
For each column c, for each active row r in active_swaps[c]:
    SWAP(data(r, c), data(r+1, c))
```

---

## Resource Scaling

| Method | CNOT-equivalent 2Q gates | Measurements | Idle overhead |
|--------|--------------------------|--------------|---------------|
| Baseline (1D FSWAP network) | $\sim L^4$ | 0 | 0 |
| Custom (3-stage + MCM) | $\sim 16 L^3$ | $\sim 4 L^3$ | $\sim 4 \mu \cdot L^3$ |

where $\mu$ is the measurement duration in CNOT-equivalent time units. For large $L$, the custom method achieves an asymptotic advantage from $O(L^4)$ to $O(L^3)$.

---
