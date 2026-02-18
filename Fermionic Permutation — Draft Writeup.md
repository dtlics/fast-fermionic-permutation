# Fermionic Permutation — Draft Writeup

## Task

Let $n$ be the number of qubits and $\pi \in S_n$ be a permutation of the set $\{1, \dots, n\}$. The **Fermionic Permutation operator** $\mathcal{F}_\pi$ is defined on the Hilbert space $\mathcal{H}^{\otimes n}$ as the composition of a permutation operator $\hat{P}_\pi$ and a pre-swap phase operator $\hat{V}_\pi$: $\mathcal{F}_\pi = \hat{P}_\pi \hat{V}_\pi$. where the operators are defined as follows: 1. The Phase Operator ( $\hat{V}_\pi$ ): This operator applies a Controlled-Z ( $CZ$ ) gate to every pair of qubits $(i, j)$ that constitutes an inversion (a pair that will eventually swap order). Let $\text{Inv}(\pi) = \{ (i, j) \mid 1 \le i < j \le n \text{ and } \pi(i) > \pi(j) \}$. The operator acts on the qubits at their initial indices: $\hat{V}_\pi = \prod_{(i,j) \in \text{Inv}(\pi)} CZ_{i, j}$. 2. $i$ to index $\pi(i)$ **The Permutation Operator ( $\hat{P}_\pi$ ):** This operator physically rearranges the quantum states. It maps any tensor product state $|\psi_1\rangle \otimes \dots \otimes |\psi_n\rangle$ to the permuted state $|\psi_{\pi^{-1}(1)}\rangle \otimes \dots \otimes |\psi_{\pi^{-1}(n)}\rangle$, effectively moving the qubit originally at index $i$ to $\pi(i)$.

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

## Our contribution (3-phase routing + phase tracking)

### Pseudocode (simplified notation)

```jsx
Algorithm FermionicPermutation2D(π)

Let L = √N  // grid is L×L
Fix a snake order x1,…,xN induced by the grid traversal.

# ---- compile the 3-stage routing plan (classical) ----
(ColA[1..L], Row[1..L], ColB[1..L]) ← Hall3StagePlan(π)
    # ColA[c] : permutation within column c such that
    #           after Column-A, each row has all distinct destination columns (indeed, all L distinct)
    # Row[r]  : permutation within row r to send each item to its destination column
    # ColB[c] : permutation within column c to send each item to its destination row

# Convert each within-line permutation into an odd-even schedule of L rounds.
SchedColA[c] ← OddEvenSchedule(ColA[c])  for all columns c
SchedRow[r]  ← OddEvenSchedule(Row[r])   for all rows r
SchedColB[c] ← OddEvenSchedule(ColB[c])  for all columns c

# ---- execute with fermionic phases ----
ExecuteColumnStage(SchedColA)
ExecuteRowStage(SchedRow)
ExecuteColumnStage(SchedColB)
```

### Row stage (easy: nearest-neighbor in snake)

```jsx
Procedure ExecuteRowStage(SchedRow)

For round t = 1..L (in the odd-even schedules):
  For each row r in parallel:
    For each scheduled horizontal edge (r,c)-(r,c+1) in round t in parallel:
      SWAP(data(r,c), data(r,c+1))
      CZ  (data(r,c), data(r,c+1))

```

### Column stage (coordinated rounds + O(1) prefix-xor + two-step CZ)

```jsx
Procedure ExecuteColumnStage(SchedCol)

For round t = 1..L:

  # Global coordination: in this round, every column uses the same row-pairs.
  # If t is odd: swap row pairs (1,2)(3,4)… ; if t is even: (2,3)(4,5)…
  rowPairs ← GlobalOddEvenRowPairs(t)

  # Build the row-wise prefix/suffix XOR "labels" needed for this round in O(1) depth.
  # Uses the CNOT-ladder-parallelization gadget with ancillas + MCM+FF (your first figure).
  BuildPrefixXOR(rowPairs)

  # For each column, perform its scheduled vertical swaps in this round.
  # For each vertical swap, add the fermionic phase for (endpoints + all between-elements).
  For each column c in parallel:
    For each scheduled vertical edge (r,c)-(r+1,c) in round t of SchedCol[c] in parallel:
        VerticalSwapWithFermionicPhase((r,c),(r+1,c))
    End
  End

  UnbuildPrefixXOR(rowPairs)  # O(1) depth
End
```

```jsx
Procedure VerticalSwapWithFermionicPhase(qTop, qBot)

# qTop and qBot are vertically adjacent data qubits in the active row-pair.
# Prefix-XOR labels for this round already exist.

isCornerEdge ← (qTop is the row-corner qubit)     # then qBot is also the row-corner qubit

TL ← TopLabel(qTop)        # inclusive parity from qTop to corner
TR ← TopNext(qTop)         # excluded parity (empty iff isCornerEdge)
BL ← BotLabel(qBot)
BR ← BotNext(qBot)

# Step 1
CZ (TL, BL)
if not isCornerEdge:
    X (TR)
    CZ (TR, BR)
    X (TR)

# Step 2
if not isCornerEdge:
    CZ (TL, TR)
    CZ (BL, BR)
    Z (TR)
    Z (BR)

```

---