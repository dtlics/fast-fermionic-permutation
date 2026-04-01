## Ancilla-free $\Gamma$ circuit

We now construct an explicit circuit for $\Gamma$ using only nearest-neighbor $\text{CZ}$, $\text{CNOT}$, and $Z$ gates on the bare $L \times L$ grid, with **zero ancilla qubits**.
The circuit has depth $8L + O(1) = 8\sqrt{N} + O(1)$ and uses $O(N)$ gates -- asymptotically negligible compared to the $O(N\sqrt{N})$ sorting stages.
(Empirically, the constant is $8L + 9$ for odd $L \ge 5$ and $8L + 10$ for even $L \ge 6$.)

We first describe the phase polynomial that $\Gamma$ must implement, then develop two pipelined circuit constructions that realize it, assemble the full circuit, and sketch the correctness argument.


### Phase polynomial and building blocks

Recall that $\Gamma$ is diagonal: $\Gamma\ket{s} = (-1)^{f(s)}\ket{s}$.
We need $f(s)$ to be a degree-2 polynomial over $\mathrm{GF}(2)$ satisfying the parity-encoding condition.
The key observation is that $f(s)$ decomposes into a sum of *triangular cross-products*, each involving data from at most two rows.
Define the primitive

$$T(x, y) = \bigoplus_{p < c'} x_p \cdot y_{c'},$$

where $x = (x_0, \ldots, x_{L-1})$ and $y = (y_0, \ldots, y_{L-1})$ are binary row vectors.
This is a strictly upper-triangular sum: every pair $(p, c')$ with $p < c'$ contributes the monomial $x_p \cdot y_{c'}$ -- a total of $\binom{L}{2}$ terms, far more than the $L - 1$ nearest-neighbor connections on a single row.

The phase polynomial has two groups.
The first operates directly on qubit values:

$$f_D(s) = \bigoplus_{r \text{ even}} \Big[ \underbrace{T(s_r, s_r)}_{\text{same-row}} \oplus \underbrace{T(s_r, s_{r+1})}_{\text{cross-row}} \Big],$$

pairing each even row $r$ with its odd neighbor $r+1$.
The second operates on *column-parity* variables $\tilde{s}_{r,c} = \bigoplus_{r' \ge r} s_{r',c}$:

$$f_B(\tilde{s}) = \bigoplus_{\substack{r \text{ even} \\ r+2 \le L-1}} \underbrace{T(\tilde{s}_r, \tilde{s}_{r+2})}_{\text{skip-row}} \;\oplus\; \bigoplus_{\substack{r \text{ even} \\ r \ge 2}} \underbrace{T(\tilde{s}_r, \tilde{s}_r)}_{\text{same-row}}.$$

Three geometric configurations appear:

**Same-row $T(x,x)$.**
All data on one row.
A prefix CNOT cascade builds running parities $\hat{x}_c = \bigoplus_{c' \le c} x_{c'}$.
A CZ between adjacent prefix-sum positions $\hat{x}_{c-1}$ and $\hat{x}_c = \hat{x}_{c-1} \oplus x_c$ contributes $\hat{x}_{c-1} \cdot \hat{x}_c = \hat{x}_{c-1} \cdot x_c + \hat{x}_{c-1}$ over $\mathrm{GF}(2)$.
Summing the degree-2 part over $c$ yields $\sum_{c} \hat{x}_{c-1} x_c = T(x,x)$; the degree-1 residual $\sum_{c=0}^{L-2} \hat{x}_c = \sum_p (L - 1 - p)\, x_p$ is corrected by single-qubit $Z$ gates on columns $p$ where $(L - 1 - p)$ is odd.

**Cross-row $T(x,y)$.**
Row $r$ holds $x$; the adjacent row $r \pm 1$ holds $y$.
A vertical CZ between prefix position $\hat{x}_c$ on row $r$ and untouched $y_c$ on row $r \pm 1$ contributes $\hat{x}_c \cdot y_c = (\bigoplus_{c' \le c} x_{c'}) \cdot y_c$.
Summing gives $\bigoplus_{p \le c'} x_p\, y_{c'}$; a vertical CZ correction subtracts the diagonal $\bigoplus_c x_c y_c$, leaving $T(x,y)$.
Row $r \pm 1$ is never modified.

**Skip-row $T(x,y)$.**
Row $r$ (even) holds $x$; row $r+2$ (even) holds $y$; the odd row $r+1$ is a routing intermediary.
A 4-gate gadget per column -- $\text{CZ}(r+1, r+2)$, $\text{CNOT}(r \to r+1)$, $\text{CZ}(r+1, r+2)$, $\text{CNOT}(r \to r+1)$ -- temporarily copies $\hat{x}_c$ onto the intermediary, applies the CZ interaction with row $r+2$, and restores the intermediary.
The two CZ gates cancel the intermediary's own state, isolating the desired $\hat{x}_c \cdot y_c$ contribution.
A matching correction gadget subtracts the diagonal terms.

In all three cases, a prefix CNOT cascade on the source row builds the running parities $\hat{x}_c = \bigoplus_{c' \le c} x_{c'}$; the interaction gates extract the cross-column products; and an undo cascade restores the original data.
Executed alone, each would cost depth $2L + O(1)$.
The key to an efficient $\Gamma$ circuit is that these interactions can be *pipelined*: the interaction gates trail the cascade wavefront at fixed column offsets, so multiple $T(\cdot,\cdot)$ terms sharing the same source row fold into a single forward-and-back sweep.


### Pipelined constructions

The terms in $f_D$ and $f_B$ naturally group into two pipelined constructions, each performing a prefix cascade on an even row while interleaving trailing interaction gates that accumulate multiple $T(x,y)$ contributions simultaneously.

**Construction A: same-row + skip-row (Algorithm PipelineSameSkip).**
For each active even row $r \ge 2$ with $r+2 \le L-1$, this combines $T(\tilde{s}_r, \tilde{s}_r)$ and $T(\tilde{s}_r, \tilde{s}_{r+2})$ into one sweep.
(Row 0 uses a skip-only variant without the same-row CZ, since $f_B$ excludes same-row $T$ for $r = 0$.)
A forward prefix cascade on row $r$ advances left-to-right; at each time step $\tau$, multiple trailing operations execute in parallel at earlier (already-cascaded) columns.
The skip-row 4-gate gadget occupies offsets $-1$ through $-4$ behind the wavefront, routing the prefix through row $r+1$ to interact with row $r+2$.
The same-row CZ occupies offsets $-5$/$-6$, further back.
All trailing operations touch distinct columns at each step, so no qubit conflicts arise within a group.
The undo cascade sweeps right-to-left with mirrored trailing corrections.

Since the skip-row gadget accesses values on row $r+2$, groups whose target rows overlap cannot run simultaneously: group $r$'s gadget reads row $r+2$ while group $r+2$'s cascade would be modifying it.
We split into two batches: $r \equiv 0 \pmod{4}$ first, then $r \equiv 2 \pmod{4}$.
Within each batch, row triples are non-overlapping (e.g., $(0,1,2)$, $(4,5,6)$, ...).

*Used in Phase 2 of Algorithm ApplyGamma. Depth per batch: $2L + O(1)$. Two batches: $4L + O(1)$ total.*


#### Algorithm: PipelineSameSkip

Same-row $T(x,x)$ + skip-row $T(x,y)$ in one sweep.

```
Input:  Even row r (source), odd row r+1 (intermediary), even row r+2 (skip target).
        Data is in column-parity basis s_tilde.
Output: Applies phase (-1)^{T(s_tilde_r, s_tilde_r) XOR T(s_tilde_r, s_tilde_{r+2})}.
        All rows restored.

--- Forward sweep ---
At each time step tau, the following operations execute in parallel
at distinct column offsets:

  Cascade (offset 0):                               row r, cols tau, tau+1
    CNOT((r, tau) -> (r, tau+1))                     // q_{r,tau+1} <- x_hat_{tau+1}

  Skip-row gadget (offsets -1 to -4):                rows r, r+1, r+2
    CZ((r+1, tau-1), (r+2, tau-1))                   // pre-cancel
    CNOT((r, tau-2) -> (r+1, tau-2))                 // copy x_hat onto intermediary
    CZ((r+1, tau-3), (r+2, tau-3))                   // interact: x_hat_{tau-3} * y_{tau-3}
    CNOT((r, tau-4) -> (r+1, tau-4))                 // restore intermediary

  Same-row CZ (offsets -5/-6):                       row r
    CZ((r, tau-6), (r, tau-5))                       // x_hat_{tau-6} * x_{tau-5}

tau ranges from 0 to L-2; out-of-bounds ops are skipped.
After tau = L-2: drain trailing ops.                 // O(1) steps

--- Undo sweep ---
Reverse cascade (right-to-left) with trailing corrections at mirrored offsets:

  Undo cascade:
    CNOT((r, tau-1) -> (r, tau))                     // restore q_{r,tau} <- s_tilde_{r,tau}

  Skip-row correction gadget (same 4-gate structure,
    trailing at offsets +1 to +4)

  Same-row Z correction (offset +5):
    Z(r, tau+5) if (L-1-(tau+5)) is odd
```


**Construction B: same-row + cross-row (Algorithm PipelineSameCross).**
For each even row $r$, this combines $T(s_r, s_r)$ and $T(s_r, s_{r+1})$ into one sweep.
The pipeline is shorter: the cross-row vertical CZ trails at offset $-2$ and the same-row CZ at offsets $-3$/$-4$.
The undo sweep carries same-row $Z$ corrections and cross-row diagonal CZ corrections at mirrored offsets.

No batching is needed: cross-row CZ gates are diagonal and do not modify the computational-basis values on row $r+1$.
All even-row groups $(r, r+1)$ for $r = 0, 2, 4, \ldots$ are disjoint and execute in parallel.

*Used in Phase 4 of Algorithm ApplyGamma. Depth: $2L + O(1)$.*


#### Algorithm: PipelineSameCross

Same-row $T(x,x)$ + cross-row $T(x,y)$ in one sweep.

```
Input:  Even row r (source), odd row r+1 (cross-row target).
        Data is in original basis s.
Output: Applies phase (-1)^{T(s_r, s_r) XOR T(s_r, s_{r+1})}.
        All rows restored.

--- Forward sweep ---
At each time step tau:

  Cascade (offset 0):                               row r, cols tau, tau+1
    CNOT((r, tau) -> (r, tau+1))                     // q_{r,tau+1} <- x_hat_{tau+1}

  Cross-row CZ (offset -2):                         rows r, r+1
    CZ((r, tau-2), (r+1, tau-2))                     // x_hat_{tau-2} * y_{tau-2}

  Same-row CZ (offsets -3/-4):                       row r
    CZ((r, tau-4), (r, tau-3))                       // x_hat_{tau-4} * x_{tau-3}

tau ranges from 0 to L-2; out-of-bounds ops are skipped.
After tau = L-2: drain trailing ops.                 // O(1) steps

--- Undo sweep ---
Reverse cascade with trailing corrections:

  Undo cascade:
    CNOT((r, tau-1) -> (r, tau))                     // restore q_{r,tau} <- s_{r,tau}

  Cross-row CZ correction (offset +2):
    CZ((r, tau+2), (r+1, tau+2))

  Same-row Z correction (offset +3):
    Z(r, tau+3) if (L-1-(tau+3)) is odd
```


### Full $\Gamma$ circuit

Algorithm ApplyGamma assembles the full $\Gamma$ circuit from four phases.
Phases 1 and 3 switch between the original and column-parity bases via vertical CNOT cascades.
Phase 2 applies $f_B$ using Construction A (two batches); Phase 4 applies $f_D$ using Construction B (single pass).


#### Algorithm: ApplyGamma

Ancilla-free $\Gamma$ on $L \times L$ grid.

```
Phase 1: Enter column-parity basis                  Depth: L-1
  For each column c in parallel:
    For r = L-2 down to 0:
      CNOT((r+1, c) -> (r, c))
  After this phase, qubit (r,c) holds s_tilde_{r,c} = XOR_{r' >= r} s_{r',c}.

Phase 2: Parity-basis interactions                   Depth: 4L + O(1)
  Batch 1: for each even r ≡ 0 (mod 4) with r+2 <= L-1, in parallel:
    if r >= 2: PipelineSameSkip(r, r+1, r+2)         // same + skip
    else (r = 0): PipelineSkipOnly(r, r+1, r+2)      // skip only
  Batch 1 also: for each even r >= 2 with r+2 > L-1 not overlapping batch rows:
    SameRowT(r)                                       // same-row only
  Batch 2: for each even r ≡ 2 (mod 4) with r+2 <= L-1, in parallel:
    PipelineSameSkip(r, r+1, r+2)

Phase 3: Exit column-parity basis                   Depth: L-1
  For each column c in parallel:
    For r = 0 to L-2:
      CNOT((r+1, c) -> (r, c))

Phase 4: Original-basis interactions                 Depth: 2L + O(1)
  For each even r with r+1 < L, in parallel:
    PipelineSameCross(r, r+1)
  If L is odd: SameRowT(L-1)                          // last even row, no cross partner
```


**Conflict-freedom.**
In Phase 2, each batch is internally conflict-free: within a batch, row triples are non-overlapping.
Note that row 0 uses skip-row only (not PipelineSameSkip), since $f_B$ has same-row $T$ for even $r \ge 2$ only.
Remaining same-row-only rows ($r \ge 2$ with $r+2 > L-1$) are scheduled with the batch whose row triples they do not overlap.
The two batches execute sequentially to resolve the cross-group conflict on shared target rows.
In Phase 4, Construction B groups use disjoint row pairs $(0,1)$, $(2,3)$, ... and run fully in parallel.
If $L$ is odd, the last even row $L-1$ has no cross-row partner and uses a standalone same-row $T$ pass.

**Resource summary.**
Total depth:

$$\underbrace{(L-1)}_{\text{Phase 1}} \;+\; \underbrace{2 \times (2L+O(1))}_{\text{Phase 2}} \;+\; \underbrace{(L-1)}_{\text{Phase 3}} \;+\; \underbrace{(2L+O(1))}_{\text{Phase 4}} \;=\; 8L + O(1).$$

Total gates: $O(N)$ ($L$ rows $\times$ $O(L)$ gates per row per phase).
Ancillas: **zero**.


### Why $\Gamma$ satisfies the parity-encoding condition

We sketch the argument for why Algorithm ApplyGamma satisfies the parity-encoding condition; the complete monomial-level verification appears in the appendix.

The core question is: when we flip the two qubits involved in a vertical swap $(r_0, c_0) \leftrightarrow (r_0+1, c_0)$, does the phase polynomial change by exactly the parity string $P = \bigoplus_{\ell=j+1}^{k-1} s_\ell$?
The answer hinges on the *parity* of $r_0$, which determines the shape of the JWT path between the two endpoints.

**Even-row hop ($r_0$ even).**
The JWT path from $(r_0, c_0)$ to $(r_0+1, c_0)$ runs rightward along row $r_0$, turns the snake corner, and returns leftward along row $r_0+1$.
The parity string therefore spans columns $c' > c_0$ on both rows.
Because flipping both $s_{r_0, c_0}$ and $s_{r_0+1, c_0}$ cancels in the column-parity basis (the two flips offset), the parity-basis terms $f_B$ are unaffected: $\Delta f_B = 0$.
The entire contribution comes from $f_D$, specifically from $T(s_{r_0}, s_{r_0})$ and $T(s_{r_0}, s_{r_0+1})$.
In both terms, the flip at column $c_0$ toggles exactly those monomials involving columns $c' > c_0$ -- which is precisely the set of qubits in the parity string.

**Odd-row hop ($r_0$ odd).**
The JWT path now runs leftward, so $P$ spans columns $c' < c_0$.
The column-parity variable $\tilde{s}_{r_0+1, c_0}$ does flip (since only one of the two qubits contributes at that level), and because $r_0+1$ is even, $f_B$ *is* affected.
Six terms across $f_D$ and $f_B$ contribute to $\Delta f$.
All contributions from columns $c' > c_0$ cancel pairwise between the two groups, while contributions from columns $c' < c_0$ survive and sum to $P$.
The parity-basis terms are precisely designed to supply the correction that the original-basis terms alone cannot provide for odd-row hops -- this is the fundamental reason the construction needs both $f_D$ and $f_B$.
