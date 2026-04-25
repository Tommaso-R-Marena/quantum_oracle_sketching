# Quantum Oracle Sketching: Theory and Derivation

This document provides a concise but rigorous treatment of the mathematics underlying Quantum Oracle Sketching (QOS).

## 1. Problem Setting

We consider a quantum computer with $S$ qubits (machine size $S$) that must process a classical dataset of size $N \gg 2^S$. The dataset is accessed only via random samples, not by storing the full data in quantum memory.

## 2. Random-Sample Oracle Model

A **quantum oracle sketch** is a unitary $U$ constructed from $M$ independent random samples such that, for any input basis state $|x\rangle$,
$$U |x\rangle \approx O |x\rangle$$
where $O$ is the exact data oracle. The approximation quality is measured by the Euclidean error (for state sketches) or operator norm error (for oracle sketches).

### 2.1 Single-Sample Gate

For a dataset $D = \{(i_t, v_t)\}_{t=1}^M$ sampled uniformly, the single-sample gate is
$$U_t |x\rangle = \exp\left(i \frac{T}{M} v_t \cdot \mathbb{1}[x = i_t]\right) |x\rangle$$
where $T$ is a time scale depending on the oracle type.

### 2.2 Expected Unitary

The expected single-sample gate is
$$\mathbb{E}[U_t] = \text{diag}\left(\exp\left(\log\left(1 + p_x (e^{i T v_x / M} - 1)\right)\right)\right)$$
where $p_x$ is the sampling probability of index $x$.

By the **mixing lemma** (arXiv:2008.11751), the expected-unitary simulation provides a conservative (pessimistic) upper bound on the real-world random-channel error:
$$\|\mathbb{E}[U_1 \cdots U_M] - O\| \leq \|\mathbb{E}[U_t]^M - O\|$$

## 3. State Sketching

### 3.1 Flat Vectors

For $v \in \{\pm 1\}^N$, the sketch prepares
$$|\psi\rangle = \frac{1}{\sqrt{N}} \sum_x \exp\left(i \frac{\pi N}{M} \sum_t \frac{1 - v_{i_t}}{2} \mathbb{1}[x = i_t]\right) |x\rangle$$

The expected-unitary error satisfies
$$\epsilon = O\left(\frac{N}{M}\right)$$
so $M = O(N / \epsilon)$ samples suffice.

### 3.2 General Vectors

For $v \in \mathbb{R}^N$, we use a **randomized Walsh-Hadamard transform** followed by phase oracle + LCU + QSVT arcsin inversion.

1. **Randomization**: $v \to D_H v$ where $D_H$ is a random diagonal sign matrix.
2. **Walsh-Hadamard**: $v \to H v$ (FJLT-style dimensionality reduction).
3. **Phase oracle**: encode $v$ as phases $e^{i B_x}$.
4. **LCU**: extract $\sin(B)$ via the circuit
   $$S X H (c^1 U^\dagger) (c^0 U) X H T$$
   where $S = \text{diag}(-i, 1)$ and $T = \text{diag}(1, -i)$.
5. **QSVT**: apply polynomial $P(x) \approx \arcsin(x) / \arcsin(1)$.
6. **Inverse FJLT**: undo the randomization.

The total sample complexity is
$$M = O\left(\frac{N \text{polylog}(1/\epsilon)}{\epsilon}\right)$$

## 4. Oracle Sketching

### 4.1 Boolean Functions

For $f: \{0,1\}^n \to \{0,1\}$, the phase oracle is
$$O_f |x\rangle = (-1)^{f(x)} |x\rangle = e^{i \pi f(x)} |x\rangle$$

Sketching uses $M = O(2^n / \epsilon)$ uniform samples of the truth table.

### 4.2 Sparse Matrix Element Oracle

For a sparse matrix $A$ with $\text{nnz}$ non-zero elements, the element oracle is
$$O_A |i\rangle|j\rangle = \arcsin(A_{ij}) |i\rangle|j\rangle$$

The LCU construction yields a block encoding of $\sin(B) = A$, using $M = O(\text{nnz} / \epsilon)$ samples.

### 4.3 Sparse Matrix Index Oracle

The row-index oracle maps $|i\rangle|k\rangle|0\rangle \to |i\rangle|k\rangle|j(i,k)\rangle$ where $j(i,k)$ is the column index of the $k$-th non-zero element in row $i$.

QOS constructs this via:
1. **Cumulative counter**: maintain $\sum_t \mathbb{1}[i_t = i, j_t < l]$ for each row $i$ and column $l$.
2. **Phase encoding**: encode the count as a phase.
3. **Threshold predicate**: use QSVT sign-function amplification to implement
   $$(-1)^{\mathbb{1}[C(i,l) < k]}$$
4. **Binary search**: iteratively apply XOR oracles to extract the binary representation of $j(i,k)$.

The sample complexity is
$$M = O\left(\frac{\text{nnz} \cdot \log N}{\epsilon}\right)$$

## 5. Full Block Encoding

Combining the element oracle $O_A$, row-index oracle $O_R$, and column-index oracle $O_C$, we construct a block encoding of $A$ via Lemma 48 of (arXiv:1806.01838v1):
$$B = \frac{1}{\sqrt{r s}} \sum_{k=1}^{r} \sum_{\ell=1}^{s} O_R^{(k)\dagger} O_A O_C^{(\ell)}$$
where $r$ is row sparsity and $s$ is column sparsity.

## 6. Amplitude Amplification

Given an unnormalized state $|\tilde{\psi}\rangle$ with norm $\alpha < 1$, amplitude amplification uses QSVT to apply the sign function to the singular values of the state, boosting the norm to $\approx 1$.

The Halmos dilation of $|\tilde{\psi}\rangle$ is
$$U = \begin{pmatrix} 0 & |\tilde{\psi}\rangle \\ \langle\tilde{\psi}| & 0 \end{pmatrix}$$
and the QSVT sign function maps $\alpha \to \text{sgn}(\alpha) \approx 1$.

## 7. Sample Complexity Scaling

Empirical benchmarks fit the ansatz
$$M = C \cdot N^\alpha / \epsilon^\beta$$

Typical fitted parameters:

| Sketch type | $\alpha$ | $\beta$ |
|-------------|----------|---------|
| Flat vector | 1.0      | 1.0     |
| General vector | 1.0   | 1.5     |
| Boolean function | 1.0 | 1.0     |
| Matrix element | 1.0   | 1.0     |
| Matrix row index | 1.0 | 1.0     |

## References

1. H. Zhao et al., "Exponential quantum advantage in processing massive classical data," arXiv:2604.07639 (2026).
2. A. Gilyén et al., "Quantum singular value transformation and beyond," arXiv:1806.01838 (2018).
3. Y. Dong et al., "Ground state preparation and energy estimation on early fault-tolerant quantum computers via quantum eigenvalue transformation of unitary matrices," arXiv:2002.11649 (2020).
4. E. Campbell, "Random compiler for fast Hamiltonian simulation," arXiv:1811.08017 (2018).
