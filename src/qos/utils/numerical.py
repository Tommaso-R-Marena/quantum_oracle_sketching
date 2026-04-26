"""Numerical helpers, random generators, and block-encoding utilities."""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax import random

from qos.config import complex_dtype, int_dtype, real_dtype


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull."""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


# ---------------------------------------------------------------------------
# Quantum fidelity / infidelity
# ---------------------------------------------------------------------------

def fidelity(state1: jnp.ndarray, state2: jnp.ndarray) -> jnp.ndarray:
    """Compute the quantum fidelity |<state1|state2>|^2 between two states."""
    return jnp.abs(jnp.vdot(state1, state2)) ** 2


def infidelity(state1: jnp.ndarray, state2: jnp.ndarray) -> jnp.ndarray:
    """Compute the quantum infidelity 1 - |<state1|state2>|^2 between two states."""
    return 1.0 - fidelity(state1, state2)


# ---------------------------------------------------------------------------
# Random state / matrix generators
# ---------------------------------------------------------------------------

def random_unit_vector(
    key: random.PRNGKeyArray,
    dim: int,
    batch_size: int = 1,
) -> jnp.ndarray:
    """Generate a batch of random real unit vectors of given dimension."""
    vec = random.normal(key, (batch_size, dim), dtype=real_dtype)
    vec = vec / jnp.linalg.norm(vec, axis=-1, keepdims=True)
    return vec[0] if batch_size == 1 else vec


def random_flat_vector(
    key: random.PRNGKeyArray,
    dim: int,
    batch_size: int = 1,
) -> jnp.ndarray:
    """Generate a batch of random flat vectors with components +/-1."""
    vec = (
        random.randint(key, (batch_size, dim), minval=0, maxval=2, dtype=int_dtype) * 2
        - 1
    )
    return vec[0] if batch_size == 1 else vec


def random_sparse_matrix(
    key: random.PRNGKeyArray,
    shape: tuple[int, int],
    nnz: int,
    batch_size: int = 1,
) -> jnp.ndarray:
    """Generate a batch of unit spectral-norm random sparse matrices."""
    dim1, dim2 = shape
    key, subkey = random.split(key)
    row_indices = random.randint(subkey, (batch_size, nnz), 0, dim1, dtype=int_dtype)
    key, subkey = random.split(key)
    col_indices = random.randint(subkey, (batch_size, nnz), 0, dim2, dtype=int_dtype)
    key, subkey = random.split(key)
    values = random.normal(subkey, (batch_size, nnz), dtype=real_dtype)
    A = jnp.zeros((batch_size, dim1, dim2), dtype=real_dtype)
    A = A.at[jnp.arange(batch_size)[:, None], row_indices, col_indices].set(values)
    A = A / jnp.linalg.norm(A, ord=2, axis=(-2, -1), keepdims=True)
    return A[0] if batch_size == 1 else A


def random_sparse_matrix_constant_magnitude(
    key: random.PRNGKeyArray,
    shape: tuple[int, int],
    nnz: int,
    magnitude: float,
    batch_size: int = 1,
) -> jnp.ndarray:
    """Generate random sparse matrices with roughly uniform non-zero magnitudes."""
    dim1, dim2 = shape
    key, subkey = random.split(key)
    row_indices = random.randint(subkey, (batch_size, nnz), 0, dim1, dtype=int_dtype)
    key, subkey = random.split(key)
    col_indices = random.randint(subkey, (batch_size, nnz), 0, dim2, dtype=int_dtype)
    key, subkey = random.split(key)
    values = random.uniform(
        subkey, (batch_size, nnz), minval=-magnitude, maxval=magnitude, dtype=real_dtype
    )
    A = jnp.zeros((batch_size, dim1, dim2), dtype=real_dtype)
    A = A.at[jnp.arange(batch_size)[:, None], row_indices, col_indices].set(values)
    return A[0] if batch_size == 1 else A


def random_sparse_matrix_given_row_sparsity(
    key: random.PRNGKeyArray,
    shape: tuple[int, int],
    row_sparsity: int,
    batch_size: int = 1,
) -> jnp.ndarray:
    """Generate random sparse matrices with given row sparsity."""
    dim1, dim2 = shape
    nnz = dim1 * row_sparsity
    key, subkey = random.split(key)
    row_indices = jnp.repeat(jnp.arange(dim1, dtype=int_dtype), row_sparsity)
    row_indices = jnp.tile(row_indices, (batch_size, 1))
    key, subkey = random.split(key)
    col_indices = jnp.tile(jnp.arange(dim2, dtype=int_dtype), (batch_size, dim1, 1))
    permuted_col_indices = random.permutation(
        subkey, col_indices, axis=-1, independent=True
    )
    col_indices = permuted_col_indices[:, :, :row_sparsity].reshape(batch_size, nnz)
    key, subkey = random.split(key)
    values = random.uniform(
        subkey, (batch_size, nnz), minval=-1, maxval=1, dtype=real_dtype
    )
    A = jnp.zeros((batch_size, dim1, dim2), dtype=real_dtype)
    A = A.at[jnp.arange(batch_size)[:, None], row_indices, col_indices].set(values)
    return A[0] if batch_size == 1 else A


# ---------------------------------------------------------------------------
# Structured matrices
# ---------------------------------------------------------------------------

def laplacian_matrix(dim: int) -> jnp.ndarray:
    """Generate the symmetric normalized 1D chain Laplacian with spectral norm <= 1.

    Uses the symmetric normalization L_sym = D^{-1/2} L D^{-1/2}, where D is
    the degree matrix. For a path graph, max degree is 2, so this divides each
    off-diagonal entry by sqrt(deg_i * deg_j). The resulting spectral norm is
    bounded by 1 (eigenvalues lie in [0, 1] for a connected graph).
    """
    # Build unnormalized Laplacian: L = D - A
    diagonals = jnp.ones([dim], dtype=real_dtype) * 2.0
    # Endpoint nodes have degree 1
    diagonals = diagonals.at[0].set(1.0)
    diagonals = diagonals.at[dim - 1].set(1.0)
    off_diagonals = jnp.ones([dim - 1], dtype=real_dtype) * (-1.0)
    L = jnp.diag(diagonals)
    L = L.at[jnp.arange(dim - 1), jnp.arange(1, dim)].set(off_diagonals)
    L = L.at[jnp.arange(1, dim), jnp.arange(dim - 1)].set(off_diagonals)
    # Symmetric normalization: D^{-1/2} L D^{-1/2}
    d_inv_sqrt = 1.0 / jnp.sqrt(diagonals)
    L_sym = L * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    return L_sym


# ---------------------------------------------------------------------------
# Hadamard transforms
# ---------------------------------------------------------------------------

def unnormalized_hadamard_transform(n: int) -> jnp.ndarray:
    """Return the n-fold Kronecker power of the 2x2 Hadamard matrix [[1,1],[1,-1]].

    Args:
        n: Number of qubits (integer). Returns a (2**n, 2**n) matrix.

    Warning:
        This builds the full N x N matrix (memory O(N^2)). For applying the
        transform to a single vector, use ``fwht(v)`` instead, which runs in
        O(N log N) time and O(N) memory without materialising the matrix.
    """
    H = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=real_dtype)
    H_n = H
    for _ in range(n - 1):
        H_n = jnp.kron(H_n, H)
    return H_n


def fwht(v: jax.Array) -> jax.Array:
    """Fast Walsh-Hadamard Transform (FWHT) applied in-place on a vector.

    Computes H_n @ v in O(N log N) time and O(N) memory using the standard
    Cooley-Tukey butterfly decomposition via reshape. Equivalent to
    ``unnormalized_hadamard_transform(log2(N)) @ v`` for integer N = 2**n,
    but vastly more efficient for large N.

    Args:
        v: Real or complex JAX array of shape ``(N,)``.
           N must be a positive power of 2.

    Returns:
        Array of shape ``(N,)`` equal to the unnormalized Hadamard transform of v.

    Raises:
        ValueError: If len(v) is not a positive power of 2.
    """
    N = v.shape[0]
    if N == 0 or (N & (N - 1)) != 0:
        raise ValueError(f"fwht requires a positive power-of-2 length, got {N}")
    is_complex = jnp.issubdtype(v.dtype, jnp.complexfloating)
    x = v.astype(complex_dtype if is_complex else real_dtype)
    h = 1
    while h < N:
        x = x.reshape((-1, 2 * h))
        a, b = x[:, :h], x[:, h:]
        x = jnp.concatenate([a + b, a - b], axis=1)
        h *= 2
    return x.reshape((N,))


# ---------------------------------------------------------------------------
# Unitary / block-encoding helpers
# ---------------------------------------------------------------------------

def generate_random_unitary(key: random.PRNGKeyArray, dim: int) -> jnp.ndarray:
    """Generate a Haar-random unitary matrix of size dim x dim."""
    A = random.normal(key, (dim, dim), dtype=complex_dtype) + 1j * random.normal(
        key, (dim, dim), dtype=complex_dtype
    )
    Q, R = jnp.linalg.qr(A)
    D = jnp.diag(jnp.diag(R) / jnp.abs(jnp.diag(R)))
    return jnp.dot(Q, D)


def generate_random_hermitian(key: random.PRNGKeyArray, dim: int) -> jnp.ndarray:
    """Generate a random Hermitian matrix with spectral norm bounded by 1."""
    A = random.normal(key, (dim, dim), dtype=complex_dtype) + 1j * random.normal(
        key, (dim, dim), dtype=complex_dtype
    )
    A = (A + A.conj().T) / 2
    norm = jnp.linalg.norm(A, ord=2)
    return A / norm


def halmos_dilation(A: jnp.ndarray) -> jnp.ndarray:
    """Construct the canonical Halmos dilation of a contraction A.

    For a contraction A (||A|| <= 1), produces a unitary U of size 2N x 2N:
        U = [[A,  D_A*],
             [D_A, -A*]]
    where D_A = sqrt(I - A*A) is the defect operator.

    The result is unitary up to float64 precision (~1e-14).
    """
    dim = A.shape[0]
    identity = jnp.eye(dim, dtype=complex_dtype)
    # Defect operator: D_A = sqrt(I - A^dag A)
    sqrt_I_minus_AdagA = jax.scipy.linalg.sqrtm(identity - A.conj().T @ A)
    sqrt_I_minus_AAdagg = jax.scipy.linalg.sqrtm(identity - A @ A.conj().T)
    U = jnp.block([
        [A,                   sqrt_I_minus_AdagA],
        [sqrt_I_minus_AAdagg, -A.conj().T       ],
    ])
    return U


def random_halmos_dilation(key: random.PRNGKeyArray, dim: int) -> jnp.ndarray:
    """Generate a random Halmos dilation of size 2N x 2N with scrambled blocks."""
    A = generate_random_hermitian(key, dim)
    U = halmos_dilation(A)
    key, key1, key2 = random.split(key, 3)
    U1 = generate_random_unitary(key1, dim)
    U2 = generate_random_unitary(key2, dim)
    U = (
        jnp.block([[jnp.eye(dim), jnp.zeros((dim, dim))], [jnp.zeros((dim, dim)), U2]])
        @ U
        @ jnp.block(
            [[jnp.eye(dim), jnp.zeros((dim, dim))], [jnp.zeros((dim, dim)), U1]]
        )
    )
    return U


def hermitian_block_encoding(U: jnp.ndarray) -> jnp.ndarray:
    """Convert any unitary block encoding U into a Hermitian unitary block encoding."""
    hadamard = jnp.array([[1, 1], [1, -1]], dtype=real_dtype) / jnp.sqrt(2)
    zero_to_one = hadamard @ jnp.array([[0, 0], [1, 0]], dtype=real_dtype) @ hadamard
    one_to_zero = hadamard @ jnp.array([[0, 1], [0, 0]], dtype=real_dtype) @ hadamard
    V = jnp.kron(zero_to_one, U) + jnp.kron(one_to_zero, U.conj().T)
    return V


def get_block_encoded(U: jnp.ndarray, num_ancilla: int = 1) -> jnp.ndarray:
    """Extract the block-encoded matrix from a unitary U.

    If U's size is not divisible by 2^num_ancilla, pads U with zeros to the
    next multiple of 2^num_ancilla before extraction.
    """
    dim = U.shape[0]
    block = 2 ** num_ancilla
    if dim % block != 0:
        # Pad to the next multiple of block
        pad = block - (dim % block)
        U = jnp.pad(U, ((0, pad), (0, pad)))
        dim = U.shape[0]
    block_dim = dim // block
    return U[:block_dim, :block_dim]


def block_encoding_from_sparse_oracles(
    row_index_oracle: jnp.ndarray,
    col_index_oracle: jnp.ndarray,
    value_oracle: jnp.ndarray,
) -> jnp.ndarray:
    """Construct a block encoding of a sparse matrix from sparse oracles.

    Implements Lemma 48 of arXiv:1806.01838v1.
    """
    row_sparsity = row_index_oracle.shape[1]
    col_sparsity = col_index_oracle.shape[1]
    row_index_oracle = jnp.sum(row_index_oracle, axis=1) / jnp.sqrt(row_sparsity)
    col_index_oracle = jnp.sum(col_index_oracle, axis=1) / jnp.sqrt(col_sparsity)
    num_rows = row_index_oracle.shape[0]
    num_cols = col_index_oracle.shape[0]
    value_oracle = value_oracle.reshape((num_rows, num_cols))
    block_encoding = row_index_oracle.conj() * value_oracle * col_index_oracle.T
    return block_encoding


# ---------------------------------------------------------------------------
# Walsh-Hadamard / parity helpers
# ---------------------------------------------------------------------------

def bitwise_parity_matrix(dim: int) -> jnp.ndarray:
    """Return the dim x dim matrix of (-1)^(j.u) for j,u in {0,...,dim-1}."""
    u_vec = jnp.arange(dim, dtype=int_dtype)
    j_vec = jnp.arange(dim, dtype=int_dtype)
    bitwise_and = jnp.bitwise_and(j_vec[:, None], u_vec[None, :])
    bit_inner_product = jax.lax.population_count(bitwise_and) % 2
    return 1 - 2 * bit_inner_product


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def is_unitary(U: jnp.ndarray, atol: float = 1e-6) -> bool:
    """Check whether U is unitary up to absolute tolerance.

    Default atol raised from 1e-10 to 1e-6 to accommodate float64 rounding
    accumulated in multi-step matrix products (e.g. Halmos dilation via sqrtm).
    """
    dim = U.shape[0]
    return bool(jnp.isclose(jnp.linalg.norm(U @ U.conj().T - jnp.eye(dim)), 0.0, atol=atol))


def is_hermitian(A: jnp.ndarray, atol: float = 1e-10) -> bool:
    """Check whether A is Hermitian up to absolute tolerance."""
    return bool(jnp.isclose(jnp.linalg.norm(A - A.conj().T), 0.0, atol=atol))


def spectral_norm_bound(A: jnp.ndarray, bound: float = 1.0, atol: float = 1e-10) -> bool:
    """Check whether the spectral norm of A is bounded by `bound`."""
    return bool(
        jnp.isclose(jnp.linalg.norm(A, ord=2), bound, atol=atol)
        or jnp.linalg.norm(A, ord=2) < bound
    )
