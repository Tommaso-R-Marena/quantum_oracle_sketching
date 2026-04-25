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
    """Generate a batch of random flat vectors with components ±1."""
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
    """Generate a batch of unit spectral-norm random sparse matrices.

    Note:
        Matrix element magnitudes scale as O(1/sqrt(nnz)).
    """
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
    """Generate random sparse matrices with roughly uniform non-zero magnitudes.

    The magnitude should typically scale ~sqrt(sqrt(dim1 * dim2) / nnz)
    to ensure the spectral norm is bounded by one.
    """
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
    """Generate random sparse matrices with given row sparsity.

    All non-zero elements have roughly the same magnitude scale.
    """
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
    """Generate the normalized 1D chain Laplacian with spectral norm bounded by 1."""
    diagonals = jnp.ones([dim], dtype=real_dtype) * 2.0
    off_diagonals = jnp.ones([dim - 1], dtype=real_dtype) * (-1.0)

    L = jnp.diag(diagonals)
    L = L.at[jnp.arange(dim - 1), jnp.arange(1, dim)].set(off_diagonals)
    L = L.at[jnp.arange(1, dim), jnp.arange(dim - 1)].set(off_diagonals)

    return L / 2.0


# ---------------------------------------------------------------------------
# Hadamard transform
# ---------------------------------------------------------------------------

def unnormalized_hadamard_transform(n: int) -> jnp.ndarray:
    """Return the n-fold Kronecker power of the 2x2 Hadamard matrix [[1,1],[1,-1]]."""
    H = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=real_dtype)
    H_n = H
    for _ in range(n - 1):
        H_n = jnp.kron(H_n, H)
    return H_n


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
    """Construct the canonical Halmos dilation of a Hermitian matrix A.

    The result is a Hermitian unitary block encoding of A.
    Uses sqrtm for numerical stability.
    """
    dim = A.shape[0]
    identity = jnp.eye(dim, dtype=complex_dtype)
    # Numerically stable matrix square root via SVD-based approach when needed.
    sqrt_A = jax.scipy.linalg.sqrtm(identity - A @ A)
    U = jnp.block([[A, sqrt_A], [sqrt_A, -A]])
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
    """Convert any unitary block encoding U into a Hermitian unitary block encoding.

    Uses the construction in Appendix C of the QSVT paper (arXiv:2002.11649).
    Compatible with QOS because c^0U and c^1U† can be implemented simultaneously
    from the same samples.
    """
    hadamard = jnp.array([[1, 1], [1, -1]], dtype=real_dtype) / jnp.sqrt(2)
    zero_to_one = hadamard @ jnp.array([[0, 0], [1, 0]], dtype=real_dtype) @ hadamard
    one_to_zero = hadamard @ jnp.array([[0, 1], [0, 0]], dtype=real_dtype) @ hadamard
    V = jnp.kron(zero_to_one, U) + jnp.kron(one_to_zero, U.conj().T)
    return V


def get_block_encoded(U: jnp.ndarray, num_ancilla: int = 1) -> jnp.ndarray:
    """Extract the block-encoded matrix from a unitary U.

    Args:
        U: Unitary matrix of shape (2^num_ancilla * dim, 2^num_ancilla * dim).
        num_ancilla: Number of ancilla qubits.

    Returns:
        The top-left block of shape (dim, dim).
    """
    dim = U.shape[0]
    if dim % (2**num_ancilla) != 0:
        raise ValueError(
            f"Unitary size {dim} must be divisible by 2^{num_ancilla} = {2 ** num_ancilla}."
        )
    block_dim = dim // (2**num_ancilla)
    return U[:block_dim, :block_dim]


def block_encoding_from_sparse_oracles(
    row_index_oracle: jnp.ndarray,
    col_index_oracle: jnp.ndarray,
    value_oracle: jnp.ndarray,
) -> jnp.ndarray:
    """Construct a block encoding of a sparse matrix from sparse oracles.

    Implements Lemma 48 of arXiv:1806.01838v1.

    Args:
        row_index_oracle: shape (num_rows, row_sparsity, num_cols).
        col_index_oracle: shape (num_cols, col_sparsity, num_rows).
        value_oracle: shape (num_rows * num_cols,).

    Returns:
        Block-encoded matrix of shape (num_rows, num_cols), normalized by
        sqrt(row_sparsity * col_sparsity).
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
    """Return the dim x dim matrix of (-1)^(j.u) for j,u in {0,...,dim-1}.

    Efficiently computed using JAX population_count on bitwise AND.
    """
    u_vec = jnp.arange(dim, dtype=int_dtype)
    j_vec = jnp.arange(dim, dtype=int_dtype)
    bitwise_and = jnp.bitwise_and(j_vec[:, None], u_vec[None, :])
    bit_inner_product = jax.lax.population_count(bitwise_and) % 2
    return 1 - 2 * bit_inner_product


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def is_unitary(U: jnp.ndarray, atol: float = 1e-10) -> bool:
    """Check whether U is unitary up to absolute tolerance."""
    dim = U.shape[0]
    return bool(jnp.isclose(jnp.linalg.norm(U @ U.conj().T - jnp.eye(dim)), 0.0, atol=atol))


def is_hermitian(A: jnp.ndarray, atol: float = 1e-10) -> bool:
    """Check whether A is Hermitian up to absolute tolerance."""
    return bool(jnp.isclose(jnp.linalg.norm(A - A.conj().T), 0.0, atol=atol))


def spectral_norm_bound(A: jnp.ndarray, bound: float = 1.0, atol: float = 1e-10) -> bool:
    """Check whether the spectral norm of A is bounded by `bound`."""
    return bool(jnp.isclose(jnp.linalg.norm(A, ord=2), bound, atol=atol) or jnp.linalg.norm(A, ord=2) < bound)
