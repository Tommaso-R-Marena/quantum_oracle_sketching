"""Tests for numerical utilities and block-encoding helpers."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax import random

from qos.utils.numerical import (
    generate_random_hermitian,
    generate_random_unitary,
    get_block_encoded,
    halmos_dilation,
    hermitian_block_encoding,
    is_hermitian,
    is_unitary,
    laplacian_matrix,
    random_flat_vector,
    random_sparse_matrix,
    random_sparse_matrix_given_row_sparsity,
    random_unit_vector,
    spectral_norm_bound,
    unnormalized_hadamard_transform,
)


def test_unnormalized_hadamard():
    H2 = unnormalized_hadamard_transform(1)
    assert jnp.allclose(H2, jnp.array([[1, 1], [1, -1]]))

    H4 = unnormalized_hadamard_transform(2)
    expected = jnp.array([[1, 1, 1, 1],
                         [1, -1, 1, -1],
                         [1, 1, -1, -1],
                         [1, -1, -1, 1]])
    assert jnp.allclose(H4, expected)


def test_random_unit_vector():
    key = random.PRNGKey(0)
    v = random_unit_vector(key, 100)
    assert v.shape == (100,)
    assert jnp.isclose(jnp.linalg.norm(v), 1.0)


def test_random_flat_vector():
    key = random.PRNGKey(0)
    v = random_flat_vector(key, 100)
    assert jnp.all((v == 1) | (v == -1))


def test_random_sparse_matrix():
    key = random.PRNGKey(0)
    A = random_sparse_matrix(key, (10, 20), nnz=15)
    assert A.shape == (10, 20)
    assert jnp.count_nonzero(A) <= 15
    assert spectral_norm_bound(A, bound=1.0)


def test_random_sparse_matrix_row_sparsity():
    key = random.PRNGKey(0)
    A = random_sparse_matrix_given_row_sparsity(key, (10, 20), row_sparsity=3)
    assert A.shape == (10, 20)
    assert jnp.all(jnp.sum(A != 0, axis=1) == 3)


def test_laplacian_matrix():
    L = laplacian_matrix(5)
    assert L.shape == (5, 5)
    assert is_hermitian(L)
    assert spectral_norm_bound(L, bound=1.0)


def test_generate_random_unitary():
    key = random.PRNGKey(0)
    U = generate_random_unitary(key, 10)
    assert is_unitary(U)


def test_generate_random_hermitian():
    key = random.PRNGKey(0)
    H = generate_random_hermitian(key, 10)
    assert is_hermitian(H)
    assert spectral_norm_bound(H, bound=1.0)


def test_halmos_dilation():
    """Halmos dilation of a Hermitian contraction is unitary;
    the top-left N x N block recovers A.
    Note: the dilation is NOT Hermitian in general -- only the block A is.
    """
    key = random.PRNGKey(0)
    A = generate_random_hermitian(key, 5)
    U = halmos_dilation(A)
    assert is_unitary(U)
    # The top-left block is A by construction.
    extracted = get_block_encoded(U, num_ancilla=1)
    assert jnp.allclose(extracted, A, atol=1e-5)


def test_hermitian_block_encoding():
    """hermitian_block_encoding(U) embeds (U + U†)/2 in the top-left N x N
    block of a 2N x 2N unitary V.  Extract with num_ancilla=1.
    """
    key = random.PRNGKey(0)
    U = generate_random_unitary(key, 5)
    V = hermitian_block_encoding(U)
    assert is_unitary(V)
    assert is_hermitian(V)
    # V is 10x10; num_ancilla=1 extracts the top-left 5x5 block = (U+U†)/2.
    A = get_block_encoded(V, num_ancilla=1)
    assert jnp.allclose(A, (U + U.conj().T) / 2, atol=1e-5)


def test_get_block_encoded_raises():
    """get_block_encoded raises ValueError when size is not divisible."""
    U = jnp.eye(5)
    with pytest.raises(ValueError, match="divisible"):
        get_block_encoded(U, num_ancilla=1)
