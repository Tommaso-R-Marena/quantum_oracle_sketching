"""Unit tests for quantum oracle sketching core functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax import random

from qos.config import complex_dtype, int_dtype, real_dtype
from qos.core.oracle_sketch import (
    q_oracle_sketch_boolean,
    q_oracle_sketch_matrix_element,
    q_oracle_sketch_matrix_index,
    q_oracle_sketch_matrix_row_index,
)
from qos.core.state_sketch import q_state_sketch, q_state_sketch_flat
from qos.utils.numerical import (
    block_encoding_from_sparse_oracles,
    is_hermitian,
    is_unitary,
    random_sparse_matrix,
    random_sparse_matrix_given_row_sparsity,
)


@pytest.fixture
def key():
    return random.PRNGKey(42)


class TestStateSketchFlat:
    def test_norm_preservation(self, key):
        N = 1000
        num_samples = 1_000_000
        x = random.randint(key, (N,), minval=0, maxval=2, dtype=int_dtype) * 2 - 1
        state, _ = q_state_sketch_flat(x, num_samples)
        norm_sq = float(jnp.linalg.norm(state) ** 2)
        assert pytest.approx(1.0, abs=1e-2) == norm_sq

    def test_reconstruction(self, key):
        N = 1000
        num_samples = 10_000_000
        x = random.randint(key, (N,), minval=0, maxval=2, dtype=int_dtype) * 2 - 1
        state, _ = q_state_sketch_flat(x, num_samples)
        recon_error = float(jnp.linalg.norm(state - x / jnp.sqrt(N)))
        assert recon_error < 1e-2


class TestStateSketchGeneral:
    def test_success_probability_and_error(self, key):
        """QSVT arcsin approximation on [-1, 1] with degree=40.
        The polynomial arcsin(x)/(pi/2) on the full domain [-1, 1] converges
        rapidly in QSP optimization (parity=1, odd function).
        """
        N = 128
        num_samples = 100_000
        key, subkey = random.split(key)
        v = random.normal(subkey, (N,), dtype=real_dtype)
        v = v / jnp.linalg.norm(v)

        state, _ = q_state_sketch(v, key, num_samples, degree=40)
        prob = float(jnp.linalg.norm(state) ** 2)
        assert prob > 0.5

        error = float(jnp.linalg.norm(v - state / jnp.linalg.norm(state)))
        assert error < 1e-1


class TestOracleSketchBoolean:
    def test_phase_oracle(self, key):
        N = 1000
        num_samples = 10_000_000
        f = random.randint(key, (N,), minval=0, maxval=2, dtype=int_dtype)
        diag, _ = q_oracle_sketch_boolean(f, num_samples)
        target = jnp.exp(1j * jnp.pi * f)
        error = float(jnp.max(jnp.abs(diag - target)))
        assert error < 1e-1


class TestOracleSketchMatrixElement:
    def test_element_oracle(self, key):
        N1, N2 = 1000, 10000
        nnz = 3000
        num_samples = 1_000_000
        A = random_sparse_matrix(key, (N1, N2), nnz)
        diag, _ = q_oracle_sketch_matrix_element(A, num_samples)
        target = A.reshape(N1 * N2)
        error = float(jnp.max(jnp.abs(diag - target)))
        assert error < 1e-1


class TestOracleSketchMatrixRowIndex:
    def test_index_reconstruction(self, key):
        dim1, dim2 = 100, 1000
        nnz = dim2 * 3
        num_samples = 10_000_000
        A = random_sparse_matrix(key, (dim1, dim2), nnz)
        row_counts = jnp.sum(A != 0, axis=1)
        row_sparsity = int(jnp.max(row_counts))

        index_oracle, _ = q_oracle_sketch_matrix_row_index(A, num_samples)
        col_mask = A != 0
        col_indices = jnp.arange(dim2)
        expected_cols = jnp.where(col_mask, col_indices, dim2)
        expected_cols = jnp.sort(expected_cols, axis=1)[:, :row_sparsity]

        pred = jnp.argmax(jnp.abs(index_oracle), axis=-1)
        valid = jnp.arange(row_sparsity)[None, :] < row_counts[:, None]
        assert jnp.all((pred == expected_cols) | ~valid)


class TestOracleSketchMatrixIndex:
    def test_index_reconstruction_with_rank(self, key):
        dim1, dim2 = 100, 10
        nnz = dim1 * 3
        num_samples = 10_000_000
        A = random_sparse_matrix(key, (dim1, dim2), nnz)
        row_counts = jnp.sum(A != 0, axis=1)
        row_sparsity = int(jnp.max(row_counts))

        index_oracle, _ = q_oracle_sketch_matrix_index(
            A, num_samples, axis=0, degree=101, scale=0.9999
        )
        col_mask = A != 0
        col_indices = jnp.arange(dim2)
        expected_cols = jnp.where(col_mask, col_indices, dim2)
        expected_cols = jnp.sort(expected_cols, axis=1)[:, :row_sparsity]

        pred = jnp.argmax(jnp.abs(index_oracle), axis=-1)
        valid = jnp.arange(row_sparsity)[None, :] < row_counts[:, None]
        assert jnp.all((pred == expected_cols) | ~valid)

        norm = jnp.linalg.norm(index_oracle, axis=-1, keepdims=True)
        normalized = index_oracle / jnp.where(norm == 0, 1.0, norm)
        pred_value = jnp.take_along_axis(normalized, pred[..., None], axis=-1)[..., 0]
        error = float(jnp.max(jnp.where(valid, jnp.abs(1.0 - pred_value), 0.0)))
        assert error < 1e-1


class TestFullBlockEncoding:
    def test_combined_block_encoding(self, key):
        dim1, dim2 = 100, 100
        nnz = dim2 * 3
        num_samples = 10_000_000
        A = random_sparse_matrix(key, (dim1, dim2), nnz)
        A = A / jnp.linalg.norm(A, ord=2)

        row_sparsity = int(jnp.max(jnp.count_nonzero(A, axis=1)))
        col_sparsity = int(jnp.max(jnp.count_nonzero(A, axis=0)))

        element_oracle, _ = q_oracle_sketch_matrix_element(A, num_samples)
        row_index_oracle, _ = q_oracle_sketch_matrix_row_index(A, num_samples)
        col_index_oracle, _ = q_oracle_sketch_matrix_index(
            A, num_samples, axis=1, degree=151, scale=0.9999
        )

        block_encoding = block_encoding_from_sparse_oracles(
            row_index_oracle, col_index_oracle, element_oracle
        )
        normalized = block_encoding / jnp.linalg.norm(block_encoding, ord=2)

        error_spec = float(
            jnp.linalg.norm(normalized - A, ord=2) / jnp.linalg.norm(A, ord=2)
        )
        assert error_spec < 1e-1
