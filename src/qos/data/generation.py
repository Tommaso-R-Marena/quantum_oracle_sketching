"""Data sampling interfaces for vectors, matrices, Boolean functions, and k-Forrelation.

# Copyright (c) 2026 Tommaso R. Marena. MIT License.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import random

from qos.config import int_dtype, real_dtype
from qos.utils.numerical import fwht  # vector transform, not the matrix builder

if TYPE_CHECKING:
    from jax import random as jax_random

__all__ = ["matrix_data", "vector_data", "boolean_data", "k_forrelation_data"]


class matrix_data:
    def __init__(self, matrix: jax.Array) -> None:
        self.matrix = matrix.astype(real_dtype)
        self.shape = matrix.shape
        self.num_generated_samples: int = 0
        self._nz_rows, self._nz_cols = jnp.nonzero(self.matrix)
        self._nz_rows = self._nz_rows.astype(int_dtype)
        self._nz_cols = self._nz_cols.astype(int_dtype)
        self._nnz = int(self._nz_rows.shape[0])

    def get_matrix_element_data(
        self,
        key: jax_random.PRNGKeyArray,
        num_samples: int,
        return_values: bool = True,
    ) -> tuple[jax.Array, ...]:
        self.num_generated_samples += num_samples
        sampled_indices = random.randint(key, shape=(num_samples,), minval=0, maxval=self._nnz, dtype=int_dtype)
        sampled_rows = self._nz_rows[sampled_indices]
        sampled_cols = self._nz_cols[sampled_indices]
        if return_values:
            sampled_values = self.matrix[sampled_rows, sampled_cols]
            return sampled_rows, sampled_cols, sampled_values
        return sampled_rows, sampled_cols

    def get_row_data(self, key: jax_random.PRNGKeyArray, num_samples: int) -> tuple[jax.Array, jax.Array]:
        num_rows = self.shape[0]
        sampled_rows = random.choice(key, jnp.arange(num_rows, dtype=int_dtype), shape=(num_samples,), replace=True)
        sampled_row_vectors = self.matrix[sampled_rows]
        self.num_generated_samples += num_samples
        return sampled_rows, sampled_row_vectors


class vector_data:
    def __init__(self, vector: jax.Array) -> None:
        self.vector = vector.astype(real_dtype)
        self.length = int(vector.shape[0])
        self.num_generated_samples: int = 0

    def get_data(self, key: jax_random.PRNGKeyArray, num_samples: int) -> tuple[jax.Array, jax.Array]:
        sampled_indices = random.choice(key, jnp.arange(self.length, dtype=int_dtype), shape=(num_samples,), replace=True).astype(int_dtype)
        sampled_values = self.vector[sampled_indices]
        self.num_generated_samples += num_samples
        return sampled_indices, sampled_values


class boolean_data:
    def __init__(self, truth_table: jax.Array) -> None:
        self.truth_table = truth_table.astype(int_dtype)
        self.length = int(truth_table.shape[0])
        self.num_generated_samples: int = 0

    def get_data(self, key: jax_random.PRNGKeyArray, num_samples: int) -> tuple[jax.Array, jax.Array]:
        sampled_indices = random.choice(key, jnp.arange(self.length, dtype=int_dtype), shape=(num_samples,), replace=True).astype(int_dtype)
        sampled_values = self.truth_table[sampled_indices]
        self.num_generated_samples += num_samples
        return sampled_indices, sampled_values


class k_forrelation_data:
    """Data generation for k-Forrelation oracle property estimation."""

    def __init__(
        self,
        n: int,
        k: int,
        key: jax.Array,
        noise_level: float = 0.0,
    ):
        self.n = n
        self.k = k
        self.key = key
        self.noise_level = noise_level
        self.dim = 2**n

    def sample_functions(self, key: jax.Array) -> list[jax.Array]:
        """Sample k random Boolean function layers as full \u00b11 arrays.

        Args:
            key: JAX PRNG key. Each layer is a full array of shape (2**n,).

        Returns:
            List of k arrays each of shape ``(2**n,)`` with \u00b11 entries.

        Mathematical note:
            Each layer f_i: {0,1}^n -> {-1, +1} is drawn uniformly at random.
            The k-Forrelation value is the nested Hadamard inner product of
            these k layers (Aaronson-Ambainis 2015).
        """
        keys = random.split(key, self.k)
        funcs = [
            random.choice(keys[i], jnp.array([-1.0, 1.0]), shape=(self.dim,))
            for i in range(self.k)
        ]
        if self.noise_level > 0:
            noise_keys = random.split(keys[-1], self.k)
            funcs = [
                jnp.where(
                    random.bernoulli(noise_keys[i], p=self.noise_level, shape=(self.dim,)),
                    -funcs[i], funcs[i]
                )
                for i in range(self.k)
            ]
        return funcs

    def compute_exact_forrelation(self, functions: list[jax.Array]) -> float:
        """Compute exact ``k``-Forrelation value using the fast Walsh-Hadamard transform.

        Args:
            functions: List of ``k`` arrays, each shape ``(2**n,)`` with \u00b11 entries.

        Returns:
            Exact scalar ``Phi_k``.

        Mathematical note:
            Phi_k = mean(f_1 * H(f_2 * H(f_3 * ... * H(f_k)...))) / dim^(k-1)
            where H is the unnormalized WHT applied as a vector operation via fwht().
            Uses fwht() which runs in O(N log N) rather than O(N^2) matrix multiply.
        """
        assert len(functions) == self.k
        result = functions[-1].astype(real_dtype)
        for f in reversed(functions[:-1]):
            result = fwht(result) / self.dim  # O(N log N) vector transform
            result = f.astype(real_dtype) * result
        return float(jnp.mean(result))

    def quantum_query_algorithm(self, oracle_diag: jax.Array) -> float:
        """Simulate an O(1)-query Hadamard-test estimator for Forrelation.

        Args:
            oracle_diag: Diagonal phase oracle representation with shape ``(2**n,)``.

        Returns:
            Scalar estimate of Forrelation.
        """
        return float(jnp.real(jnp.mean(oracle_diag)))

    def classical_streaming_complexity(self, epsilon: float) -> int:
        """Return theoretical classical complexity lower bound.

        Args:
            epsilon: Additive error target.

        Returns:
            Integer lower bound ``ceil(N^(1-1/k) / epsilon^2)`` matching the
            Aaronson-Ambainis k-Forrelation separation.
        """
        exponent = 1.0 - 1.0 / self.k
        return int(jnp.ceil((self.dim ** exponent) / (epsilon ** 2)))
