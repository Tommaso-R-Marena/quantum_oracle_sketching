"""Data sampling interfaces for vectors, matrices, and Boolean functions.

Provides ``matrix_data``, ``vector_data``, and ``boolean_data`` classes that
simulate classical data access by returning random uniform samples. The
``num_generated_samples`` attribute tracks the total sample complexity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import random

from qos.config import int_dtype, real_dtype

if TYPE_CHECKING:
    from jax import random as jax_random


class matrix_data:
    """Simulate access to a sparse matrix via uniform random sampling.

    Attributes:
        matrix: The target matrix (read-only).
        shape: Matrix dimensions.
        num_generated_samples: Cumulative number of samples drawn.
        _nz_rows: Row indices of non-zero elements.
        _nz_cols: Column indices of non-zero elements.
        _nnz: Total number of non-zero elements.
    """

    def __init__(self, matrix: jnp.ndarray) -> None:
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
    ) -> tuple[jnp.ndarray, ...]:
        """Sample non-zero matrix elements uniformly at random.

        Args:
            key: JAX PRNGKey.
            num_samples: Number of samples.
            return_values: Whether to return the sampled values.

        Returns:
            If ``return_values``: ``(rows, cols, values)``.
            Otherwise: ``(rows, cols)``.
        """
        self.num_generated_samples += num_samples
        sampled_indices = random.randint(
            key, shape=(num_samples,), minval=0, maxval=self._nnz, dtype=int_dtype
        )
        sampled_rows = self._nz_rows[sampled_indices]
        sampled_cols = self._nz_cols[sampled_indices]

        if return_values:
            sampled_values = self.matrix[sampled_rows, sampled_cols]
            return sampled_rows, sampled_cols, sampled_values
        return sampled_rows, sampled_cols

    def get_row_data(
        self,
        key: jax_random.PRNGKeyArray,
        num_samples: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample rows uniformly at random (with replacement).

        Args:
            key: JAX PRNGKey.
            num_samples: Number of row samples.

        Returns:
            ``(sampled_row_indices, sampled_row_vectors)``.
        """
        num_rows = self.shape[0]
        sampled_rows = random.choice(
            key,
            jnp.arange(num_rows, dtype=int_dtype),
            shape=(num_samples,),
            replace=True,
        )
        sampled_row_vectors = self.matrix[sampled_rows]
        self.num_generated_samples += num_samples
        return sampled_rows, sampled_row_vectors


class vector_data:
    """Simulate access to a vector via uniform random sampling.

    Attributes:
        vector: The target vector.
        length: Vector dimension.
        num_generated_samples: Cumulative sample count.
    """

    def __init__(self, vector: jnp.ndarray) -> None:
        self.vector = vector.astype(real_dtype)
        self.length = int(vector.shape[0])
        self.num_generated_samples: int = 0

    def get_data(
        self,
        key: jax_random.PRNGKeyArray,
        num_samples: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample vector components uniformly at random (with replacement).

        Args:
            key: JAX PRNGKey.
            num_samples: Number of samples.

        Returns:
            ``(sampled_indices, sampled_values)``.
        """
        sampled_indices = random.choice(
            key,
            jnp.arange(self.length, dtype=int_dtype),
            shape=(num_samples,),
            replace=True,
        ).astype(int_dtype)
        sampled_values = self.vector[sampled_indices]
        self.num_generated_samples += num_samples
        return sampled_indices, sampled_values


class boolean_data:
    """Simulate access to a Boolean function via uniform random query sampling.

    Attributes:
        truth_table: Boolean values (0 or 1).
        length: Support size.
        num_generated_samples: Cumulative query count.
    """

    def __init__(self, truth_table: jnp.ndarray) -> None:
        self.truth_table = truth_table.astype(int_dtype)
        self.length = int(truth_table.shape[0])
        self.num_generated_samples: int = 0

    def get_data(
        self,
        key: jax_random.PRNGKeyArray,
        num_samples: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample random queries to the Boolean function.

        Args:
            key: JAX PRNGKey.
            num_samples: Number of queries.

        Returns:
            ``(sampled_indices, sampled_values)``.
        """
        sampled_indices = random.choice(
            key,
            jnp.arange(self.length, dtype=int_dtype),
            shape=(num_samples,),
            replace=True,
        ).astype(int_dtype)
        sampled_values = self.truth_table[sampled_indices]
        self.num_generated_samples += num_samples
        return sampled_indices, sampled_values
