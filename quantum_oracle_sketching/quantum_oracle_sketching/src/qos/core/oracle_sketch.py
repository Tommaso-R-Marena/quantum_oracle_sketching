"""Quantum oracle sketching: implement phase and block-encoding oracles from samples.

This module provides the expected-unitary (deterministic) implementations of:
    - Boolean function phase oracles.
    - Sparse matrix element oracles.
    - Sparse matrix row-index oracles.
    - Sparse matrix column-index oracles (with rank register).

All functions are JAX-transformable and conservative (pessimistic) simulations of
the real-world random-channel performance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import random

from qos.config import DEFAULT_CONFIG, complex_dtype, int_dtype, real_dtype
from qos.utils.numerical import unnormalized_hadamard_transform

if TYPE_CHECKING:
    from jax import random as jax_random


def q_oracle_sketch_boolean(
    truth_table: jnp.ndarray,
    unit_num_samples: int,
) -> tuple[jnp.ndarray, int]:
    """Construct the quantum phase-oracle sketch of a Boolean function.

    Uses 0 ancilla qubits. Returns the diagonal of the phase oracle
    ``|x> -> (-1)^{f(x)} |x>``.

    Args:
        truth_table: Boolean values (0 or 1) of shape ``(dim,)``.
        unit_num_samples: Number of effective samples.

    Returns:
        ``(diag, num_samples)`` where ``diag`` has shape ``(dim,)``.
    """
    dim = truth_table.shape[0]
    prob = jnp.ones_like(truth_table, dtype=real_dtype) / dim
    t = jnp.pi * dim

    log_diag = jnp.log1p(prob * jnp.expm1(1j * t / unit_num_samples * truth_table))
    log_diag = unit_num_samples * log_diag
    diag = jnp.exp(log_diag)

    return diag, int(unit_num_samples)


def q_oracle_sketch_matrix_element(
    matrix: jnp.ndarray,
    unit_num_samples: int,
) -> tuple[jnp.ndarray, int]:
    """Construct a Hermitian block encoding of the sparse element oracle.

    Uses 1 ancilla qubit. Returns the diagonal of the top-left block encoding
    the oracle ``|i>|j> -> A_{ij} |i>|j>``.

    Args:
        matrix: Input matrix of shape ``(num_rows, num_cols)`` with entries in ``[-1, 1]``.
        unit_num_samples: Number of effective samples.

    Returns:
        ``(diag, num_samples)`` where ``diag`` has shape ``(num_rows * num_cols,)``.
    """
    dims = matrix.shape
    nnz = jnp.count_nonzero(matrix)
    if nnz == 0:
        raise ValueError("Matrix has no non-zero elements.")

    t = float(nnz)

    # Uniform probability over non-zero elements.
    prob = jnp.where(matrix != 0, 1.0 / nnz, 0.0).astype(real_dtype)

    # arcsin map for LCU compatibility.
    log_diag = jnp.log1p(
        prob * jnp.expm1(1j * t / unit_num_samples * jnp.arcsin(matrix))
    )
    log_diag = unit_num_samples * log_diag
    diag = jnp.exp(log_diag)

    diag = diag.reshape(dims[0] * dims[1])

    # LCU: sin(B) = (U - U†)/(2i)
    sin = (diag - diag.conj()) / (2j)

    return sin, int(unit_num_samples)


def q_oracle_sketch_matrix_row_index(
    matrix: jnp.ndarray,
    unit_num_samples: int,
) -> tuple[jnp.ndarray, int]:
    """Construct a block encoding of the sparse row-index oracle.

    Uses 2 ancilla qubits. The oracle maps
    ``|i>|k>|0> -> |i>|k>|j(i, k)>`` where ``j(i, k)`` is the column index of
    the k-th non-zero element in row ``i``.

    Args:
        matrix: Sparse matrix of shape ``(num_rows, num_cols)``.
        unit_num_samples: Number of effective samples.

    Returns:
        ``(oracle, num_samples)`` where ``oracle`` has shape
        ``(num_rows, sparsity, num_cols)``.
    """
    dims = matrix.shape
    row_counts = jnp.count_nonzero(matrix, axis=1)
    sparsity = int(jnp.max(row_counts))
    if sparsity == 0:
        raise ValueError("Matrix has no non-zero elements.")

    bitlength_col = int(jnp.ceil(jnp.log2(dims[1])))

    # Non-zero column indices per row, padded to row sparsity.
    nz_col_indices = jnp.argsort(matrix != 0, axis=1, descending=True)[:, :sparsity]

    bit_positions = jnp.arange(bitlength_col - 1, -1, -1)
    col_bits = (
        nz_col_indices[..., None] >> bit_positions
    ) & 1  # shape (num_rows, sparsity, bitlength_col)

    # Controlled phase oracle for each bit.
    t = jnp.pi * dims[0]
    prob = jnp.ones((dims[0],), dtype=real_dtype) / dims[0]

    log_diag = jnp.log1p(
        prob[:, None, None] * jnp.expm1(1j * t / unit_num_samples * col_bits)
    )
    log_diag = unit_num_samples * log_diag
    diag = jnp.exp(log_diag)

    controlled_diag = jnp.stack(
        [jnp.ones_like(diag), diag], axis=-1
    )  # shape (num_rows, sparsity, bitlength_col, 2)

    # Convert to XOR oracle via Hadamard sandwich.
    hadamard = jnp.array([[1, 1], [1, -1]], dtype=real_dtype) / jnp.sqrt(2)
    xor_oracle = jnp.einsum(
        "am,ijkm,mn->ijkan",
        hadamard,
        controlled_diag,
        hadamard,
    )  # shape (num_rows, sparsity, bitlength_col, 2, 2)

    # Tensor product over bits to get full index state.
    state = xor_oracle[:, :, 0, :, 0]  # shape (num_rows, sparsity, 2)
    for bit in range(1, bitlength_col):
        state = jnp.einsum(
            "ija,ijb->ijab",
            state,
            xor_oracle[:, :, bit, :, 0],
        )
        state = state.reshape(dims[0], sparsity, -1)
    state = state.reshape(dims[0], sparsity, -1)

    state = state[:, :, : dims[1]]  # truncate

    return state, int(unit_num_samples)


def q_oracle_sketch_matrix_index(
    matrix: jnp.ndarray,
    unit_num_samples: int,
    axis: int,
    degree: int = DEFAULT_CONFIG.sign_degree,
    scale: float = DEFAULT_CONFIG.sign_rescale,
    angle_set: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, int]:
    """Construct a block encoding of the sparse row/column index oracle (with rank register).

    Uses 2 ancilla qubits. Implements the cumulative-counter phase oracle followed
    by QSVT sign-function amplification to yield the binary threshold predicate.

    Args:
        matrix: Sparse matrix of shape ``(num_rows, num_cols)``.
        unit_num_samples: Number of effective samples per elementary gate.
        axis: ``0`` for row-index oracle, ``1`` for column-index oracle.
        degree: Odd polynomial degree for the sign-function QSVT approximation.
        scale: Target magnitude of the sign function.
        angle_set: Pre-computed QSVT angles. If ``None``, generated internally.

    Returns:
        ``(oracle, total_samples)`` where ``oracle`` has shape
        ``(num_rows, sparsity, num_cols)`` for ``axis=0`` or
        ``(num_cols, sparsity, num_rows)`` for ``axis=1``.
    """
    from qos.qsvt.angles import get_qsvt_angles_sign
    from qos.qsvt.transform import apply_qsvt_diag

    num_rows = matrix.shape[axis]
    bitlength_col = int(jnp.ceil(jnp.log2(matrix.shape[1 - axis])))
    orig_num_cols = matrix.shape[1 - axis]
    num_cols = 2**bitlength_col

    sparsity = int(jnp.max(jnp.count_nonzero(matrix, axis=1 - axis)))
    nnz = jnp.count_nonzero(matrix)
    if nnz == 0:
        raise ValueError("Matrix has no non-zero elements.")

    k_indices = jnp.arange(sparsity, dtype=int_dtype) + 1
    t = jnp.pi * nnz / (2 * sparsity + 1)
    k_phase_scale = jnp.pi / (2 * sparsity + 1)

    # Cumulative probability matrix (row=i, col=l): P[j<l].
    prob_matrix = matrix if axis == 0 else matrix.T
    prob = jnp.zeros_like(prob_matrix, dtype=real_dtype)
    prob = prob.at[prob_matrix != 0].set(1.0 / nnz)
    prob = jnp.pad(
        prob, ((0, 0), (0, num_cols - orig_num_cols)), mode="constant", constant_values=0.0
    )
    prob = jnp.cumsum(prob, axis=1) - prob  # exclusive cumulative sum

    # Expected phase gate.
    log_diag = jnp.log1p(prob * jnp.expm1(1j * t / unit_num_samples))
    log_diag = unit_num_samples * log_diag

    # Expand to include sparsity index k.
    log_diag = jnp.repeat(log_diag[:, None, :], sparsity, axis=1)
    log_diag = log_diag - 1j * (k_indices[None, :, None] - 0.5) * k_phase_scale

    diag = jnp.exp(log_diag).reshape(num_rows * sparsity * num_cols)

    # LCU: phase -> sin.
    sin = (diag - jnp.conj(diag)) / (2j)
    cos = (diag + jnp.conj(diag)) / 2
    block_encoding = jnp.stack(
        [jnp.stack([sin, cos], axis=0), jnp.stack([cos, -sin], axis=0)],
        axis=0,
    )  # shape (2, 2, num_rows * sparsity * num_cols)

    # QSVT sign function.
    if angle_set is None:
        threshold = jnp.pi / (4 * sparsity + 2) * DEFAULT_CONFIG.sign_threshold_factor
        angle_set, _ = get_qsvt_angles_sign(
            degree=degree, threshold=float(threshold), rescale=scale
        )
        angle_set = angle_set.astype(real_dtype)

    block_encoding = apply_qsvt_diag(
        block_encoding, num_ancilla=1, angle_set=angle_set
    )

    block_encoding = jnp.real(block_encoding[0, 0])
    block_encoding = block_encoding.reshape(num_rows, sparsity, num_cols)

    # XOR oracle via Hadamard sandwich.
    hadamard = jnp.array([[1, 1], [1, -1]], dtype=real_dtype) / jnp.sqrt(2)
    cont_block_encoding = jnp.stack(
        [jnp.ones_like(block_encoding), block_encoding], axis=-1
    )
    xor_oracle = jnp.einsum(
        "am,ijkm,mn->ijkan",
        hadamard,
        cont_block_encoding,
        hadamard,
    )

    # Binary search: apply SWAP_{l_t,o} X_{l_t} O X_{l_t} MSB-first.
    state_lo = jnp.zeros((num_cols, 2), dtype=real_dtype)
    state_lo = state_lo.at[0, 0].set(1.0)
    state_lo = jnp.tile(state_lo[None, None, :, :], (num_rows, sparsity, 1, 1))

    for bit in range(bitlength_col - 1, -1, -1):
        high = 1 << (bitlength_col - bit - 1)
        low = 1 << bit

        # X_{l_t}
        state_lo = state_lo.reshape(num_rows, sparsity, high, 2, low, 2)
        state_lo = state_lo[:, :, :, ::-1, :, :]
        state_lo = state_lo.reshape(num_rows, sparsity, num_cols, 2)

        # O
        state_lo = jnp.matvec(xor_oracle, state_lo)

        # X_{l_t}
        state_lo = state_lo.reshape(num_rows, sparsity, high, 2, low, 2)
        state_lo = state_lo[:, :, :, ::-1, :, :]
        state_lo = state_lo.reshape(num_rows, sparsity, num_cols, 2)

        # SWAP_{l_t, o}
        state_lo = state_lo.reshape(num_rows, sparsity, high, 2, low, 2)
        state_lo = state_lo.transpose(0, 1, 2, 5, 4, 3)
        state_lo = state_lo.reshape(num_rows, sparsity, num_cols, 2)

    # Truncate to original column size and extract index register.
    state_lo = state_lo[:, :, :orig_num_cols, 0]

    total_samples = unit_num_samples * (angle_set.shape[0] - 1) * bitlength_col
    return state_lo, int(total_samples)
