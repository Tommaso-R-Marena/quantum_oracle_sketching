"""Active random-sampling implementation of QOS.

This module assembles each oracle query with randomly sampled data points.
It is more direct and intuitive than the expected-unitary path, but the mixing
lemma implies that the Euclidean error is quadratically overestimated relative
to the real-world random-channel performance.

See https://arxiv.org/abs/2008.11751 for discussion of the mixing lemma in
quantum simulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import random

from qos.config import DEFAULT_CONFIG, complex_dtype, int_dtype, real_dtype
from qos.data.generation import boolean_data, matrix_data, vector_data
from qos.utils.numerical import bitwise_parity_matrix, unnormalized_hadamard_transform

if TYPE_CHECKING:
    from jax import random as jax_random


def q_state_sketch_flat(
    data: tuple[jnp.ndarray, jnp.ndarray],
    dim: int,
) -> jnp.ndarray:
    """Construct a flat-vector state sketch from actively sampled data.

    Args:
        data: Tuple ``(sampled_indices, sampled_values)`` where values are ±1.
        dim: Target dimension (must match the vector length).

    Returns:
        State vector of shape ``(dim,)``.
    """
    sampled_indices, sampled_values = data
    num_samples = sampled_indices.shape[0]

    t = jnp.pi * dim
    phase = jnp.zeros(dim, dtype=real_dtype)
    phase = phase.at[sampled_indices].add((1 - sampled_values) / 2)
    phase = phase * t / num_samples
    diag = jnp.exp(1j * phase)

    return diag / jnp.sqrt(dim)


def q_state_sketch_flat_unitary(
    data: tuple[jnp.ndarray, jnp.ndarray],
    dim: int,
) -> jnp.ndarray:
    """Return only the diagonal unitary (not applied to |+>) for the flat sketch.

    Args:
        data: Tuple ``(sampled_indices, sampled_values)`` with values in {+1,-1}.
        dim: Target dimension.

    Returns:
        Diagonal of the unitary, shape ``(dim,)``.
    """
    sampled_indices, sampled_values = data
    num_samples = sampled_indices.shape[0]

    t = jnp.pi * dim
    phase = jnp.zeros(dim, dtype=real_dtype)
    phase = phase.at[sampled_indices].add((1 - sampled_values) / 2)
    phase = phase * t / num_samples
    return jnp.exp(1j * phase)


def q_state_sketch(
    data: tuple[jnp.ndarray, jnp.ndarray],
    dim: int,
    norm: float,
    key: jax_random.PRNGKeyArray,
    degree: int = DEFAULT_CONFIG.arcsin_degree,
) -> jnp.ndarray:
    """Construct a general-vector state sketch from actively sampled data.

    Args:
        data: Tuple ``(sampled_indices, sampled_values)``.
        dim: Target dimension (must be a power of 2).
        norm: L2 norm of the original vector.
        key: JAX PRNGKey for random sign randomization.
        degree: Even polynomial degree for arcsin QSVT.

    Returns:
        State vector of shape ``(dim,)``.
    """
    from qos.qsvt.angles import get_qsvt_angles
    from qos.qsvt.transform import apply_qsvt_imperfect_diag

    sampled_indices, sampled_values = data
    num_samples = sampled_indices.shape[0]

    if num_samples % degree != 0:
        raise ValueError(
            f"num_samples ({num_samples}) must be divisible by degree ({degree})."
        )

    sampled_values = sampled_values / norm

    key, subkey = random.split(key)
    random_signs = random.choice(subkey, jnp.array([1.0, -1.0]), shape=(dim,))

    inner_prod_signs = bitwise_parity_matrix(dim)

    # Aggregate samples into degree groups.
    sampled_indices = sampled_indices.reshape(degree, -1)
    sampled_values = sampled_values.reshape(degree, -1)

    aggregated = jnp.zeros((degree, dim), dtype=real_dtype)
    aggregated = aggregated.at[jnp.arange(degree)[:, None], sampled_indices].add(
        sampled_values
    )
    aggregated = aggregated / (num_samples / degree)
    aggregated = aggregated * random_signs[None, :]

    t = dim / norm / 3.0
    contribution = (aggregated @ inner_prod_signs) * t

    # arcsin(x) / arcsin(1) QSVT.
    angle_set = get_qsvt_angles(
        func=lambda x: jnp.arcsin(x) / jnp.arcsin(1.0),
        degree=degree,
        rescale=1.0,
        cheb_domain=(-jnp.sin(1.0), jnp.sin(1.0)),
        ensure_bounded=False,
        parity=0,
    )

    sin = jnp.sin(contribution)
    cos = jnp.cos(contribution)
    block_encoding = jnp.stack([sin, cos, cos, -sin], axis=0).reshape(
        2, 2, degree, dim
    )
    block_encoding = block_encoding.transpose(2, 0, 1, 3)

    # Imperfect QSVT over the degree sample groups.
    block_encoding = apply_qsvt_imperfect_diag(
        block_encoding[:-1], num_ancilla=1, angle_set=angle_set
    )

    state = jnp.real(block_encoding[0, 0]) / jnp.sqrt(dim)

    hadamard = unnormalized_hadamard_transform(int(jnp.round(jnp.log2(dim))))
    state = hadamard @ state / jnp.sqrt(dim)
    state = random_signs * state

    return state


def q_oracle_sketch_boolean(
    data: tuple[jnp.ndarray, jnp.ndarray],
    dim: int,
) -> jnp.ndarray:
    """Construct a Boolean phase-oracle sketch from actively sampled data.

    Args:
        data: Tuple ``(sampled_indices, sampled_values)`` with values in {0,1}.
        dim: Support size of the Boolean function.

    Returns:
        Diagonal of the phase oracle, shape ``(dim,)``.
    """
    sampled_indices, sampled_values = data
    num_samples = sampled_indices.shape[0]

    t = jnp.pi * dim
    phase = jnp.zeros(dim, dtype=real_dtype)
    phase = phase.at[sampled_indices].add(sampled_values)
    phase = phase * t / num_samples
    return jnp.exp(1j * phase)


def q_oracle_sketch_matrix_element(
    data: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    dims: tuple[int, int],
    nnz: int,
) -> jnp.ndarray:
    """Construct a Hermitian block encoding of the sparse element oracle from samples.

    Args:
        data: Tuple ``(sampled_row_indices, sampled_col_indices, sampled_values)``.
        dims: ``(num_rows, num_cols)``.
        nnz: Number of non-zero elements.

    Returns:
        Diagonal of the top-left block, shape ``(num_rows * num_cols,)``.
    """
    sampled_row_indices, sampled_col_indices, sampled_values = data
    num_samples = sampled_row_indices.shape[0]

    t = float(nnz)

    phase = jnp.zeros((dims[0], dims[1]), dtype=real_dtype)
    phase = phase.at[sampled_row_indices, sampled_col_indices].add(
        jnp.arcsin(sampled_values)
    )
    phase = phase * t / num_samples
    phase = phase.reshape(dims[0] * dims[1])

    return jnp.sin(phase)


def q_oracle_sketch_matrix_row_index(
    data: tuple[jnp.ndarray, jnp.ndarray],
    dims: tuple[int, int],
    sparsity: int,
) -> jnp.ndarray:
    """Construct a block encoding of the sparse row-index oracle from random row samples.

    Args:
        data: Tuple ``(sampled_row_indices, sampled_row_vectors)``.
        dims: ``(num_rows, num_cols)``.
        sparsity: Maximum row sparsity.

    Returns:
        Oracle array of shape ``(num_rows, sparsity, num_cols)``.
    """
    sampled_row_indices, sampled_row_vectors = data
    num_samples = sampled_row_indices.shape[0]
    bitlength_col = int(jnp.ceil(jnp.log2(dims[1])))

    def _row_nonzero_indices(row_vector: jnp.ndarray) -> jnp.ndarray:
        return jnp.nonzero(
            row_vector != 0,
            size=sparsity,
            fill_value=2**bitlength_col - 1,
        )[0]

    sampled_nonzero_col_indices = jax.vmap(_row_nonzero_indices)(sampled_row_vectors)
    bit_positions = jnp.arange(bitlength_col - 1, -1, -1)
    sampled_bits = (sampled_nonzero_col_indices[..., None] >> bit_positions) & 1

    row_bits = jnp.zeros((dims[0], sparsity, bitlength_col), dtype=real_dtype)
    row_bits = row_bits.at[sampled_row_indices].add(sampled_bits)

    t = jnp.pi * dims[0]
    phase = row_bits * t / num_samples
    diag = jnp.exp(1j * phase)
    controlled_diag = jnp.stack([jnp.ones_like(diag), diag], axis=-1)

    hadamard = jnp.array([[1, 1], [1, -1]], dtype=real_dtype) / jnp.sqrt(2)
    xor_oracle = jnp.einsum(
        "am,ijkm,mn->ijkan",
        hadamard,
        controlled_diag,
        hadamard,
    )

    state = xor_oracle[:, :, 0, :, 0]
    for bit in range(1, bitlength_col):
        state = jnp.einsum("ija,ijb->ijab", state, xor_oracle[:, :, bit, :, 0])
        state = state.reshape(dims[0], sparsity, -1)
    state = state.reshape(dims[0], sparsity, -1)

    return state[:, :, : dims[1]]


def q_oracle_sketch_matrix_index(
    data_gen: matrix_data,
    key: jax_random.PRNGKeyArray,
    unit_sample_size: int,
    dims: tuple[int, int],
    axis: int,
    sparsity: int,
    nnz: int,
) -> jnp.ndarray:
    """Construct a block encoding of the sparse row/column index oracle (streaming QSVT).

    This is the active-sampling implementation that streams QSVT gates one at a
    time to avoid storing all ``degree`` imperfect instantiations in memory.

    Args:
        data_gen: ``matrix_data`` instance with ``_nz_rows``, ``_nz_cols``, ``_nnz``.
        key: JAX PRNGKey.
        unit_sample_size: Samples per elementary block-encoding gate.
        dims: ``(num_rows, num_cols)``.
        axis: ``0`` for row index, ``1`` for column index.
        sparsity: Maximum sparsity along the queried axis.
        nnz: Total non-zero elements.

    Returns:
        Oracle array of shape ``(dims[axis], sparsity, dims[1-axis])``.
    """
    from qos.qsvt.angles import get_qsvt_angles_sign
    from tqdm import tqdm

    num_rows = dims[axis]
    bitlength_col = int(jnp.ceil(jnp.log2(dims[1 - axis])))
    num_cols = 2**bitlength_col
    unit_sample_size = int(unit_sample_size)

    k_indices = jnp.arange(sparsity, dtype=real_dtype) + 1
    t = jnp.pi * nnz / (2 * sparsity + 1)
    k_phase_scale = jnp.pi / (2 * sparsity + 1)

    nz_rows = data_gen._nz_rows
    nz_cols = data_gen._nz_cols
    nnz_entries = data_gen._nnz

    def _assemble_phase(sample_key: jax_random.PRNGKeyArray) -> jnp.ndarray:
        phase = jnp.zeros((num_rows, num_cols), dtype=real_dtype)
        sampled_indices = random.randint(
            sample_key,
            shape=(unit_sample_size,),
            minval=0,
            maxval=nnz_entries,
            dtype=int_dtype,
        )
        sampled_rows = nz_rows[sampled_indices]
        sampled_cols = nz_cols[sampled_indices]
        if axis == 0:
            phase = phase.at[sampled_rows, sampled_cols].add(1)
        else:
            phase = phase.at[sampled_cols, sampled_rows].add(1)

        counts = phase
        phase = phase.cumsum(axis=1)
        phase = phase - counts
        phase = phase.astype(real_dtype) * (t / unit_sample_size)
        phase = jnp.repeat(phase[:, None, :], sparsity, axis=1)
        phase = phase - (k_indices[None, :, None] - 0.5) * k_phase_scale
        return phase.reshape(num_rows * sparsity * num_cols)

    _assemble_phase = jax.jit(_assemble_phase)

    threshold = jnp.pi / (4 * sparsity + 2)
    degree = DEFAULT_CONFIG.sign_degree
    angle_set, _ = get_qsvt_angles_sign(
        degree=degree, threshold=float(threshold), rescale=0.99
    )
    angle_set = angle_set.astype(real_dtype)

    hadamard = jnp.array([[1, 1], [1, -1]], dtype=real_dtype) / jnp.sqrt(2)

    state_lo = jnp.zeros((num_cols, 2), dtype=real_dtype)
    state_lo = state_lo.at[0, 0].set(1.0)
    state_lo = jnp.tile(state_lo[None, None, :, :], (num_rows, sparsity, 1, 1))

    dim = num_rows * sparsity * num_cols
    qsp_mask = jnp.array([1.0, -1.0], dtype=angle_set.dtype)
    phase0 = jnp.exp(1j * angle_set[0] * qsp_mask)
    global_phase = jnp.exp(1j * (-jnp.pi / 2) * angle_set.shape[0])

    for bit in tqdm(range(bitlength_col - 1, -1, -1), desc="Bits"):
        qsvt_circ = global_phase * jnp.diag(phase0)
        qsvt_circ = jnp.broadcast_to(qsvt_circ[:, :, None], (2, 2, dim))

        for m in range(degree):
            key, subkey = random.split(key)
            phase = _assemble_phase(subkey)
            sin = jnp.sin(phase)
            cos = jnp.cos(phase)
            u_gate = jnp.stack(
                [jnp.stack([sin, cos], axis=0), jnp.stack([cos, -sin], axis=0)],
                axis=0,
            )
            qsvt_circ = jnp.einsum("abm,bcm->acm", qsvt_circ, u_gate)
            phase_rot = jnp.exp(1j * angle_set[m + 1] * qsp_mask)
            qsvt_circ = qsvt_circ * phase_rot[None, :, None]

        data_gen.num_generated_samples += unit_sample_size * degree

        block_encoding = jnp.real(qsvt_circ[0, 0]).reshape(num_rows, sparsity, num_cols)
        cont_block_encoding = jnp.stack(
            [jnp.ones_like(block_encoding), block_encoding], axis=-1
        )
        xor_oracle = jnp.einsum(
            "am,ijkm,mn->ijkan",
            hadamard,
            cont_block_encoding,
            hadamard,
        )

        high = 1 << (bitlength_col - bit - 1)
        low = 1 << bit

        state_lo = state_lo.reshape(num_rows, sparsity, high, 2, low, 2)
        state_lo = state_lo[:, :, :, ::-1, :, :]
        state_lo = state_lo.reshape(num_rows, sparsity, num_cols, 2)

        state_lo = jnp.matvec(xor_oracle, state_lo)

        state_lo = state_lo.reshape(num_rows, sparsity, high, 2, low, 2)
        state_lo = state_lo[:, :, :, ::-1, :, :]
        state_lo = state_lo.reshape(num_rows, sparsity, num_cols, 2)

        state_lo = state_lo.reshape(num_rows, sparsity, high, 2, low, 2)
        state_lo = state_lo.transpose(0, 1, 2, 5, 4, 3)
        state_lo = state_lo.reshape(num_rows, sparsity, num_cols, 2)

    state_lo = state_lo[:, :, : dims[1 - axis], 0]
    return state_lo
