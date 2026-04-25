"""Tests for QSVT angle generation and transform application."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from qos.qsvt.angles import (
    get_qsvt_angles,
    get_qsvt_angles_inverse,
    get_qsvt_angles_sign,
)
from qos.qsvt.transform import apply_qsvt, apply_qsvt_diag, apply_qsvt_imperfect
from qos.utils.numerical import (
    get_block_encoded,
    halmos_dilation,
    hermitian_block_encoding,
    is_hermitian,
    is_unitary,
    random_halmos_dilation,
)


def test_get_qsvt_angles_even_polynomial():
    angle_set = get_qsvt_angles(
        func=lambda x: 2 * x**2 - 1,
        degree=10,
        rescale=0.9,
    )
    assert angle_set.ndim == 1
    assert angle_set.shape[0] > 0
    assert jnp.all(jnp.isfinite(angle_set))


def test_get_qsvt_angles_odd_polynomial():
    angle_set = get_qsvt_angles(
        func=lambda x: jnp.arcsin(jnp.sin(1.0) * x),
        degree=11,
        rescale=0.9,
        parity=1,
    )
    assert angle_set.ndim == 1
    assert jnp.all(jnp.isfinite(angle_set))


def test_apply_qsvt_identity_approximation():
    dim = 50
    key = random.PRNGKey(0)
    U = random_halmos_dilation(key, dim).astype(jnp.complex128)
    V = hermitian_block_encoding(U)
    A = get_block_encoded(V, num_ancilla=2)

    # Identity polynomial: f(x) = x (degree 1, odd).
    angle_set = get_qsvt_angles(
        func=lambda x: 0.9 * x,
        degree=1,
        rescale=0.9,
        parity=1,
    )

    V_qsvt = apply_qsvt(V, num_ancilla=2, angle_set=angle_set)
    A_qsvt = get_block_encoded(V_qsvt, num_ancilla=2)

    eigvals = jnp.linalg.eigvalsh(A)
    eigvals_qsvt = jnp.linalg.eigvalsh(A_qsvt)
    eigvals_target = jnp.sort(0.9 * eigvals)

    error = float(jnp.max(jnp.abs(eigvals_qsvt - eigvals_target)))
    assert error < 1e-4


def test_apply_qsvt_inverse():
    dim = 50
    kappa = 5.0
    epsilon = 0.05
    key = random.PRNGKey(42)
    U = random_halmos_dilation(key, dim).astype(jnp.complex128)
    V = hermitian_block_encoding(U)
    A = get_block_encoded(V, num_ancilla=2)

    angle_set, scale = get_qsvt_angles_inverse(kappa=kappa, epsilon=epsilon)
    V_qsvt = apply_qsvt(V, num_ancilla=2, angle_set=angle_set)
    A_qsvt = get_block_encoded(V_qsvt, num_ancilla=2)

    eigvals = jnp.linalg.eigvalsh(A)
    eigvals_qsvt = jnp.linalg.eigvalsh(A_qsvt)
    eigvals_target = jnp.sort(scale / eigvals)

    num_outliers = int(jnp.sum(jnp.abs(eigvals) < 1.0 / kappa))
    error = jnp.abs(eigvals_qsvt - eigvals_target)
    error = jnp.sort(error)[:-num_outliers]

    assert float(jnp.max(error)) < 10 * epsilon


def test_apply_qsvt_sign():
    dim = 50
    threshold = 0.1
    degree = 51
    scale = 0.9
    key = random.PRNGKey(42)
    U = random_halmos_dilation(key, dim).astype(jnp.complex128)
    V = hermitian_block_encoding(U)
    A = get_block_encoded(V, num_ancilla=2)

    angle_set, _ = get_qsvt_angles_sign(
        threshold=threshold, degree=degree, rescale=scale
    )
    V_qsvt = apply_qsvt(V, num_ancilla=2, angle_set=angle_set)
    A_qsvt = get_block_encoded(V_qsvt, num_ancilla=2)

    eigvals = jnp.linalg.eigvalsh(A)
    eigvals_qsvt = jnp.linalg.eigvalsh(A_qsvt)
    eigvals_target = jnp.sort(scale * jnp.sign(eigvals))

    num_outliers = int(jnp.sum(jnp.abs(eigvals) < 1.5 * threshold))
    error = jnp.abs(eigvals_qsvt - eigvals_target)
    error = jnp.sort(error)[:-num_outliers]

    assert float(jnp.max(error)) < 1e-2


def test_apply_qsvt_diag():
    dim = 100
    degree = 10
    rescale = 0.9

    angle_set = get_qsvt_angles(
        func=lambda x: 2 * x**2 - 1,
        degree=degree,
        rescale=rescale,
    )

    key = random.PRNGKey(0)
    phase = random.uniform(key, minval=0.0, maxval=2 * jnp.pi, shape=(dim,))
    sin = jnp.sin(phase)
    cos = jnp.cos(phase)
    V = jnp.array([[sin, cos], [cos, -sin]]).reshape(2, 2, dim)

    V_qsvt = apply_qsvt_diag(V, num_ancilla=1, angle_set=angle_set)
    A_qsvt = jnp.real(V_qsvt[0, 0])

    target = rescale * (2 * sin**2 - 1)
    error = float(jnp.max(jnp.abs(A_qsvt - target)))
    assert error < 1e-4


def test_apply_qsvt_imperfect_noise():
    dim = 50
    degree = 10
    rescale = 0.5
    noise_level = 0.001

    angle_set = get_qsvt_angles(
        func=lambda x: rescale * (2 * x**2 - 1),
        degree=degree,
        rescale=rescale,
    )

    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    U = random_halmos_dilation(subkey, dim).astype(jnp.complex128)
    A = get_block_encoded(U, num_ancilla=1)

    num_gates = angle_set.shape[0] - 1
    V_sequence = []
    for _ in range(num_gates):
        key, subkey = random.split(key)
        noise = random.normal(subkey, (U.shape[0],)) * noise_level
        noisy_phase = jnp.exp(1j * noise)
        U_noisy = U * noisy_phase[None, :]
        V_sequence.append(hermitian_block_encoding(U_noisy))
    V_sequence = jnp.stack(V_sequence, axis=0)

    V_qsvt = apply_qsvt_imperfect(V_sequence, num_ancilla=2, angle_set=angle_set)
    A_qsvt = get_block_encoded(V_qsvt, num_ancilla=2)

    eigvals = jnp.linalg.eigvalsh(A)
    eigvals_qsvt = jnp.linalg.eigvalsh(A_qsvt)
    eigvals_target = jnp.sort(rescale * (2 * eigvals**2 - 1))

    error = float(jnp.max(jnp.abs(eigvals_qsvt - eigvals_target)))
    assert error < 2.0 * noise_level or noise_level < 1e-5
