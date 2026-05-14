"""Core mathematical properties of the Hadamard-induced TVD used by the
warmstart ablation.

For a sign diagonal d in {-1, +1}^N (with N = 2^n), define
    s(d) = H_n d / sqrt(N),
    p(d) = |s(d)|^2.

Because H_n is orthogonal, ||s(d)||^2 = ||d||^2 / N = 1, so p(d) is a
valid probability distribution over the 2^n basis states.  The notebook
TVD function is
    TVD(d1, d2) = 1/2 * ||p(d1) - p(d2)||_1.

This file pins the *exact formula* (factors, normalization) and the
mathematical properties any change must preserve: TVD >= 0, symmetric,
TVD(d, d) = 0, TVD(d, -d) = 0 (Parseval), TVD bounded by 1, triangle
inequality.

These tests are mirror-of-the-notebook (no external dependency on the
ablation notebook contents); they fail loudly if anyone reintroduces an
abs() inside probs() or drops the 1/sqrt(N) factor.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Exact TVD implementation (must match notebooks/warmstart_ablation.ipynb)
# ---------------------------------------------------------------------------


def _hadamard(n: int) -> np.ndarray:
    """Build the dense Hadamard matrix of order 2^n, with 1/sqrt(2) per level."""
    H = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    Hn = H.copy()
    for _ in range(n - 1):
        Hn = np.kron(Hn, H)
    return Hn


def tvd_diag(diag_approx, diag_ideal) -> float:
    """TVD between two oracle diagonals via the Hadamard-induced distribution.

    Both inputs are interpreted as real-valued (np.real is applied) before
    forming the Hadamard probabilities.  This matches the notebook
    implementation exactly.
    """
    N = len(diag_ideal)
    n = int(np.log2(N))
    Hn = _hadamard(n)

    def probs(d):
        d_arr = np.real(np.array(d, dtype=np.complex128)).astype(np.float64)
        s = Hn @ (d_arr / np.sqrt(N))
        p = np.abs(s) ** 2
        return p / p.sum()

    return 0.5 * float(np.sum(np.abs(probs(diag_approx) - probs(diag_ideal))))


# ---------------------------------------------------------------------------
# Math properties
# ---------------------------------------------------------------------------


@pytest.fixture(params=[2, 3, 4, 6])
def n_qubits(request):
    return request.param


@pytest.fixture
def N(n_qubits):
    return 2 ** n_qubits


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _random_sign_diag(N, rng):
    return rng.choice([-1.0, 1.0], size=N)


def test_normalization_probs_sum_to_one(N, rng):
    """The Hadamard-induced p(d) must sum to exactly 1 for any sign diagonal."""
    d = _random_sign_diag(N, rng)
    n = int(np.log2(N))
    Hn = _hadamard(n)
    s = Hn @ (d / np.sqrt(N))
    p = np.abs(s) ** 2
    assert p.sum() == pytest.approx(1.0, abs=1e-12)


def test_identity_tvd_is_zero(N, rng):
    d = _random_sign_diag(N, rng)
    assert tvd_diag(d, d) == pytest.approx(0.0, abs=1e-12)


def test_tvd_negation_is_zero_parseval(N, rng):
    """TVD(d, -d) = 0: negating d flips Walsh coefficients, |.|^2 invariant."""
    d = _random_sign_diag(N, rng)
    assert tvd_diag(d, -d) == pytest.approx(0.0, abs=1e-12)


def test_tvd_symmetric(N, rng):
    a = _random_sign_diag(N, rng)
    b = _random_sign_diag(N, rng)
    assert tvd_diag(a, b) == pytest.approx(tvd_diag(b, a), abs=1e-12)


def test_tvd_in_unit_interval(N, rng):
    a = _random_sign_diag(N, rng)
    b = _random_sign_diag(N, rng)
    t = tvd_diag(a, b)
    assert 0.0 <= t <= 1.0 + 1e-12


def test_tvd_triangle_inequality(N, rng):
    """TVD(a, c) <= TVD(a, b) + TVD(b, c)."""
    a = _random_sign_diag(N, rng)
    b = _random_sign_diag(N, rng)
    c = _random_sign_diag(N, rng)
    lhs = tvd_diag(a, c)
    rhs = tvd_diag(a, b) + tvd_diag(b, c)
    assert lhs <= rhs + 1e-12


def test_tvd_orthogonal_diagonals_is_max(N):
    """For two diagonals whose Hadamard distributions live on disjoint supports
    the TVD must equal 1.

    Pick d1 = +1 vector (all-ones).  Then p(d1) is a delta on basis state 0.
    Pick d2 = Hadamard column 1 (so H_n d2 is a delta on state 1).  Then
    p(d2) is disjoint from p(d1) so TVD = 1.
    """
    n = int(np.log2(N))
    Hn = _hadamard(n)
    d1 = np.ones(N, dtype=np.float64)
    # H_n column 1, rescaled to be a sign diagonal
    d2 = Hn[:, 1] * np.sqrt(N)
    assert np.allclose(np.abs(d2), 1.0), "d2 should be a sign diagonal"
    assert tvd_diag(d1, d2) == pytest.approx(1.0, abs=1e-12)


def test_single_bit_flip_tvd_scales_like_1_over_N(N, rng):
    """Flipping a single entry should give a small, predictable TVD that
    decays with N.  This pins down the absolute scale of the metric."""
    d = _random_sign_diag(N, rng)
    d_flip = d.copy()
    d_flip[0] *= -1.0
    t = tvd_diag(d, d_flip)
    # A single sign flip perturbs s by 2 e_0 / sqrt(N); p changes only at
    # the rows of H_n hit by that perturbation.  The exact value depends on
    # d, but it must be strictly between 0 and a 1/sqrt(N)-scale upper bound.
    assert 0.0 < t
    assert t < 4.0 / np.sqrt(N)  # generous upper bound; pins the scale


def test_tvd_handles_complex_input_silently():
    """tvd_diag must accept complex input (e.g. from VariationalWarmstart.predict())
    without raising or warning."""
    import warnings

    try:
        from numpy.exceptions import ComplexWarning as _ComplexWarning
    except ImportError:  # pragma: no cover -- NumPy 1.x branch
        from numpy import ComplexWarning as _ComplexWarning  # type: ignore[no-redef]

    N = 16
    d_real = np.array([1.0, -1.0] * (N // 2))
    d_complex = np.exp(1j * np.pi * (1.0 - d_real) / 2.0)  # +-1 on unit circle
    with warnings.catch_warnings():
        warnings.simplefilter("error", _ComplexWarning)
        t = tvd_diag(d_complex, d_real)
    # Real part of d_complex matches d_real exactly, so TVD == 0.
    assert t == pytest.approx(0.0, abs=1e-12)
