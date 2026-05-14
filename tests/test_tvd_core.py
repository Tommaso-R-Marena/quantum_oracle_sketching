"""Core mathematical properties of the two TVD-style metrics used in this
project. Both are useful; they measure different things, and earlier
versions of this codebase conflated them. This file pins each definition
separately so the conflation cannot reoccur.

## Metric 1: ``tvd_diag(d1, d2) = 0.5 * ||d1 - d2||_1 / N``

The raw L1 distance between the diagonal vectors themselves, rescaled so
that for ±1-valued diagonals the metric lies in ``[0, 1]``. Useful when
"these two oracle diagonals are close as vectors" is what you mean (e.g.
when reporting a sanity bound on the gate-by-gate error after sketching).
For sign diagonals, this satisfies ``tvd_diag(d, d) = 0`` and
``tvd_diag(d, -d) = 1`` (every entry differs by 2; rescaled by 1/(2N) ->
1).

## Metric 2: ``hadamard_distribution_tvd(d1, d2)``

The total variation distance between the **basis-state measurement
distributions** induced by Hadamard-transforming each diagonal:

    s_i  = (H_n d_i) / sqrt(N)
    p_i  = |s_i|^2  (a valid probability distribution because H_n is
                     orthogonal and the diagonal is sign-valued)
    TVD  = 0.5 * ||p_1 - p_2||_1

This is the right metric when downstream usage measures the prepared
state in the computational basis (which is how the
``q_oracle_sketch_boolean`` consumers exercise the diagonal). Crucially
it is **invariant under global negation**: ``|H_n(-d)|^2 = |H_n d|^2``,
so ``hadamard_distribution_tvd(d, -d) = 0``. That is the *correct
behaviour* for a measurement metric — a global phase / global sign is
unobservable — and earlier audits in this repository (commit ``219f459``)
already pinned this property. It is **not** the L1 metric; the rename
in this file makes the distinction explicit.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Reference implementations (must match scripts/verify_*.py exactly)
# ---------------------------------------------------------------------------


def tvd_diag(d1, d2) -> float:
    """Raw-diagonal TVD: ``0.5 * ||d1 - d2||_1 / N``.

    Both inputs are interpreted as real-valued; complex input is silently
    projected via ``np.real`` (matching the notebook's tolerance for
    ``VariationalWarmstart.predict()`` output).
    """
    a = np.real(np.asarray(d1, dtype=np.complex128)).astype(np.float64)
    b = np.real(np.asarray(d2, dtype=np.complex128)).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"tvd_diag: shape mismatch {a.shape} vs {b.shape}")
    N = a.shape[0]
    return 0.5 * float(np.sum(np.abs(a - b))) / float(N)


def _hadamard(n: int) -> np.ndarray:
    """Dense Hadamard matrix of order ``2**n`` with 1/sqrt(2) per level."""
    H = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    Hn = H.copy()
    for _ in range(n - 1):
        Hn = np.kron(Hn, H)
    return Hn


def hadamard_distribution_tvd(d_approx, d_ideal) -> float:
    """TVD between two diagonals' Hadamard-induced measurement distributions.

    See module docstring for the formula.  Real projection is applied to
    complex inputs (matching the warmstart ablation pipeline).
    """
    N = len(d_ideal)
    n = int(np.log2(N))
    Hn = _hadamard(n)

    def probs(d):
        d_arr = np.real(np.array(d, dtype=np.complex128)).astype(np.float64)
        s = Hn @ (d_arr / np.sqrt(N))
        p = np.abs(s) ** 2
        return p / p.sum()

    return 0.5 * float(np.sum(np.abs(probs(d_approx) - probs(d_ideal))))


# ---------------------------------------------------------------------------
# Fixtures
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


# ===========================================================================
# Metric 1: tvd_diag  (raw L1)
# ===========================================================================


class TestTvdDiag:
    """Properties of the raw-diagonal L1 metric ``tvd_diag``."""

    def test_identity(self, N, rng):
        d = _random_sign_diag(N, rng)
        assert tvd_diag(d, d) == pytest.approx(0.0, abs=1e-12)

    def test_opposite_diagonals(self, N, rng):
        """For ±1 diagonals, ``tvd_diag(d, -d) = 1.0`` exactly.

        Every entry differs by 2, so ``||d - (-d)||_1 = 2N`` and the
        leading ``0.5 / N`` factor rescales to 1.
        """
        d = _random_sign_diag(N, rng)
        assert tvd_diag(d, -d) == pytest.approx(1.0, abs=1e-12)

    def test_symmetric(self, N, rng):
        a = _random_sign_diag(N, rng)
        b = _random_sign_diag(N, rng)
        assert tvd_diag(a, b) == pytest.approx(tvd_diag(b, a), abs=1e-12)

    def test_in_unit_interval(self, N, rng):
        a = _random_sign_diag(N, rng)
        b = _random_sign_diag(N, rng)
        t = tvd_diag(a, b)
        assert 0.0 <= t <= 1.0 + 1e-12

    def test_triangle_inequality(self, N, rng):
        a = _random_sign_diag(N, rng)
        b = _random_sign_diag(N, rng)
        c = _random_sign_diag(N, rng)
        assert tvd_diag(a, c) <= tvd_diag(a, b) + tvd_diag(b, c) + 1e-12

    def test_single_bit_flip_is_1_over_N(self, N, rng):
        """Flipping one entry of a ±1 diagonal changes ``tvd_diag`` by
        exactly ``1/N`` (an L1 mass of 2, rescaled by ``0.5/N``)."""
        d = _random_sign_diag(N, rng)
        d_flip = d.copy()
        d_flip[0] *= -1.0
        assert tvd_diag(d, d_flip) == pytest.approx(1.0 / N, abs=1e-12)

    def test_handles_complex_input_silently(self):
        """Complex input on the unit circle whose real part matches d_real
        yields ``tvd_diag = 0`` exactly (real projection is applied)."""
        try:
            from numpy.exceptions import ComplexWarning as _CW
        except ImportError:  # pragma: no cover -- NumPy 1.x
            from numpy import ComplexWarning as _CW  # type: ignore[no-redef]
        d_real = np.array([1.0, -1.0] * 8)
        d_complex = np.exp(1j * np.pi * (1.0 - d_real) / 2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", _CW)
            t = tvd_diag(d_complex, d_real)
        assert t == pytest.approx(0.0, abs=1e-12)

    def test_normalisation_against_full_table(self):
        """Closed-form spot check: d=(+1,+1,+1,+1), b=(-1,-1,-1,-1) -> 1.0."""
        d = np.ones(8)
        assert tvd_diag(d, -d) == pytest.approx(1.0, abs=1e-12)
        assert tvd_diag(d, d) == pytest.approx(0.0, abs=1e-12)


# ===========================================================================
# Metric 2: hadamard_distribution_tvd
# ===========================================================================


class TestHadamardDistributionTvd:
    """Properties of the Hadamard-induced measurement-distribution TVD.

    This is the metric used by the warmstart-ablation algorithm because it
    matches the downstream basis-state-measurement semantics. It is
    **invariant under global negation** by design (``|H_n(-d)|^2 =
    |H_n d|^2``).
    """

    def test_probs_sum_to_one(self, N, rng):
        d = _random_sign_diag(N, rng)
        n = int(np.log2(N))
        Hn = _hadamard(n)
        s = Hn @ (d / np.sqrt(N))
        p = np.abs(s) ** 2
        assert p.sum() == pytest.approx(1.0, abs=1e-12)

    def test_identity(self, N, rng):
        d = _random_sign_diag(N, rng)
        assert hadamard_distribution_tvd(d, d) == pytest.approx(0.0, abs=1e-12)

    def test_global_sign_invariance(self, N, rng):
        """``hadamard_distribution_tvd(d, -d) = 0`` exactly.

        This is the *defining* invariance of this metric:
        ``|H_n d|^2 = |-H_n d|^2 = |H_n(-d)|^2`` element-wise.
        """
        d = _random_sign_diag(N, rng)
        assert hadamard_distribution_tvd(d, -d) == pytest.approx(0.0, abs=1e-12)

    def test_symmetric(self, N, rng):
        a = _random_sign_diag(N, rng)
        b = _random_sign_diag(N, rng)
        assert hadamard_distribution_tvd(a, b) == pytest.approx(
            hadamard_distribution_tvd(b, a), abs=1e-12
        )

    def test_in_unit_interval(self, N, rng):
        a = _random_sign_diag(N, rng)
        b = _random_sign_diag(N, rng)
        t = hadamard_distribution_tvd(a, b)
        assert 0.0 <= t <= 1.0 + 1e-12

    def test_triangle_inequality(self, N, rng):
        a = _random_sign_diag(N, rng)
        b = _random_sign_diag(N, rng)
        c = _random_sign_diag(N, rng)
        lhs = hadamard_distribution_tvd(a, c)
        rhs = hadamard_distribution_tvd(a, b) + hadamard_distribution_tvd(b, c)
        assert lhs <= rhs + 1e-12

    def test_orthogonal_distributions_is_max(self, N):
        """If p(d1) and p(d2) live on disjoint basis-state supports the
        metric returns 1.  We engineer two sign diagonals whose Hadamard
        spectra are deltas on different basis states."""
        n = int(np.log2(N))
        Hn = _hadamard(n)
        d1 = np.ones(N, dtype=np.float64)
        d2 = Hn[:, 1] * np.sqrt(N)
        assert np.allclose(np.abs(d2), 1.0)
        assert hadamard_distribution_tvd(d1, d2) == pytest.approx(1.0, abs=1e-12)

    def test_single_bit_flip_scale(self, N, rng):
        """A single bit flip stays bounded by ``4/sqrt(N)`` (the
        Hadamard-perturbation scale)."""
        d = _random_sign_diag(N, rng)
        d_flip = d.copy()
        d_flip[0] *= -1.0
        t = hadamard_distribution_tvd(d, d_flip)
        assert 0.0 < t
        assert t < 4.0 / np.sqrt(N)


# ===========================================================================
# Cross-metric contrast: the two metrics disagree exactly where they should
# ===========================================================================


def test_metrics_disagree_on_global_sign_flip():
    """Explicit pinning of the metric-separation contract.

    For any ±1 diagonal ``d``:
      - ``tvd_diag(d, -d) == 1.0``           (every entry differs by 2)
      - ``hadamard_distribution_tvd(d, -d) == 0.0``  (global sign is
                                                      unobservable)
    """
    d = np.array([1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0])
    assert tvd_diag(d, -d) == pytest.approx(1.0, abs=1e-12)
    assert hadamard_distribution_tvd(d, -d) == pytest.approx(0.0, abs=1e-12)


def test_metrics_agree_on_identity():
    """Identity case must give 0 under both metrics."""
    d = np.array([1.0, -1.0, 1.0, -1.0])
    assert tvd_diag(d, d) == pytest.approx(0.0, abs=1e-12)
    assert hadamard_distribution_tvd(d, d) == pytest.approx(0.0, abs=1e-12)
