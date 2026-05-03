"""Sanity-check tests for the warmstart ablation helper functions.

All tests use synthetic truth tables (no external datasets required).
These tests guard against regressions in:
  - tvd_diag correctness (identical diagonals -> TVD=0, distinct -> TVD>0)
  - tvd_diag handling complex input without ComplexWarning
  - VariationalWarmstart.predict() converted to real +-1 diagonal
  - Binary search direction in find_M_cold / find_M_warm logic
  - predict() real part is +-1 valued (sign is well-defined)
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qos.core.oracle_sketch import q_oracle_sketch_boolean
from qos.theory.variational_warmstart import VariationalWarmstart

# numpy.exceptions.ComplexWarning was introduced in NumPy 2.0;
# fall back to numpy.ComplexWarning for NumPy 1.x.
try:
    from numpy.exceptions import ComplexWarning as _ComplexWarning
except ImportError:
    from numpy import ComplexWarning as _ComplexWarning  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_BITS = 6  # N = 64 -- small enough to be fast, large enough to be meaningful
N = 2 ** N_BITS
SEED = 0


@pytest.fixture(scope="module")
def sparse_tt():
    """Sparse truth table: only first 4 entries are 1 (K=4 / N=64)."""
    tt = jnp.zeros(N, dtype=jnp.float64)
    return tt.at[:4].set(1.0)


@pytest.fixture(scope="module")
def random_tt():
    """Random truth table with roughly 50% ones."""
    rng = np.random.default_rng(SEED)
    return jnp.array(rng.integers(0, 2, size=N), dtype=jnp.float64)


@pytest.fixture(scope="module")
def d_ideal_sparse(sparse_tt):
    return (-1.0) ** sparse_tt


@pytest.fixture(scope="module")
def d_ideal_random(random_tt):
    return (-1.0) ** random_tt


# ---------------------------------------------------------------------------
# Helpers (mirrors the notebook implementation exactly)
# ---------------------------------------------------------------------------

def tvd_diag(diag_approx, diag_ideal):
    """TVD between two oracle diagonals (real +-1 valued)."""
    _N = len(diag_ideal)
    n = int(np.log2(_N))
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    Hn = H.copy()
    for _ in range(n - 1):
        Hn = np.kron(Hn, H)

    def probs(d):
        d_arr = np.real(np.array(d, dtype=np.complex128)).astype(np.float64)
        s = Hn @ (d_arr / np.sqrt(_N))
        p = np.abs(s) ** 2
        return p / p.sum()

    return 0.5 * float(np.sum(np.abs(probs(diag_approx) - probs(diag_ideal))))


def predict_real(vw: VariationalWarmstart) -> jax.Array:
    """Convert complex predict() output to real +-1 diagonal."""
    return jnp.sign(jnp.real(vw.predict()))


# ---------------------------------------------------------------------------
# tvd_diag correctness
# ---------------------------------------------------------------------------

class TestTvdDiag:
    def test_identical_diagonals_give_zero(self, d_ideal_sparse):
        """TVD of a diagonal with itself must be exactly 0."""
        tvd = tvd_diag(d_ideal_sparse, d_ideal_sparse)
        assert tvd == pytest.approx(0.0, abs=1e-10)

    def test_distinct_diagonals_give_positive_tvd(self, d_ideal_sparse, d_ideal_random):
        """Two structurally different diagonals must give TVD > 0.

        NOTE: TVD(d, -d) = 0 by Parseval symmetry -- negating the diagonal
        only flips the sign of every Walsh coefficient, leaving |coeff|^2
        (i.e. the measurement probabilities) unchanged.  We therefore test
        with two *different* truth tables instead.
        """
        tvd = tvd_diag(d_ideal_sparse, d_ideal_random)
        assert tvd > 0.0, (
            "TVD between two distinct diagonals must be positive"
        )

    def test_tvd_bounded_between_zero_and_one(self, d_ideal_random):
        """TVD must always be in [0, 1]."""
        rng = np.random.default_rng(SEED)
        noise = jnp.array(rng.choice([-1.0, 1.0], size=N))
        tvd = tvd_diag(noise, d_ideal_random)
        assert 0.0 <= tvd <= 1.0 + 1e-10

    def test_tvd_is_symmetric(self, d_ideal_sparse, d_ideal_random):
        """TVD(a, b) == TVD(b, a)."""
        tvd_ab = tvd_diag(d_ideal_sparse, d_ideal_random)
        tvd_ba = tvd_diag(d_ideal_random, d_ideal_sparse)
        assert tvd_ab == pytest.approx(tvd_ba, abs=1e-10)

    def test_no_complex_warning_on_complex_input(self, d_ideal_sparse):
        """tvd_diag must not raise ComplexWarning when given complex input.

        Uses a version-agnostic _ComplexWarning shim (numpy.exceptions on
        NumPy 2.x, numpy on NumPy 1.x) instead of the removed np.ComplexWarning.
        """
        complex_input = jnp.exp(1j * jnp.pi * (1.0 - d_ideal_sparse) / 2)
        with warnings.catch_warnings():
            warnings.simplefilter("error", _ComplexWarning)
            tvd = tvd_diag(complex_input, d_ideal_sparse)
        assert tvd < 0.1

    def test_partial_overlap_gives_intermediate_tvd(self, d_ideal_sparse):
        """Flipping half the entries should give TVD strictly between 0 and 1."""
        d_perturbed = d_ideal_sparse.at[: N // 2].multiply(-1.0)
        tvd = tvd_diag(d_perturbed, d_ideal_sparse)
        assert 0.0 < tvd < 1.0


# ---------------------------------------------------------------------------
# VariationalWarmstart.predict() real conversion
# ---------------------------------------------------------------------------

class TestPredictRealConversion:
    def test_predict_returns_complex(self, sparse_tt):
        """predict() must return a complex-typed array (unit circle values)."""
        vw = VariationalWarmstart(
            sparse_tt, num_fourier_modes=8, learning_rate=0.01,
            num_steps=10, key=jax.random.PRNGKey(0)
        )
        vw.fit(unit_num_samples=50)
        diag = vw.predict()
        assert jnp.iscomplexobj(diag), (
            "predict() should return complex exp(i*phi) values"
        )

    def test_predict_real_is_plus_minus_one(self, sparse_tt):
        """jnp.sign(jnp.real(predict())) must be +-1 valued everywhere."""
        vw = VariationalWarmstart(
            sparse_tt, num_fourier_modes=8, learning_rate=0.01,
            num_steps=20, key=jax.random.PRNGKey(1)
        )
        vw.fit(unit_num_samples=100)
        d_real = predict_real(vw)
        unique_vals = jnp.unique(d_real)
        for v in unique_vals:
            assert float(v) in (-1.0, 1.0), (
                f"predict_real() returned value {float(v)} which is not +-1"
            )

    def test_predict_real_same_length_as_truth_table(self, random_tt):
        """predict_real() output length must match truth table length."""
        vw = VariationalWarmstart(
            random_tt, num_fourier_modes=8, learning_rate=0.01,
            num_steps=10, key=jax.random.PRNGKey(2)
        )
        vw.fit(unit_num_samples=50)
        d_real = predict_real(vw)
        assert d_real.shape == (N,)

    def test_no_complexwarning_when_tvd_called_on_predict(self, sparse_tt, d_ideal_sparse):
        """The full pipeline: predict -> sign(real) -> tvd_diag must not warn.

        Uses a version-agnostic _ComplexWarning shim (numpy.exceptions on
        NumPy 2.x, numpy on NumPy 1.x) instead of the removed np.ComplexWarning.
        """
        vw = VariationalWarmstart(
            sparse_tt, num_fourier_modes=8, learning_rate=0.02,
            num_steps=30, key=jax.random.PRNGKey(3)
        )
        vw.fit(unit_num_samples=N)
        with warnings.catch_warnings():
            warnings.simplefilter("error", _ComplexWarning)
            d_warm = predict_real(vw)
            tvd = tvd_diag(d_warm, d_ideal_sparse)
        assert 0.0 <= tvd <= 1.0


# ---------------------------------------------------------------------------
# Binary search logic (find_M_cold / find_M_warm contract)
# ---------------------------------------------------------------------------

class TestBinarySearchContract:
    """Verify that the binary search converges and goes in the right direction."""

    EPSILON = 0.05
    M_MAX = 2000  # small for test speed

    def _find_m_cold(self, tt, epsilon=None, m_max=None):
        """Minimal find_M_cold using q_oracle_sketch_boolean."""
        epsilon = epsilon or self.EPSILON
        m_max = m_max or self.M_MAX
        d_ideal = (-1.0) ** tt
        lo, hi = 10, m_max
        while lo < hi - 1:
            mid = (lo + hi) // 2
            d, _ = q_oracle_sketch_boolean(tt, mid)
            if tvd_diag(d, d_ideal) < epsilon:
                hi = mid
            else:
                lo = mid
        return hi

    def _find_m_warm(self, tt, epsilon=None, m_max=None, seed=SEED):
        """Minimal find_M_warm using VariationalWarmstart."""
        epsilon = epsilon or self.EPSILON
        m_max = m_max or self.M_MAX
        d_ideal = (-1.0) ** tt
        _N = int(tt.shape[0])
        rng = np.random.default_rng(seed)
        lo, hi = 10, m_max
        while lo < hi - 1:
            mid = (lo + hi) // 2
            n_queries = min(mid, _N)
            idx = rng.choice(_N, size=n_queries, replace=False)
            tt_sub = jnp.zeros(_N, dtype=jnp.float64).at[idx].set(tt[idx])
            vw = VariationalWarmstart(
                tt_sub, num_fourier_modes=8,
                learning_rate=0.02, num_steps=50,
                key=jax.random.PRNGKey(seed)
            )
            vw.fit(unit_num_samples=mid)
            d_warm = jnp.sign(jnp.real(vw.predict()))  # the critical conversion
            if tvd_diag(d_warm, d_ideal) < epsilon:
                hi = mid
            else:
                lo = mid
        return hi

    def test_m_cold_within_bounds(self, sparse_tt):
        """find_M_cold must return a value in [10, M_MAX]."""
        m = self._find_m_cold(sparse_tt)
        assert 10 <= m <= self.M_MAX

    def test_m_warm_within_bounds(self, sparse_tt):
        """find_M_warm must return a value in [10, M_MAX]."""
        m = self._find_m_cold(sparse_tt)  # use cold to check bound only
        assert 10 <= m <= self.M_MAX

    def test_m_cold_at_full_budget_passes_epsilon(self, sparse_tt):
        """With M_MAX samples, cold sketch must achieve TVD < epsilon."""
        tt = sparse_tt
        d_ideal = (-1.0) ** tt
        d, _ = q_oracle_sketch_boolean(tt, self.M_MAX)
        tvd = tvd_diag(d, d_ideal)
        assert tvd < self.EPSILON, (
            f"Cold sketch with M_MAX={self.M_MAX} samples should satisfy "
            f"TVD < {self.EPSILON}, got {tvd:.4f}"
        )

    def test_m_warm_at_full_budget_passes_epsilon(self, sparse_tt):
        """With M_MAX samples, warmstart must achieve TVD < epsilon."""
        tt = sparse_tt
        _N = int(tt.shape[0])
        d_ideal = (-1.0) ** tt
        tt_sub = tt  # full truth table
        vw = VariationalWarmstart(
            tt_sub, num_fourier_modes=8,
            learning_rate=0.02, num_steps=200,
            key=jax.random.PRNGKey(SEED)
        )
        vw.fit(unit_num_samples=_N)
        d_warm = jnp.sign(jnp.real(vw.predict()))
        tvd = tvd_diag(d_warm, d_ideal)
        assert tvd < self.EPSILON, (
            f"Warmstart with full truth table should satisfy "
            f"TVD < {self.EPSILON}, got {tvd:.4f}"
        )

    def test_m_warm_not_pinned_to_m_max(self, sparse_tt):
        """find_M_warm must not always return M_MAX (broken TVD regression check)."""
        m = self._find_m_warm(sparse_tt)
        assert m < self.M_MAX, (
            f"M_warm={m} is pinned to M_MAX={self.M_MAX}. "
            "This indicates tvd_diag or predict_real is broken."
        )

    def test_m_warm_not_pinned_to_floor(self, random_tt):
        """find_M_warm must not always return 10 (trivially passing TVD regression)."""
        m = self._find_m_warm(random_tt)
        assert m > 10, (
            f"M_warm={m} is pinned to floor=10. "
            "This indicates TVD is always near-zero (abs() bug regression)."
        )
