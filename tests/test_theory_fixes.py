import jax
import jax.numpy as jnp

from qos.theory.interferometric_shadow import InterferometricClassicalShadow
from qos.theory.variational_warmstart import VariationalWarmstart


def test_variational_warmstart_beats_baseline():
    """Variational error must be strictly less than baseline for all KF."""
    rng = jax.random.PRNGKey(0)
    N = 64
    phases = jax.random.uniform(rng, (N,), minval=0.0, maxval=2 * jnp.pi)
    f = jnp.exp(1j * phases)

    for KF in [8, 16, 32]:
        vw = VariationalWarmstart(
            f,
            num_fourier_modes=KF,
            learning_rate=0.001,
            num_steps=500,
            key=jax.random.PRNGKey(KF),
        )
        result = vw.fit()
        assert result["variational_error"] < result["baseline_error"], (
            f"KF={KF}: variational={result['variational_error']:.4f} "
            f">= baseline={result['baseline_error']:.4f}"
        )


def test_shadow_error_decays_with_T():
    """Shadow error should shrink with T and converge to the true Re<w|x> value."""
    key = jax.random.PRNGKey(1)
    N = 32
    k1, k2, k3 = jax.random.split(key, 3)
    w = jax.random.normal(k1, (N,)) + 1j * jax.random.normal(k2, (N,))
    w = w / jnp.linalg.norm(w)
    x = jax.random.normal(k3, (1, N))
    gt = float(jnp.real(jnp.dot(jnp.conj(w), x[0])))

    errors = []
    for T in [50, 200, 1000, 5000]:
        shadow = InterferometricClassicalShadow(w, num_shadows=T, key=jax.random.PRNGKey(T))
        shadow.build_shadow()
        pred = shadow.predict(x)[0, 0]
        errors.append(abs(float(pred) - gt))

    assert errors[1] < errors[0] * 1.1, (
        f"T=200 error {errors[1]:.4f} not < T=50 error {errors[0]:.4f}"
    )
    assert errors[2] < errors[1] * 1.1, (
        f"T=1000 error {errors[2]:.4f} not < T=200 error {errors[1]:.4f}"
    )
    assert abs(errors[-1] - 0.0) < 0.10, (
        f"T=5000 absolute error {errors[-1]:.4f} not close to 0"
    )
