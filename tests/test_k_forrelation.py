import jax
import jax.numpy as jnp

from qos.data.generation import k_forrelation_data


def test_k2_forrelation_matches_paper_formula():
    key = jax.random.PRNGKey(0)
    gen = k_forrelation_data(n=3, k=2, key=key)
    f1 = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(8,))
    f2 = jax.random.choice(jax.random.PRNGKey(1), jnp.array([-1.0, 1.0]), shape=(8,))
    had = jnp.array(
        [[1, 1, 1, 1, 1, 1, 1, 1],
         [1, -1, 1, -1, 1, -1, 1, -1],
         [1, 1, -1, -1, 1, 1, -1, -1],
         [1, -1, -1, 1, 1, -1, -1, 1],
         [1, 1, 1, 1, -1, -1, -1, -1],
         [1, -1, 1, -1, -1, 1, -1, 1],
         [1, 1, -1, -1, -1, -1, 1, 1],
         [1, -1, -1, 1, -1, 1, 1, -1]],
        dtype=jnp.float32,
    )
    manual = float(jnp.mean(f1 * ((had @ f2) / 8.0)))
    val = gen.compute_exact_forrelation([f1, f2])
    assert abs(val - manual) < 1e-6


def test_quantum_query_zero_error_for_exact_oracle():
    gen = k_forrelation_data(n=3, k=2, key=jax.random.PRNGKey(1))
    oracle = jnp.ones((8,), dtype=jnp.complex64)
    est = gen.quantum_query_algorithm(oracle)
    assert abs(est - 1.0) < 0.01


def test_classical_complexity_increases_with_k():
    """N^(1-1/k) increases as k increases for fixed N > 1."""
    gen2 = k_forrelation_data(n=10, k=2, key=jax.random.PRNGKey(0))
    gen4 = k_forrelation_data(n=10, k=4, key=jax.random.PRNGKey(1))
    assert gen4.classical_streaming_complexity(0.1) > gen2.classical_streaming_complexity(0.1)


def test_noise_degrades_forrelation_estimate():
    """Noise-flipped layers differ from clean layers in mean absolute value."""
    clean = k_forrelation_data(n=4, k=3, key=jax.random.PRNGKey(0), noise_level=0.0)
    noisy = k_forrelation_data(n=4, k=3, key=jax.random.PRNGKey(0), noise_level=0.3)

    # sample_functions(key) returns a list of k full ±1 arrays of shape (2**n,)
    key = jax.random.PRNGKey(42)
    funcs_clean = clean.sample_functions(key)
    funcs_noisy = noisy.sample_functions(key)

    # Stack into (k, 2**n) matrices for element-wise comparison
    clean_stack = jnp.stack(funcs_clean)   # (k, dim)
    noisy_stack = jnp.stack(funcs_noisy)   # (k, dim)

    # With noise_level=0.3, ~30% of entries are flipped; mean abs diff > 0
    assert float(jnp.mean(jnp.abs(noisy_stack - clean_stack))) > 0.0
