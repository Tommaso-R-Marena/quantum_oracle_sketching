import jax

from qos.experiments.non_iid_scaling import fit_non_iid_exponent, run_non_iid_scaling_experiment


def test_iid_baseline_matches_theorem_d12():
    df = run_non_iid_scaling_experiment(128, [1], [128, 256, 512], 2, jax.random.PRNGKey(0))
    errs = df.sort_values("M")["error_mean"].values
    assert errs[0] >= errs[-1]


def test_error_increases_with_repetition_number():
    df = run_non_iid_scaling_experiment(128, [1, 2, 4], [256], 2, jax.random.PRNGKey(1))
    arr = df.sort_values("R")["error_mean"].values
    assert arr[0] <= arr[-1]


def test_fit_exponent_returns_dict_with_keys():
    df = run_non_iid_scaling_experiment(64, [1, 2], [128, 256], 1, jax.random.PRNGKey(2))
    out = fit_non_iid_exponent(df)
    assert set(out.keys()) == {"M_exponent", "R_exponent", "intercept"}
