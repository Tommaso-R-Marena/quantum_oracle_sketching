"""Debug utility for interferometric classical shadow scaling diagnostics."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from qos.theory.interferometric_shadow import InterferometricClassicalShadow


def main() -> None:
    key = jax.random.PRNGKey(1)
    n = 32
    k1, k2, k3 = jax.random.split(key, 3)

    w = jax.random.normal(k1, (n,)) + 1j * jax.random.normal(k2, (n,))
    w = w / jnp.linalg.norm(w)

    x_batch = jax.random.normal(k3, (1, n))
    x = x_batch[0]

    gt = float(jnp.real(jnp.dot(jnp.conj(w), x)))

    print(f"N={n}")
    print(f"||w||={float(jnp.linalg.norm(w)):.8f}")
    print(f"||x||={float(jnp.linalg.norm(x)):.8f}")
    print(f"gt={gt:.8f}")

    for t in [50, 200, 1000, 5000]:
        shadow = InterferometricClassicalShadow(w, num_shadows=t, key=jax.random.PRNGKey(t))
        shadow.build_shadow()

        re_vals = []
        for (bit_re, _bit_im), (phases, perm) in zip(shadow._shadow_bits, shadow._shadow_ops):
            rotated_w = phases * shadow.weight_state[perm]
            channel_re = float(jnp.real(jnp.dot(jnp.conj(rotated_w), x)))
            re_vals.append((1 - 2 * bit_re) * channel_re)

        raw_mean = float(jnp.mean(jnp.array(re_vals)))
        pred_2n = float(2 * n * raw_mean)
        err = abs(pred_2n - gt)

        first_five_bits = shadow._shadow_bits[:5]
        print(f"T={t}")
        print(f"  first_5_bits={first_five_bits}")
        print(f"  raw_mean={raw_mean:.8f}")
        print(f"  pred_2N={pred_2n:.8f}")
        print(f"  gt={gt:.8f}")
        print(f"  error={err:.8f}")

    # Prefix-slicing diagnostic from a single large shadow build.
    prefix_shadow = InterferometricClassicalShadow(w, num_shadows=5000, key=jax.random.PRNGKey(1))
    prefix_shadow.build_shadow()
    prefix_shadow._shadow_bits = prefix_shadow._shadow_bits[:1000]
    prefix_shadow._shadow_ops = prefix_shadow._shadow_ops[:1000]

    prefix_re_vals = []
    for (bit_re, _bit_im), (phases, perm) in zip(prefix_shadow._shadow_bits, prefix_shadow._shadow_ops):
        rotated_w = phases * prefix_shadow.weight_state[perm]
        channel_re = float(jnp.real(jnp.dot(jnp.conj(rotated_w), x)))
        prefix_re_vals.append((1 - 2 * bit_re) * channel_re)

    raw_prefix_mean = float(jnp.mean(jnp.array(prefix_re_vals)))
    print("Prefix T=1000 from single T=5000 shadow")
    print(f"  raw_mean_unscaled={raw_prefix_mean:.8f}")
    for label, scale in [("N", n), ("2N", 2 * n), ("N/2", n / 2), ("1", 1)]:
        pred = float(scale * raw_prefix_mean)
        err = abs(pred - gt)
        print(f"  scale={label:>3} pred={pred:.8f} gt={gt:.8f} error={err:.8f}")


if __name__ == "__main__":
    main()
