"""Polynomial generators and angle computation for QSVT.

Extends ``pyqsp`` with domain-restricted Chebyshev fitting and improved
numerical stability for high-degree polynomials.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import numpy as np
import scipy
from pyqsp import angle_sequence
from pyqsp.poly import PolyGenerator

from qos.config import real_dtype


class PolyTaylorSeries(PolyGenerator):
    """Extended Chebyshev polynomial generator with domain masking and parity control.

    Unlike the base ``pyqsp`` implementation, this class performs Chebyshev
    interpolation (not Taylor expansion) on a user-specified domain, yielding
    superior stability for high-degree approximations.
    """

    def taylor_series(
        self,
        func: Callable[[float], float],
        degree: int,
        ensure_bounded: bool = True,
        return_scale: bool = False,
        npts: int = 100,
        max_scale: float = 0.9,
        cheb_domain: tuple[float, float] = (-1.0, 1.0),
        parity: int | None = None,
    ) -> np.polynomial.chebyshev.Chebyshev | tuple[np.polynomial.chebyshev.Chebyshev, float]:
        """Compute a Chebyshev interpolant of ``func`` over ``cheb_domain``.

        Args:
            func: Scalar function to approximate.
            degree: Polynomial degree.
            ensure_bounded: Whether to rescale to ``max_scale`` on the domain.
            return_scale: Whether to return the rescaling factor.
            npts: Number of evaluation points for error reporting.
            max_scale: Maximum allowed absolute value on the domain.
            cheb_domain: 2-tuple specifying the fitting interval.
            parity: If 0 or 1, restrict to even/odd terms only.

        Returns:
            Chebyshev polynomial object, optionally paired with its scale.
        """
        cheb_samples = 2 * degree
        samples = np.polynomial.chebyshev.chebpts1(cheb_samples)
        scale = 1.0

        vals = np.array(list(map(func, samples)))
        mask = (samples >= cheb_domain[0]) & (samples <= cheb_domain[1])

        degree_list = (
            np.arange(parity, degree + 1, 2)
            if parity is not None
            else degree
        )
        cheb_coefs = np.polynomial.chebyshev.chebfit(
            samples, vals, degree_list, w=mask
        )
        cheb_poly = np.polynomial.chebyshev.Chebyshev(cheb_coefs)

        if ensure_bounded:
            res_1 = scipy.optimize.minimize(
                lambda x: -abs(cheb_poly(x[0])),
                (0.1,),
                bounds=[cheb_domain],
            )
            res_2 = scipy.optimize.minimize(
                lambda x: -abs(cheb_poly(x[0])),
                (-0.1,),
                bounds=[cheb_domain],
            )
            pmax_vals = [abs(cheb_poly(res_1.x[0])), abs(cheb_poly(res_2.x[0]))]
            scale = max_scale / max(pmax_vals)
            cheb_poly = scale * cheb_poly
            print(
                f"[PolyTaylorSeries] max {scale:.4f} at "
                f"{res_1.x[0] if pmax_vals[0] > pmax_vals[1] else res_2.x[0]:.4f}: normalizing"
            )

        adat = np.linspace(cheb_domain[0], cheb_domain[1], npts)
        pdat = cheb_poly(adat)
        edat = scale * func(adat)
        avg_err = abs(edat - pdat).mean()
        print(
            f"[PolyTaylorSeries] avg error = {avg_err:.3e} in "
            f"[{cheb_domain[0]}, {cheb_domain[1]}] using degree {degree}"
        )

        if ensure_bounded and return_scale:
            return cheb_poly, scale
        return cheb_poly


def get_qsvt_angles(
    func: Callable[[float], float],
    degree: int,
    rescale: float,
    cheb_domain: tuple[float, float] = (-1.0, 1.0),
    ensure_bounded: bool = True,
    parity: int | None = None,
) -> jnp.ndarray:
    """Compute QSVT phase angles for a target polynomial approximation.

    Args:
        func: Target function.
        degree: Polynomial degree.
        rescale: Scaling factor to ensure the polynomial is bounded in ``[-1, 1]``.
        cheb_domain: Domain for Chebyshev fitting.
        ensure_bounded: Whether to enforce the bound.
        parity: ``0`` for even, ``1`` for odd, ``None`` for auto.

    Returns:
        QSVT angle set as a JAX array.
    """
    poly = PolyTaylorSeries().taylor_series(
        func=func,
        degree=degree,
        max_scale=rescale,
        ensure_bounded=ensure_bounded,
        cheb_domain=cheb_domain,
        parity=parity,
    )

    phi_set, _, _ = angle_sequence.QuantumSignalProcessingPhases(
        poly, method="sym_qsp", chebyshev_basis=True
    )

    if not isinstance(phi_set, np.ndarray):
        raise RuntimeError("Failed to compute QSVT angles from pyqsp.")

    # Eq. (15) in arXiv:2002.11649.
    phi_to_angle = (
        np.array([1 / 4] + [1 / 2] * (phi_set.shape[0] - 2) + [1 / 4]) * np.pi
    )
    angle_set = phi_set + phi_to_angle
    return jnp.array(angle_set, dtype=real_dtype)


def get_qsvt_angles_inverse(
    kappa: float,
    epsilon: float = 0.1,
) -> tuple[jnp.ndarray, float]:
    """Compute QSVT angles for the inverse function ``1/x``.

    Args:
        kappa: Condition number (bounds the interval ``[1/kappa, 1]``).
        epsilon: Approximation error tolerance.

    Returns:
        ``(angle_set, scale)`` where ``scale`` is the polynomial scaling factor.
    """
    from pyqsp.poly import PolyOneOverX

    poly = PolyOneOverX()
    pcoefs, scale = poly.generate(
        kappa=kappa,
        epsilon=epsilon,
        return_coef=True,
        ensure_bounded=True,
        return_scale=True,
        chebyshev_basis=True,
    )

    pcoefs = np.asarray(pcoefs, dtype=np.float64)
    phi_set, _, _ = angle_sequence.QuantumSignalProcessingPhases(
        pcoefs, method="sym_qsp", chebyshev_basis=True
    )

    if not isinstance(phi_set, np.ndarray):
        raise RuntimeError("Failed to compute inverse QSVT angles.")

    phi_to_angle = (
        np.array([1 / 4] + [1 / 2] * (phi_set.shape[0] - 2) + [1 / 4]) * np.pi
    )
    angle_set = phi_set + phi_to_angle
    return jnp.array(angle_set, dtype=real_dtype), float(scale)


def get_qsvt_angles_sign(
    degree: int,
    threshold: float = 0.1,
    rescale: float = 0.9,
) -> tuple[jnp.ndarray, float]:
    """Compute QSVT angles for the sign function approximation.

    Args:
        degree: Polynomial degree (must be odd).
        threshold: Transition width around zero.
        rescale: Maximum magnitude of the approximation.

    Returns:
        ``(angle_set, scale)``.
    """
    from pyqsp.poly import PolySign

    poly = PolySign()
    pcoefs, scale = poly.generate(
        degree=degree,
        delta=2.0 / threshold,
        ensure_bounded=True,
        return_scale=True,
        chebyshev_basis=True,
        max_scale=rescale,
    )

    phi_set, _, _ = angle_sequence.QuantumSignalProcessingPhases(
        pcoefs, method="sym_qsp", chebyshev_basis=True
    )

    if not isinstance(phi_set, np.ndarray):
        raise RuntimeError("Failed to compute sign QSVT angles.")

    phi_to_angle = (
        np.array([1 / 4] + [1 / 2] * (phi_set.shape[0] - 2) + [1 / 4]) * np.pi
    )
    angle_set = phi_set + phi_to_angle
    return jnp.array(angle_set, dtype=real_dtype), float(scale)
