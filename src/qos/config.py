"""Global configuration and numerical settings for QOS."""

from __future__ import annotations

import os
from typing import Literal

import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Numerical dtypes
# ---------------------------------------------------------------------------

# Default to 64-bit precision for quantum simulation accuracy.
# Users can override via environment variable QOS_PRECISION=32.
_PRECISION_ENV = os.environ.get("QOS_PRECISION", "64")

if _PRECISION_ENV == "32":
    real_dtype = jnp.float32
    complex_dtype = jnp.complex64
    int_dtype = jnp.int32
else:
    real_dtype = jnp.float64
    complex_dtype = jnp.complex128
    int_dtype = jnp.int64

jax.config.update("jax_enable_x64", _PRECISION_ENV == "64")


# ---------------------------------------------------------------------------
# Pydantic configuration model
# ---------------------------------------------------------------------------

class QOSConfig(BaseModel):
    """Runtime configuration for quantum oracle sketching.

    Attributes:
        precision: Numerical precision (32 or 64 bits).
        arcsin_degree: Default polynomial degree for arcsin QSVT approximation.
            Must be **odd** because arcsin is an odd function (parity=1 in pyqsp).
        sign_degree: Default polynomial degree for sign-function QSVT approximation.
        sign_threshold_factor: Multiplicative factor for sign threshold relative to sparsity.
        sign_rescale: Target magnitude of the sign function approximation.
        max_scale: Maximum rescaling factor for Chebyshev polynomial fitting.
        amplitude_amplification_degree: Default degree for amplitude amplification.
        amplitude_amplification_target_norm: Target norm after amplitude amplification.
        flat_vector_time_scale: Time parameter t = jnp.pi * dim for flat vector sketching.
        general_vector_time_scale: Time parameter scaling for general vector sketching.
        matrix_element_time_scale: Time parameter scaling for matrix elements.
        matrix_index_time_scale: Time parameter scaling for matrix index oracle.
        logspace_fit_npts: Number of points for log-space fit visualization.
        random_seed: Default random seed for reproducibility.
    """

    precision: Literal[32, 64] = Field(
        default=64 if _PRECISION_ENV == "64" else 32,
        description="Numerical precision in bits.",
    )
    # arcsin is an odd function -> QSVT polynomial must have parity=1 (odd degree).
    arcsin_degree: int = Field(default=21, ge=1, description="Degree of arcsin polynomial (must be odd).")
    sign_degree: int = Field(default=101, ge=1, description="Degree of sign polynomial.")
    sign_threshold_factor: float = Field(
        default=0.8, gt=0, lt=1, description="Threshold factor for sign function."
    )
    sign_rescale: float = Field(
        default=0.9999, gt=0, le=1, description="Rescaling factor for sign function."
    )
    max_scale: float = Field(
        default=0.9, gt=0, le=1, description="Max scale for Chebyshev fitting."
    )
    amplitude_amplification_degree: int = Field(
        default=51, ge=1, description="Degree for amplitude amplification polynomial."
    )
    amplitude_amplification_target_norm: float = Field(
        default=0.98, gt=0, le=1, description="Target norm after amplitude amplification."
    )
    flat_vector_time_scale: str = Field(
        default="pi*dim", description="Time scale formula for flat vectors."
    )
    general_vector_time_scale: float = Field(
        default=5.0, gt=0, description="Denominator for general vector time scale."
    )
    matrix_element_time_scale: str = Field(
        default="nnz", description="Time scale formula for matrix elements."
    )
    matrix_index_time_scale: str = Field(
        default="pi*nnz/(2*sparsity+1)", description="Time scale formula for matrix indices."
    )
    logspace_fit_npts: int = Field(default=100, ge=10, description="Points for log-space fits.")
    random_seed: int = Field(default=42, description="Default random seed.")

    @field_validator("arcsin_degree")
    @classmethod
    def _arcsin_must_be_odd(cls, v: int) -> int:  # noqa: N805
        """arcsin(x) is an odd function; QSVT requires parity=1, i.e. odd degree."""
        if v % 2 == 0:
            raise ValueError(
                f"arcsin_degree must be odd (got {v}). "
                "arcsin is an odd function so pyqsp requires parity=1, "
                "which demands an odd-degree polynomial."
            )
        return v

    @field_validator("sign_degree")
    @classmethod
    def _sign_must_be_odd(cls, v: int) -> int:  # noqa: N805
        if v % 2 == 0:
            raise ValueError("sign_degree must be odd.")
        return v


# Global default configuration instance.
DEFAULT_CONFIG = QOSConfig()


def get_default_config() -> QOSConfig:
    """Return a copy of the default configuration."""
    return QOSConfig.model_validate(DEFAULT_CONFIG.model_dump())
