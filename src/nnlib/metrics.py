from typing import Callable, Dict, Optional

import jax.numpy as jnp
from jaxtyping import Array

#
## Point wise metrics
#


def abs_error(p_ref: Array, p_pred: Array) -> Array:
    """Compute absolute error between reference and predicted values."""
    return jnp.abs(p_pred - p_ref)


def sq_error(p_ref: Array, p_pred: Array) -> Array:
    """Compute squared error between reference and predicted values."""
    return (p_pred - p_ref) ** 2


def relative(p_ref: Array, p_pred: Array, eps: float = 1e-8) -> Array:
    """Compute relative error between reference and predicted values."""
    return jnp.abs(p_pred - p_ref) / (jnp.abs(p_ref) + jnp.abs(p_pred) + eps)


def log_error(p_ref: Array, p_pred: Array) -> Array:
    """Compute absolute difference in log-scaled values."""
    return jnp.abs(jnp.log1p(jnp.abs(p_ref)) - jnp.log1p(jnp.abs(p_pred)))


def diff(p_ref: Array, p_pred: Array) -> Array:
    """Compute simple difference (prediction minus reference)."""
    return p_pred - p_ref


point_wise_metrics: Dict[str, Callable[..., Array]] = {
    "abs_error": abs_error,
    "sq_error": sq_error,
    "rel_error": relative,
    "log_error": log_error,
    "diff": diff,
}


#
## Global metrics
#


def mse(p_ref: Array, p_pred: Array) -> Array:
    """Compute mean squared error over all points."""
    return jnp.mean((p_pred - p_ref) ** 2)


def rmse(p_ref: Array, p_pred: Array) -> Array:
    """Compute root mean squared error over all points."""
    return jnp.sqrt(jnp.mean((p_pred - p_ref) ** 2))


def mae(p_ref: Array, p_pred: Array) -> Array:
    """Compute mean absolute error over all points."""
    return jnp.mean(jnp.abs(p_pred - p_ref))


def nrmse_range(p_ref: Array, p_pred: Array, eps: float = 1e-8) -> Array:
    """Compute normalized RMSE using the data range."""
    return jnp.sqrt(jnp.mean((p_pred - p_ref) ** 2)) / (p_ref.max() - p_ref.min() + eps)


def nrmse_std(p_ref: Array, p_pred: Array, eps: float = 1e-8) -> Array:
    """Compute normalized RMSE using the reference standard deviation."""
    return jnp.sqrt(jnp.mean((p_pred - p_ref) ** 2)) / (jnp.std(p_ref) + eps)


def mean_rel_error(p_ref: Array, p_pred: Array, eps: float = 1e-8) -> Array:
    """Compute mean relative error over all points."""
    return jnp.mean(jnp.abs(p_pred - p_ref) / (jnp.abs(p_ref) + jnp.abs(p_pred) + eps))


def mag_phase(
    x: Array,
    y: Array,
    axis: Optional[int] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Array:
    """Compute combined magnitude and phase error."""
    return alpha * jnp.mean(
        (jnp.abs(x) - jnp.abs(y)) ** 2, axis=axis
    ) + beta * jnp.mean(jnp.angle(jnp.exp(1j * (x - y))) ** 2, axis=axis)


aggregated_metrics: Dict[str, Callable[..., Array]] = {
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
    "nrmse_range": nrmse_range,
    "nrmse_std": nrmse_std,
    "mean_rel_error": mean_rel_error,
    "mag_phase": mag_phase,
}
