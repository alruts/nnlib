from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp


def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32


def lift_tuple(
    f: Callable[[jnp.ndarray], Any],
) -> Callable[..., Tuple[Any, ...]]:
    """
    Lift a single-array function to take any number of scalar arguments
    and return a tuple of results.

    Example:
        >>> import jax.numpy as jnp
        >>> def sum_sq(arr):
        ...     return jnp.sum(arr ** 2)
        >>> lifted = lift_tuple(sum_sq)
        >>> lifted(1, 2, 3)
        (Array(1, dtype=int32), Array(4, dtype=int32), Array(9, dtype=int32))
    """
    return lambda *args: tuple(f(jnp.array([arg])) for arg in args)
