from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

_default_constants = {
    "wave_speed": 343.20,
    "medium_density": 1.2043,
}  # matches COMSOL defaults


def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32


def default_complex_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.complex128
    else:
        return jnp.complex64


def default_wave_speed() -> float:
    return _default_constants["wave_speed"]


def default_medium_density() -> float:
    return _default_constants["medium_density"]


def apply_model(model, params, *args):
    """Tric to enable gradient with respect to weights."""
    _, static = eqx.partition(model, eqx.is_inexact_array)
    model = eqx.combine(params, static)
    return model(*args)


def get_parameters(model):
    """Returns the parameters of the model."""
    params, _ = eqx.partition(model, eqx.is_array)
    return params


def args_to_array(f):
    """
    Wraps a function f(*args) to f_array(x_array) where x_array is a 1D array of all arguments.
    Returns a function that splits x_array into individual arguments internally.

    Example:
    >>> import jax.numpy as jnp
    >>> f = lambda x, y, z: x + y * z
    >>> f_array = args_to_array(f)
    >>> x_array = jnp.array([2., 3., 4.])
    >>> print(f_array(x_array))
    14.0
    >>> # It behaves equivalently to f(*x_array)
    >>> print(f(*x_array))
    14.0
    """

    def wrapper(x_array):
        # Convert 1D array to tuple of scalars for f
        args = tuple(x_array)
        return f(*args)

    return wrapper


def array_to_args(f):
    """
    Wraps a function f_array(x_array) -> scalar to a version f(*args)
    where the arguments are packed into a single 1D array internally.

    This is the inverse of `args_to_array`.

    Example:
    >>> import jax.numpy as jnp
    >>> f = lambda x, y, z: x + y * z
    >>> f_arr = args_to_array(f)
    >>> print(f_arr(jnp.array([2.0, 3.0, 4.0])))
    14.0
    >>> f_restored = array_to_args(f_arr)
    >>> print(f(2, 3, 4) == f_restored(2, 3, 4))
    True
    """

    def wrapper(*args):
        # Pack args into a 1D array for f_array
        x_array = jnp.array(args)
        return f(x_array)

    return wrapper
