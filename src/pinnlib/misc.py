from functools import wraps
from typing import Any, Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from matplotlib import axes

_default_constants = {
    "wave_speed": 343.20,
    "medium_density": 1.2043,
}  # matches COMSOL defaults


def split_real_and_imaginary_activation(fn: Callable):
    """Split activation to apply separately to real and imaginary part."""
    return lambda z: fn(jnp.real(z)) + 1j * fn(jnp.imag(z))


def split_real_and_imaginary_metric(fn: Callable):
    """Split metric to apply separately to real and imaginary part."""
    return lambda x, xx: fn(x.real, xx.real) + 1j * fn(x.imag, xx.imag)


def split_real_and_imaginary_loss(fn: Callable):
    """Split loss to apply separately to real and imaginary part."""
    return lambda x, xx: fn(x.real, xx.real) + fn(x.imag, xx.imag)


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
    """Trick to enable gradient with respect to weights."""
    _, static = eqx.partition(model, eqx.is_inexact_array)
    model = eqx.combine(params, static)
    return model(*args)


def get_parameters(model):
    """Returns the parameters of the model."""
    params, _ = eqx.partition(model, eqx.is_array)
    return params


def grid_map(fn: Callable, axis_mask: Sequence[bool | int]):
    """
    Return a vectorized version of a function that maps over meshgrid arrays
    while keeping some arguments static.

    Args:
        fn: function taking separate arguments (x, y, z, ...)
        axis_mask: sequence of bools, one per argument.
                   True  -> argument is mapped (batched)
                   False -> argument is static (not batched)

    Returns:
        A new function that can be called like fn(X, Y, Z, ...) with meshgrids.

    Example:
    >>> x = jnp.array([0, 1])
    >>> y = jnp.array([10, 20])
    >>> X, Y = jnp.meshgrid(x, y, indexing='ij')
    >>> f = lambda x, y, t: x + y + t
    >>> grid_f = grid_map(f, [1, 1, 0])
    >>> print(grid_f(X, Y, 1))
    [[11 21]
     [12 22]]

    """

    def mapped_fn(*args):
        if len(axis_mask) != len(args):
            raise ValueError("`axis_mask` must have same length as args")

        mapped_args = [a.ravel() for a, m in zip(args, axis_mask) if m]
        static_args = [a for a, m in zip(args, axis_mask) if not m]

        def wrapper(*mapped_vals):
            out_args = []
            mapped_idx = 0
            static_idx = 0
            for m in axis_mask:
                if m:
                    out_args.append(mapped_vals[mapped_idx])
                    mapped_idx += 1
                else:
                    out_args.append(static_args[static_idx])
                    static_idx += 1
            return fn(*out_args)

        vals_flat = jax.vmap(wrapper)(*mapped_args)
        grid_shape = next(a.shape for a, m in zip(args, axis_mask) if m)
        return vals_flat.reshape(grid_shape)

    return mapped_fn


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
