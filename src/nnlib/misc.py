from functools import wraps
from typing import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from matplotlib import axes

_default_constants = {
    "wave_speed": 343.20,
    "medium_density": 1.2043,
}  # matches COMSOL defaults


def split_activation(fn: Callable):
    """Lift activation from R->R to C->C."""
    return lambda z: fn(jnp.real(z)) + 1j * fn(jnp.imag(z))


def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32


def lift_to_args(fn: Callable):
    """
    Lift a function that expects a single array input to accept *args as scalars.

    >>> import jax
    >>> import jax.numpy as jnp
    >>> def dummy_net(x):
    ...     x, y, z = x
    ...     return x**2 + y**2 + z
    >>> lifted_net = lift_to_args(dummy_net)
    >>> # Call with scalars
    >>> print(lifted_net(1.0, 2.0, 3.0))
    8.0
    >>> print(jax.grad(lifted_net, argnums=0)(1.0, 2.0, 3.0))
    2.0
    >>> print(jax.grad(lifted_net, argnums=1)(1.0, 2.0, 3.0))
    4.0
    >>> print(jax.grad(lifted_net, argnums=2)(1.0, 2.0, 3.0))
    1.0
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        x = jnp.stack(args)
        return fn(x, **kwargs)

    return wrapper


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


def nested_vmap(fn: Callable, in_axes_list: list[tuple], out_axes_list=None):
    for i, in_axes in enumerate(reversed(in_axes_list)):
        out_axes = None
        if out_axes_list is not None:
            out_axes = out_axes_list[-(i + 1)]
        fn = jax.vmap(fn, in_axes=in_axes, out_axes=out_axes)
    return fn


def grid_vmap(fn: Callable, axis_mask: Sequence[bool | int]):
    """
    Lift a function that takes single values into grid

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>>
        >>> f = lambda static, x, y, t: x + y + t
        >>>
        >>> x = jnp.linspace(0, 1, 3)
        >>> (X, Y), t = jnp.meshgrid(x, x, indexing="ij"), 1.0
        >>> grid_f = grid_vmap(f, [0, 1, 1, 0])
        >>> print(grid_f(..., X, Y, t))
        [[1.  1.5 2. ]
         [1.5 2.  2.5]
         [2.  2.5 3. ]]
    """
    fst = tuple(1 if x else None for x in axis_mask)
    snd = tuple(0 if x else None for x in axis_mask)

    in_axes_list = [fst, snd]
    out_axes_list = [0 for _ in range(len(in_axes_list))]
    return nested_vmap(fn, in_axes_list, out_axes_list)
