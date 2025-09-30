from functools import wraps

import equinox as eqx
import jax
import jax.numpy as jnp


def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32


def lift_to_args(fn):
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


def apply_model(self, params, *args):
    """Trick to enable gradient with respect to weights."""
    _, static = eqx.partition(self.model, eqx.is_inexact_array)
    model = eqx.combine(params, static)
    return model(*args[: model.in_features])


def get_parameters(self):
    """Returns the parameters of the model."""
    params, _ = eqx.partition(self.model, eqx.is_inexact_array)
    return params
