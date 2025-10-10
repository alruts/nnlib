from typing import Callable

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from jax import Array
from jaxtyping import PRNGKeyArray


def reparametrize_linear(
    linear_layer: eqx.nn.Linear,
    weight_dist: Callable[[PRNGKeyArray, tuple[int, ...]], Array],
    bias_dist: Callable[[PRNGKeyArray, tuple[int, ...]], Array] | None = None,
    *,
    key: PRNGKeyArray,
) -> eqx.nn.Linear:
    """Reparametrize a Linear layer's weights (and bias).

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import equinox as eqx
        >>> key = jax.random.PRNGKey(0)
        >>> linear = eqx.nn.Linear(2, 3, key=key)
        >>> orig_w = linear.weight
        >>> orig_b = linear.bias
        >>> def w_dist(k, shape): return jax.random.normal(k, shape)
        >>> def b_dist(k, shape): return jax.random.normal(k, shape)
        >>> new_linear = reparametrize_linear(linear, w_dist, b_dist, key=key)
        >>> new_linear.weight.shape
        (3, 2)
        >>> new_linear.bias.shape
        (3,)
        >>> not jnp.allclose(new_linear.weight, orig_w)
        True
        >>> not jnp.allclose(new_linear.bias, orig_b)
        True
    """

    wkey, bkey = jrandom.split(key)

    new_layer = eqx.tree_at(
        lambda layer: layer.weight,
        linear_layer,
        weight_dist(wkey, linear_layer.weight.shape),
    )

    # False mismatch handling
    if linear_layer.use_bias and bias_dist is None:
        raise ValueError("Linear layer has bias=True but no bias_dist was provided.")
    if not linear_layer.use_bias and bias_dist is not None:
        raise ValueError("Linear layer has bias=False but bias_dist was provided.")

    if bias_dist and linear_layer.bias is not None:
        new_layer = eqx.tree_at(
            lambda layer: layer.bias,
            new_layer,
            bias_dist(bkey, linear_layer.bias.shape),
        )

    return new_layer


def siren_weight_dist(shape, omega_0, *, is_first=False, key: PRNGKeyArray) -> Array:
    """SIREN initialization distribution for weights."""
    out_features, in_features = shape

    if is_first:
        lim = 1.0 / in_features
    else:
        lim = jnp.sqrt(6.0 / in_features) / omega_0
    return jrandom.uniform(key, (out_features, in_features), minval=-lim, maxval=lim)


def siren_bias_dist(shape, *, is_first=False, key: PRNGKeyArray) -> Array:
    """SIREN initialization distribution for biases."""
    if is_first:
        lim = 1
        return jrandom.uniform(key, shape, minval=-lim, maxval=lim)
    else:
        return jnp.zeros(shape)


def _siren_activation(x, angular_frequency):
    return jnp.sin(angular_frequency * x)


def _identity(x):
    return x
