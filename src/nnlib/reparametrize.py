from typing import Any, Callable, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Complex, Float, PRNGKeyArray, PyTree


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


def siren_weight_dist(
    key: PRNGKeyArray,
    shape: tuple[int, ...],
    dtype: Float | Complex,
    *,
    omega_0: float,
    is_first: bool = False,
) -> jnp.ndarray:
    """SIREN initialization distribution for weights.

    Args:
        shape: (out_features, in_features)
        omega_0: frequency scaling parameter
        is_first: whether this is the first layer
        key: JAX PRNG key
        dtype: optional JAX dtype (if complex, initializes complex weights)
    """
    out_features, in_features = shape

    if is_first:
        lim = 1.0 / in_features
    else:
        lim = jnp.sqrt(6.0 / in_features) / omega_0

    weights = jrandom.uniform(key, (out_features, in_features), minval=-lim, maxval=lim)

    # handle complex-valued case
    if dtype is not None and jnp.issubdtype(dtype, jnp.complexfloating):
        lim /= jnp.sqrt(2.0)  # half the variance
        re_key, im_key = jrandom.split(key)

        re_weights = jrandom.uniform(
            re_key, (out_features, in_features), minval=-lim, maxval=lim
        )
        im_weights = jrandom.uniform(
            im_key, (out_features, in_features), minval=-lim, maxval=lim
        )
        weights = re_weights + 1j * im_weights

    return weights


def siren_bias_dist(
    key: PRNGKeyArray,
    shape: tuple[int, ...],
    dtype: Float | Complex,
    *,
    is_first: bool = False,
) -> Array:
    """SIREN initialization distribution for biases."""
    if is_first:
        lim = 1.0
        biases = jrandom.uniform(key, shape, minval=-lim, maxval=lim)
    else:
        biases = jnp.zeros(shape, dtype)

    # handle complex-valued case
    if dtype is not None and jnp.issubdtype(dtype, jnp.complexfloating):
        lim = 1.0
        re_key, im_key = jrandom.split(key)
        re_biases = jrandom.uniform(re_key, shape, minval=-lim, maxval=lim)
        im_biases = jrandom.uniform(im_key, shape, minval=-lim, maxval=lim)
        biases = re_biases + 1j * im_biases

    return biases


def make_nd_array_filter(n: int) -> Callable[[Array], bool]:
    """
    Return a function that checks if an array is a valid n-dimensional array.

    Example
        >>> import jax.numpy as jnp
        >>> import equinox as eqx
        >>> is_2d_array = make_nd_array_filter(2)
        >>> W = jnp.array([[0.1, 0.2, 0.3, 0.4],
        ...                [0.5, 0.6, 0.7, 0.8],
        ...                [0.9, 1.0, 1.1, 1.2]])
        >>> is_2d_array(W)
        True

        >>> is_1d_array = make_nd_array_filter(1)
        >>> b = jnp.array([0.1, 0.2, 0.3])
        >>> is_2d_array(b)
        False
        >>> is_1d_array(b)
        True
    """
    return lambda x: eqx.is_array(x) and len(x.shape) == n


def make_is_leaf_of_filter(pytree: PyTree) -> Callable[[Array], bool]:
    """ """
    leaves = jax.tree.leaves(pytree)
    return lambda x: any(x is leaf for leaf in leaves)


def reparam_model(model, filter_spec, new_distribution, dtype, *, key):
    params, static = eqx.partition(model, filter_spec)
    leaves, treedef = jax.tree.flatten(params)
    keys = jax.random.split(key, len(leaves))
    new_leaves = [
        new_distribution(k, leaf.shape, dtype) for k, leaf in zip(keys, leaves)
    ]
    new_params = jax.tree.unflatten(treedef, new_leaves)
    return eqx.combine(new_params, static)
