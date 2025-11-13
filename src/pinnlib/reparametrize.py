from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.nn import initializers
from jaxtyping import Array, DTypeLike, PRNGKeyArray, PyTree


def reparametrize_linear(
    linear_layer: eqx.nn.Linear,
    weight_dist: Callable[[PRNGKeyArray, tuple[int, ...]], Array],
    bias_dist: Callable[[PRNGKeyArray, tuple[int, ...]], Array] | None = None,
    *,
    key: PRNGKeyArray,
) -> eqx.nn.Linear:
    """Re-parameterize a Linear layer's weights (and bias).

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


def siren_weight_initializer(omega_0: float, is_first: bool = False):
    def init(
        key: Array,
        shape: tuple[int, ...],
        dtype: DTypeLike,
    ) -> Array:
        assert dtype is not None, "must provide dtype!"

        in_features = shape[-1]

        scale = 1.0 / in_features if is_first else jnp.sqrt(6.0 / in_features) / omega_0

        return initializers.variance_scaling(
            scale=scale,
            mode="fan_in",
            distribution="uniform",
        )(key, shape, dtype)  # variance scaling handles complex dtypes

    return init


def siren_bias_initializer(is_first: bool = False):
    """
    SIREN bias initializer:
      - First layer: uniform [-1, 1] for real, unit circle for complex
      - Other layers: zeros
    """

    def init(
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        dtype: DTypeLike,
    ) -> Array:
        assert dtype is not None, "must provide dtype!"

        if is_first:
            if jnp.issubdtype(dtype, jnp.complexfloating):
                # Sample random angle theta uniformly in [0, 2π)
                theta = jrandom.uniform(
                    key, shape, dtype=jnp.float32, minval=0.0, maxval=2 * jnp.pi
                )
                r = jrandom.uniform(
                    key, shape, dtype=jnp.float32, minval=0.0, maxval=1.0
                )
                return r * jnp.exp(1j * theta).astype(dtype)  # unit circle
            else:
                return jrandom.uniform(key, shape, dtype=dtype, minval=-1.0, maxval=1.0)
        else:
            return jnp.zeros(shape, dtype)

    return init


def make_nd_array_filter(n: int) -> Callable[[Array], bool]:
    """
    Return a function that checks if an array is n-dimensional.

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
    """
    Return a function that checks if an array is a leaf of the given PyTree.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> pytree = {"a": jnp.array([1.0, 2.0]), "b": (jnp.array([3.0]), 4)}
        >>> is_leaf = make_is_leaf_of_filter(pytree)
        >>> leaf_a = pytree["a"]
        >>> leaf_b = pytree["b"][0]
        >>> is_leaf(leaf_a)
        True
        >>> is_leaf(leaf_b)
        True
        >>> is_leaf(jnp.array([99.0]))
        False
    """
    leaves = jax.tree.leaves(pytree)
    return lambda x: any(x is leaf for leaf in leaves)


def reparam_pytree(
    filter_spec: Callable[[Any], bool],
    new_distribution: initializers.Initializer | Callable,
    dtype: Any,
    *,
    key: PRNGKeyArray,
):
    """
    Return a function that re-parameterizes model parameters using a new distribution.

    Args:
        filter_spec: Boolean function specifying which leaves to reparameterize.
        new_distribution: Callable with signature (key, shape, dtype) -> jnp.ndarray.
        dtype: Desired dtype for new parameters.
        key: JAX PRNGKey used to sample from the new distribution.

    Returns:
        A callable that takes a model and returns a new model with updated parameters.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import equinox as eqx
        >>> class Simple(eqx.Module):
        ...     weight: jnp.ndarray
        ...     bias: jnp.ndarray
        >>> model = Simple(jnp.ones((2, 2)), jnp.zeros((2,)))
        >>> filter_spec = lambda x: True
        >>> new_distribution = lambda key, shape, dtype: jax.random.normal(key, shape, dtype)
        >>> key = jax.random.PRNGKey(0)
        >>> reparam = reparam_pytree(filter_spec, new_distribution, jnp.float32, key=key)
        >>> new_model = reparam(model)
        >>> isinstance(new_model, Simple)
        True
        >>> new_model.weight.shape == model.weight.shape
        True
        >>> print((new_model.weight == model.weight).all())
        False
    """

    def apply(model: PyTree) -> PyTree:
        params, static = eqx.partition(model, filter_spec)
        leaves, treedef = jax.tree.flatten(params)
        keys = jax.random.split(key, len(leaves))
        new_leaves = [
            new_distribution(k, leaf.shape, dtype) for k, leaf in zip(keys, leaves)
        ]
        new_params = jax.tree.unflatten(treedef, new_leaves)
        return eqx.combine(new_params, static)

    return apply


def transform_pytree(filter_fn: Callable[[Any], bool], transform_fn: Callable):
    """
    Return a function that applies a transformation to selected model parameters.

    Args:
        filter_fn: Function specifying which leaves to transform (bool-returning).
        transform_fn: Callable with signature (leaf) -> transformed leaf.

    Returns:
        A callable that takes a model and returns a new model with transformed parameters.

    Example:
        >>> import jax.numpy as jnp
        >>> import equinox as eqx
        >>> class Simple(eqx.Module):
        ...     weight: jnp.ndarray
        ...     bias: jnp.ndarray
        >>> model = Simple(jnp.ones((2, 2)), jnp.zeros((2,)))
        >>> filter_fn = lambda x: x.ndim == 2
        >>> transform_fn = lambda x: 0.5 * x
        >>> transform = transform_pytree(filter_fn, transform_fn)
        >>> new_model = transform(model)
        >>> isinstance(new_model, Simple)
        True
        >>> print(jnp.allclose(new_model.weight, 0.5 * model.weight))
        True
        >>> print(jnp.allclose(new_model.bias, model.bias))
        True
    """

    def apply(model: PyTree) -> PyTree:
        params, static = eqx.partition(model, filter_fn)
        transformed_params = jax.tree.map(transform_fn, params)
        return eqx.combine(transformed_params, static)

    return apply
