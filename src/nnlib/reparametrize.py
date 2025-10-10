from typing import Callable, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import Array
from jaxtyping import PRNGKeyArray

from nnlib.architectures import ModifiedMLP


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


def make_siren(
    in_size: int | Literal["scalar"],
    out_size: int | Literal["scalar"],
    width_size: int,
    depth: int,
    activation: Callable = _siren_activation,
    final_activation: Callable = _identity,
    use_bias: bool = True,
    use_final_bias: bool = True,
    dtype=None,
    angular_frequency=30.0,
    *,
    key: PRNGKeyArray,
) -> eqx.nn.MLP:
    """
    Constructs a SIREN (Sinusoidal Representation Network [1]) using Equinox's MLP,
    with custom sinusoidal initialization for weights.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> key = jax.random.PRNGKey(0)
        >>> model = make_siren(2, 2, width_size=16, depth=3, key=key)
        >>> x = jnp.array([1.0, 0.0])
        >>> y = model(x)
        >>> y.shape
        (2,)

    [1] V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell, and G.
    Wetzstein, "Implicit Neural Representations with Periodic Activation
    Functions." arXiv, Jun. 17, 2020. Accessed: Mar. 08, 2024. [Online].
    Available: http://arxiv.org/abs/2006.09661
    """
    keys = jrandom.split(key, depth + 3)
    mlp_key, *keys = keys
    key_iter = iter(keys)

    mlp = eqx.nn.MLP(
        in_size=in_size,
        out_size=out_size,
        width_size=width_size,
        depth=depth,
        activation=lambda x: activation(x, angular_frequency),
        final_activation=final_activation,
        use_bias=use_bias,
        use_final_bias=use_final_bias,
        dtype=dtype,
        key=mlp_key,
    )

    first_layer = reparametrize_linear(
        mlp.layers[0],
        weight_dist=lambda k, s: siren_weight_dist(
            s, omega_0=angular_frequency, is_first=True, key=k
        ),
        bias_dist=lambda k, s: siren_bias_dist(s, is_first=True, key=k),
        key=next(key_iter),
    )

    other_layers = (
        reparametrize_linear(
            layer,
            weight_dist=lambda k, s: siren_weight_dist(
                s, omega_0=angular_frequency, is_first=False, key=k
            ),
            bias_dist=lambda k, s: siren_bias_dist(s, is_first=False, key=k),
            key=next(key_iter),
        )
        for layer in mlp.layers[1:]
    )

    # model surgery
    for idx, new_layer in enumerate([first_layer, *other_layers]):
        mlp = eqx.tree_at(lambda m: m.layers[idx], mlp, new_layer)

    return mlp


def make_modified_siren(
    in_size: int | Literal["scalar"],
    out_size: int | Literal["scalar"],
    width_size: int,
    depth: int,
    activation: Callable = _siren_activation,
    final_activation: Callable = _identity,
    use_bias: bool = True,
    use_final_bias: bool = True,
    dtype=None,
    angular_frequency=30.0,
    *,
    key: PRNGKeyArray,
) -> ModifiedMLP:
    """
    Constructs a modified SIREN (Sinusoidal Representation Network [1]) using Equinox's MLP,
    with custom sinusoidal initialization for weights.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> key = jax.random.PRNGKey(0)
        >>> model = make_modified_siren(2, 2, width_size=2, depth=2, key=key)
        >>> x = jnp.array([1.0, 0.0])
        >>> y = model(x)
        >>> y.shape
        (2,)

    [1] V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell, and G.
    Wetzstein, "Implicit Neural Representations with Periodic Activation
    Functions." arXiv, Jun. 17, 2020. Accessed: Mar. 08, 2024. [Online].
    Available: http://arxiv.org/abs/2006.09661
    """
    mlp_key, *keys = jrandom.split(key, depth + 4)
    key_iter = iter(keys)

    mod_mlp = ModifiedMLP(
        in_size=in_size,
        out_size=out_size,
        width_size=width_size,
        depth=depth,
        activation=lambda x: activation(x, angular_frequency),
        final_activation=final_activation,
        use_bias=use_bias,
        use_final_bias=use_final_bias,
        dtype=dtype,
        key=mlp_key,
    )

    first_layer, u, v = (
        reparametrize_linear(
            layer,
            weight_dist=lambda k, s: siren_weight_dist(
                s, omega_0=angular_frequency, is_first=True, key=k
            ),
            bias_dist=lambda k, s: siren_bias_dist(s, is_first=True, key=k),
            key=next(key_iter),
        )
        for layer in (mod_mlp.layers[0], mod_mlp.u, mod_mlp.v)
    )

    other_layers = (
        reparametrize_linear(
            layer,
            weight_dist=lambda k, s: siren_weight_dist(
                s, omega_0=angular_frequency, is_first=False, key=k
            ),
            bias_dist=lambda k, s: siren_bias_dist(s, is_first=False, key=k),
            key=next(key_iter),
        )
        for layer in mod_mlp.layers[1:]
    )

    # model surgery
    mod_siren = mod_mlp
    for idx, new_layer in enumerate([first_layer, *other_layers]):
        mod_siren = eqx.tree_at(lambda m: m.layers[idx], mod_siren, new_layer)

    mod_siren = eqx.tree_at(lambda m: m.u, mod_siren, u)
    mod_siren = eqx.tree_at(lambda m: m.v, mod_siren, v)

    return mod_siren
