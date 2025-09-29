from typing import Callable, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float, PRNGKeyArray


class ModifiedMLP(eqx.nn.MLP):
    """
    A modified multi-layer perceptron (MLP) from [1] that applies learned linear
    modulators `u` and `v` to the hidden layers before the final output.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import equinox as eqx
        >>> key = jax.random.PRNGKey(0)
        >>> net_key, x_key = jax.random.split(key)
        >>> net = ModifiedMLP(in_size='scalar', out_size='scalar', width_size=8, depth=3, key=net_key)
        >>> # Input values
        >>> y = net(jnp.array(1.9))
        >>> y.shape
        ()

        >>> key = jax.random.PRNGKey(0)
        >>> net_key, x_key = jax.random.split(key)
        >>> net = ModifiedMLP(in_size=4, out_size=2, width_size=8, depth=3, key=net_key)
        >>> # Input values
        >>> x = jnp.ones(4)
        >>> y = net(x)
        >>> y.shape
        (2,)
    """

    u: eqx.nn.Linear
    v: eqx.nn.Linear

    def __init__(
        self,
        in_size: int | Literal["scalar"],
        out_size: int | Literal["scalar"],
        width_size: int,
        depth: int,
        activation: Callable = jax.nn.tanh,
        final_activation: Callable = lambda x: x,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        u_key, v_key, mlp_key = jrandom.split(key, 3)

        super().__init__(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            final_activation=final_activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            dtype=dtype,
            key=mlp_key,
        )

        self.u, self.v = (
            eqx.nn.Linear(
                in_features=in_size,
                out_features=width_size,
                use_bias=use_bias,
                dtype=dtype,
                key=k,
            )
            for k in (u_key, v_key)
        )

    def __call__(self, x: Array, *, key: PRNGKeyArray | None = None) -> Array:
        """Forward pass."""

        u = self.activation(self.u(x))
        v = self.activation(self.v(x))

        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            x = x * u + (1 - x) * v

        x = self.final_activation(self.layers[-1](x))  # Last layer

        return x


class PirateBlock(eqx.Module):
    """
    Implements an adaptive residual block as proposed in [1].

    The block consists of a sequence of eqx.nn.Linear layers with a chosen
    activation function applied in between. It also includes an adaptive
    skip connection via the parameter alpha, which is initialized to zero
    and learned during training. This allows the network to adaptively
    balance between residual and identity mappings, improving stability
    and expressiveness.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import equinox as eqx
        >>> key = jax.random.PRNGKey(42)
        >>> # split keys so block, u and v are independently initialized
        >>> block_key, u_key, v_key = jax.random.split(key, 3)
        >>> block = PirateBlock(io_size=4, width_size=8, key=block_key)
        >>> x = jnp.ones((4,))
        >>> # u and v must output hidden_features to match intermediate x
        >>> u = eqx.nn.Linear(in_features=4, out_features=8, key=u_key)
        >>> v = eqx.nn.Linear(in_features=4, out_features=8, key=v_key)
        >>> y = block(x, u, v)
        >>> y.shape
        (4,)
        >>> bool(jnp.allclose(y, x))  # alpha==0 -> identity mapping at initialization
        True

    [1] S. Wang, B. Li, Y. Chen, and P. Perdikaris, "PirateNets: Physics-informed
    Deep Learning with Residual Adaptive Networks." arXiv, Feb. 11, 2024. Accessed:
    Jun. 05, 2024. [Online]. Available: http://arxiv.org/abs/2402.00326

    """

    layers: tuple[eqx.nn.Linear, ...]
    io_size: int | Literal["scalar"]
    width_size: int
    activation: Callable
    alpha: Float

    def __init__(
        self,
        io_size: int | Literal["scalar"],
        width_size: int,
        activation: Callable = jax.nn.tanh,
        use_bias: bool = True,
        dtype=None,
        *,
        key=jrandom.PRNGKey(0),
    ):
        fst_key, snd_key, thd_key = jax.random.split(key, 3)

        first_layer = eqx.nn.Linear(
            in_features=io_size,
            out_features=width_size,
            dtype=dtype,
            key=fst_key,
            use_bias=use_bias,
        )
        second_layer = eqx.nn.Linear(
            in_features=width_size,
            out_features=width_size,
            dtype=dtype,
            key=snd_key,
            use_bias=use_bias,
        )
        last_layer = eqx.nn.Linear(
            in_features=width_size,
            out_features=io_size,
            dtype=dtype,
            key=thd_key,
            use_bias=use_bias,
        )

        self.layers = (first_layer, second_layer, last_layer)
        self.alpha = jnp.array(0.0, dtype=dtype)
        self.activation = activation
        self.io_size = io_size
        self.width_size = width_size

    def __call__(self, x, u, v):
        """Forward pass."""
        identity = x

        u = self.activation(u(x))
        v = self.activation(v(x))

        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
            x = x * u + (1 - x) * v

        x = self.layers[-1](x)  # Last layer
        x = self.alpha * x + (1 - self.alpha) * identity
        return x


class PirateNet(eqx.Module):
    """
    Implements an PirateNet as proposed in [1]. A pirate net consists of multiple PirateBlock
    layers.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import equinox as eqx
        >>> key = jax.random.PRNGKey(0)
        >>> net_key, x_key = jax.random.split(key)
        >>> net = PirateNet(in_size='scalar', out_size='scalar', width_size=8, depth=3, key=net_key)
        >>> # Input values
        >>> y = net(jnp.array(1.9))
        >>> y.shape
        ()

        >>> key = jax.random.PRNGKey(0)
        >>> net_key, x_key = jax.random.split(key)
        >>> net = PirateNet(in_size=4, out_size=2, width_size=8, depth=3, key=net_key)
        >>> # Input values
        >>> x = jnp.ones(4)
        >>> y = net(x)
        >>> y.shape
        (2,)

    [1] S. Wang, B. Li, Y. Chen, and P. Perdikaris, "PirateNets: Physics-informed
    Deep Learning with Residual Adaptive Networks." arXiv, Feb. 11, 2024. Accessed:
    Jun. 05, 2024. [Online]. Available: http://arxiv.org/abs/2402.00326
    """

    pirate_layers: tuple[PirateBlock, ...]
    last_layer: eqx.nn.Linear
    u: eqx.nn.Linear
    v: eqx.nn.Linear
    activation: Callable
    final_activation: Callable
    use_bias: bool = eqx.field(static=True)
    use_final_bias: bool = eqx.field(static=True)
    in_size: int | Literal["scalar"] = eqx.field(static=True)
    out_size: int | Literal["scalar"] = eqx.field(static=True)
    width_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __init__(
        self,
        in_size: int | Literal["scalar"],
        out_size: int | Literal["scalar"],
        width_size: int,
        depth: int,
        activation: Callable = jax.nn.tanh,
        final_activation: Callable = lambda x: x,
        use_bias: bool = True,
        use_final_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        u_key, v_key, *keys = jax.random.split(key, depth + 2)
        keys_iter = iter(keys)

        self.u, self.v = (
            eqx.nn.Linear(
                in_features=in_size,
                out_features=width_size,
                use_bias=use_bias,
                dtype=dtype,
                key=k,
            )
            for k in (u_key, v_key)
        )

        self.pirate_layers = tuple(
            PirateBlock(
                io_size=in_size,
                width_size=width_size,
                activation=activation,
                dtype=dtype,
                key=next(keys_iter),
            )
            for _ in range(depth - 1)
        )

        self.last_layer = eqx.nn.Linear(
            in_size,
            out_size,
            use_final_bias,
            dtype=dtype,
            key=next(keys_iter),
        )

        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth

        # In case `activation` or `final_activation` are learnt, then make a separate
        # copy of their weights for every neuron.
        self.activation = eqx.filter_vmap(
            eqx.filter_vmap(lambda: activation, axis_size=width_size), axis_size=depth
        )()
        if out_size == "scalar":
            self.final_activation = final_activation
        else:
            self.final_activation = eqx.filter_vmap(
                lambda: final_activation, axis_size=out_size
            )()
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias

    def __call__(self, x: Array, *, key: PRNGKeyArray | None = None) -> Array:
        """Forward pass."""
        for block in self.pirate_layers:
            x = block(x, self.u, self.v)

        # Last layer
        x = self.final_activation(self.last_layer(x))
        return x
