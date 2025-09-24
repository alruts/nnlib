from typing import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Float


class PirateBlock(eqx.Module):
    """
    Implements an adaptive residual block as proposed in [1].

    The block consists of a sequence of eqx.nn.Linear layers with a chosen
    activation function applied in between. It also includes an adaptive
    skip connection via the parameter alpha, which is initialized to zero
    and learned during training. This allows the network to adaptively
    balance between residual and identity mappings, improving stability
    and expressiveness.

    [1] S. Wang, B. Li, Y. Chen, and P. Perdikaris, "PirateNets: Physics-informed
    Deep Learning with Residual Adaptive Networks." arXiv, Feb. 11, 2024. Accessed:
    Jun. 05, 2024. [Online]. Available: http://arxiv.org/abs/2402.00326
    """

    layers: Sequence[Callable]
    alpha: Float
    activation_fn: Callable

    def __init__(
        self,
        in_features,
        hidden_features,
        *,
        key=jrandom.PRNGKey(0),
    ):
        last_key, *keys = jax.random.split(key, 3)
        first_keys = iter(keys)

        self.layers = []

        # First layer
        self.layers.append(
            eqx.nn.Linear(
                in_features=in_features,
                out_features=hidden_features,
                key=next(first_keys),
            )
        )

        # Second layer
        self.layers.append(
            eqx.nn.Linear(
                in_features=hidden_features,
                out_features=hidden_features,
                key=next(first_keys),
            )
        )

        # Last layer
        self.layers.append(
            eqx.nn.Linear(
                in_features=hidden_features,
                out_features=in_features,
                key=last_key,
            )
        )

        self.alpha = jnp.array(0.0)
        self.activation_fn = jax.nn.tanh

    def __call__(self, x, u, v):
        """Forward pass."""
        identity = x
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation_fn(x)
            x = x * u + (1 - x) * v

        x = self.layers[-1](x)  # Last layer
        x = self.alpha * x + (1 - self.alpha) * identity
        return x
