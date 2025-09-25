from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray


class SineLayer(eqx.Module):
    """Sine layer. This module implements a single layer of a SIREN network.
    Additonally, it handles weight initialization according to description
    in paper [1].

    Args:
        - omega_0: Frequency of the sine function.
        - is_first: Whether this is the first layer.
        - in_features: Number of input features.
        - out_features: Number of output features.
        - key: Random key.

    Returns:
        SineLayer: Sine layer module.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import equinox as eqx
        >>> key = jax.random.PRNGKey(0)
        >>> layer = SineLayer(omega_0=30.0, is_first=True, in_features=3, out_features=4, key=key)
        >>> x = jnp.array([0.1, 0.2, 0.3])
        >>> y = layer(x)
        >>> isinstance(y, jax.Array)
        True
        >>> y.shape
        (4,)

    [1] V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell, and G.
    Wetzstein, "Implicit Neural Representations with Periodic Activation
    Functions." arXiv, Jun. 17, 2020. Accessed: Mar. 08, 2024. [Online].
    Available: http://arxiv.org/abs/2006.09661
    """

    omega_0: float
    is_first: bool
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    linear: eqx.nn.Linear

    def __init__(
        self,
        omega_0: float,
        is_first: bool,
        in_features: int = eqx.field(static=True),
        out_features: int = eqx.field(static=True),
        *,
        key: PRNGKeyArray,
    ):
        # Split key
        layer_key, init_key = jrandom.split(key)

        # Set attributes
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.out_features = out_features

        # Initialize linear layer
        self.linear = eqx.nn.Linear(
            in_features, out_features, use_bias=True, key=layer_key
        )

        # Initialize weights
        self.linear = self.init_weights(self.linear, key=init_key)

    def init_weights(
        self, linear: eqx.nn.Linear, *, key: PRNGKeyArray
    ) -> eqx.nn.Linear:
        """Initialize the weights of the layer."""

        # First layer needs a special case
        if self.is_first:
            limit = 1.0 / self.in_features
        else:
            limit = jnp.sqrt(6.0 / self.in_features) / self.omega_0

        # Initialize weights
        new_weights = jrandom.uniform(
            key,
            (self.out_features, self.in_features),
            minval=-limit,
            maxval=limit,
        )

        # Update linear layer
        return eqx.tree_at(lambda layer: layer.weight, linear, new_weights)

    def __call__(self, x):
        """Forward pass."""
        return jnp.sin(self.omega_0 * self.linear(x))
