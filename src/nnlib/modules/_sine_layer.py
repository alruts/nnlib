import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom


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

    [1] V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell, and G.
    Wetzstein, "Implicit Neural Representations with Periodic Activation
    Functions." arXiv, Jun. 17, 2020. Accessed: Mar. 08, 2024. [Online].
    Available: http://arxiv.org/abs/2006.09661
    """

    omega_0: float
    is_first: bool
    in_features: int
    out_features: int
    linear: eqx.nn.Linear

    def __init__(self, omega_0, is_first, in_features, out_features, *, key):
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

    def init_weights(self, linear, *, key):
        """Initialize the weights of the layer."""
        if self.is_first:
            limit = 1.0 / self.in_features
            new_weights = jrandom.uniform(
                key,
                (self.out_features, self.in_features),
                minval=-limit,
                maxval=limit,
            )

        else:
            limit = jnp.sqrt(6.0 / self.in_features) / self.omega_0
            new_weights = jrandom.uniform(
                key,
                (self.out_features, self.in_features),
                minval=-limit,
                maxval=limit,
            )

        # updated weights
        return eqx.tree_at(lambda layer: layer.weight, linear, new_weights)

    def __call__(self, x):
        """Forward pass."""
        return jnp.sin(self.omega_0 * self.linear(x))
