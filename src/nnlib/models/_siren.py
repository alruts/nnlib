import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from nnlib.modules import SineLayer


class SIREN(eqx.Module):
    """SIREN model.

    This model is based on the SIREN architecture proposed by Sitzmann et al.
    The model consists of a series of SineLayer modules which are densely
    connected layers with a sinusoidal activation function with a specific
    initialization scheme.

    Args:
        - key: Random key.
        - in_features: Number of input features.
        - hidden_features: Number of hidden features.
        - hidden_layers: Number of hidden layers.
        - out_features: Number of output features.
        - outermost_linear: Whether the last layer is linear.
        - first_omega_0: Frequency of the first layer.
        - hidden_omega_0: Frequency of the hidden layers.

    [1] V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell, and G.
    Wetzstein, "Implicit Neural Representations with Periodic Activation
    Functions." arXiv, Jun. 17, 2020. Accessed: Mar. 08, 2024. [Online].
    Available: http://arxiv.org/abs/2006.09661
    """

    layers: list
    in_features: int
    hidden_features: int
    hidden_layers: int
    out_features: int

    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.0,
        key=jrandom.PRNGKey(0),
    ):
        # Initialize the model
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features

        keys = jax.random.split(key, hidden_layers + 2)
        first_key, *hidden_keys, last_key = keys

        self.layers = []

        # First layer
        self.layers.append(
            SineLayer(
                omega_0=first_omega_0,
                is_first=True,
                in_features=in_features,
                out_features=hidden_features,
                key=first_key,
            )
        )

        # Hidden layers
        for i in range(hidden_layers):
            self.layers.append(
                SineLayer(
                    omega_0=hidden_omega_0,
                    is_first=False,
                    in_features=hidden_features,
                    out_features=hidden_features,
                    key=hidden_keys[i],
                )
            )

        # Last layer
        if outermost_linear:
            init_key, last_key = jax.random.split(last_key)
            final_layer = eqx.nn.Linear(
                hidden_features, out_features, use_bias=True, key=last_key
            )

            # Initialize the weights
            lim = jnp.sqrt(6.0 / hidden_features) / hidden_omega_0
            new_weights = jrandom.uniform(
                init_key, (out_features, hidden_features), minval=-lim, maxval=lim
            )
            final_layer = eqx.tree_at(
                lambda layer: layer.weight, final_layer, new_weights
            )
            self.layers.append(final_layer)

        else:
            self.layers.append(
                SineLayer(
                    omega_0=hidden_omega_0,
                    is_first=False,
                    in_features=hidden_features,
                    out_features=out_features,
                    key=last_key,
                )
            )

    def __call__(self, *args):
        """Forward pass."""
        x = jnp.array([*args])  # Stack the input variables
        for layer in self.layers:
            x = layer(x)
        return x
