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
        - num_hidden: Number of hidden layers.
        - out_features: Number of output features.
        - outermost_linear: Whether the last layer is linear.
        - first_omega_0: Frequency of the first layer.
        - hidden_omega_0: Frequency of the hidden layers.

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import equinox as eqx
        >>> key = jax.random.PRNGKey(0)
        >>> siren = SIREN(in_features=3, hidden_features=4, num_hidden=2, out_features=2, key=key)
        >>> x, y, z = 0.1, 0.2, 0.3
        >>> y = siren(x, y, z)
        >>> isinstance(y, jax.Array)
        True
        >>> y.shape
        (2,)

    [1] V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell, and G.
    Wetzstein, "Implicit Neural Representations with Periodic Activation
    Functions." arXiv, Jun. 17, 2020. Accessed: Mar. 08, 2024. [Online].
    Available: http://arxiv.org/abs/2006.09661
    """

    layers: tuple[SineLayer, ...]
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    hidden_features: int = eqx.field(static=True)
    num_hidden: int = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_hidden: int,
        out_features: int,
        outermost_linear: bool = False,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        key: jax.Array = jrandom.PRNGKey(0),
    ):
        # Initialize the model
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_hidden = num_hidden
        self.out_features = out_features

        last_key, *keys = jax.random.split(key, num_hidden + 2)
        keys_iter = iter(keys)

        # First layer
        first_layer = SineLayer(
            omega_0=first_omega_0,
            is_first=True,
            in_features=in_features,
            out_features=hidden_features,
            key=next(keys_iter),
        )

        # Hidden layers
        hidden_layers = (
            SineLayer(
                omega_0=hidden_omega_0,
                is_first=False,
                in_features=hidden_features,
                out_features=hidden_features,
                key=next(keys_iter),
            )
            for _ in range(num_hidden)
        )

        # Last layer
        if outermost_linear:
            init_key, last_key = jax.random.split(last_key)
            last_layer = eqx.nn.Linear(
                hidden_features, out_features, use_bias=True, key=last_key
            )

            # Initialize the weights
            lim = jnp.sqrt(6.0 / hidden_features) / hidden_omega_0
            new_weights = jrandom.uniform(
                init_key, (out_features, hidden_features), minval=-lim, maxval=lim
            )
            last_layer = eqx.tree_at(
                lambda layer: layer.weight, last_layer, new_weights
            )

        else:
            last_layer = SineLayer(
                omega_0=hidden_omega_0,
                is_first=False,
                in_features=hidden_features,
                out_features=out_features,
                key=last_key,
            )

        # Compile layers into tuple
        self.layers = (first_layer, *hidden_layers, last_layer)

    def __call__(self, *args):
        """Forward pass."""
        x = jnp.array([*args])  # Stack the input variables
        for layer in self.layers:
            x = layer(x)
        return x
