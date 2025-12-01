import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array


class Identity(eqx.Module):
    """Identity function that simply returns inputs unchanged.

    Example:
        >>> import jax.numpy as jnp
        >>> import equinox as eqx
        >>> transform = Identity()
        >>> x = jnp.array([0.0])
        >>> print(transform(x))
        [0.]
    """

    def __call__(self, x):
        return x


class PeriodicFeatures(eqx.Module):
    """Periodic embeddings with per-axis trainable periods applied to all inputs.

    Example:
        >>> import jax.numpy as jnp
        >>> import equinox as eqx
        >>> periods = (1.0, 2.0)
        >>> transform = PeriodicFeatures(periods=periods)
        >>> x = jnp.array([0.0])
        >>> print(transform(x))
        [1. 0.]
    """

    periods: tuple[float, ...]

    def __call__(self, x):
        # Apply cos/sin embedding to each axis
        return jnp.hstack(
            [
                jnp.array([jnp.cos(p * xi), jnp.sin(p * xi)])
                for xi, p in zip(x, self.periods)
            ]
        )


class RandomFourierFeatures(eqx.Module):
    """Fourier embeddings with random Gaussian kernel

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import equinox as eqx
        >>> key = jax.random.PRNGKey(0)
        >>> transform = RandomFourierFeatures(embed_scale=1.0, embed_dim=8, in_dim=3, key=key)
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> y = transform(x)
        >>> y.shape
        (8,)
    """

    embed_scale: float
    embed_dim: int
    kernel: Array

    def __init__(self, embed_scale, embed_dim, in_dim, *, key):
        _, key = jax.random.split(key, 2)

        assert embed_dim % 2 == 0, "Embedded dimension must be a positive multiple of 2"

        self.kernel = jax.random.normal(key, (in_dim, embed_dim // 2)) * embed_scale
        self.embed_scale = embed_scale
        self.embed_dim = embed_dim

    def __call__(self, x):
        proj = jnp.dot(x, self.kernel)
        return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)
