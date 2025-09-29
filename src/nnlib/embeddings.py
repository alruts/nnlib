from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp


class PeriodEmbs(eqx.Module):
    """Periodic embeddings with per-axis trainable periods applied to all inputs.

    Example:
        >>> import jax.numpy as jnp
        >>> import equinox as eqx
        >>> periods = (jnp.array(1.0), jnp.array(2.0))
        >>> emb = PeriodEmbs(periods=periods)
        >>> x = jnp.array([0.0])
        >>> emb(x)
        Array([1., 0.], dtype=float32)
    """

    periods: Tuple[jnp.ndarray, ...]

    def __call__(self, x):
        # Apply cos/sin embedding to each axis
        return jnp.hstack(
            [
                jnp.array([jnp.cos(p * xi), jnp.sin(p * xi)])
                for xi, p in zip(x, self.periods)
            ]
        )


class FourierEmbedding(eqx.Module):
    """Fourier embeddings with random Gaussian kernel

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import equinox as eqx
        >>> key = jax.random.PRNGKey(0)
        >>> emb = FourierEmbedding(embed_scale=1.0, embed_dim=8, in_dim=3, key=key)
        >>> x = jnp.array([1.0, 2.0, 3.0])
        >>> y = emb(x)
        >>> y.shape
        (8,)
    """

    embed_scale: float
    embed_dim: int
    kernel: jax.Array

    def __init__(self, embed_scale, embed_dim, in_dim, *, key):
        _, key = jax.random.split(key, 2)
        self.kernel = jax.random.normal(key, (in_dim, embed_dim // 2)) * embed_scale
        self.embed_scale = embed_scale
        self.embed_dim = embed_dim

    def __call__(self, x):
        proj = jnp.dot(x, self.kernel)
        return jnp.concatenate([jnp.cos(proj), jnp.sin(proj)], axis=-1)
