from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp


class PeriodEmbs(eqx.Module):
    """Periodic embeddings with per-axis trainable periods applied to all inputs."""

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
    """Fourier embeddings with random Gaussian kernel"""

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
