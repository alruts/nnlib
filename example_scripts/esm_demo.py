from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array


class EquivalentSourceModel(eqx.Module):
    num_sources: int
    coords: Array
    amplitudes: Array
    frequency: float
    bounds: Sequence[tuple[float, float]] = eqx.field(static=True)
    wave_speed: float = 1.0  # default_wave_speed()

    def __init__(self, num_sources, bounds, frequency, *, key):
        self.num_sources = num_sources
        self.frequency = frequency
        self.bounds = bounds

        # Random coordinates
        coords_key, amplitude_key = jax.random.split(key)
        axes = [
            jax.random.uniform(subkey, (num_sources,), minval=start, maxval=end)
            for (start, end), subkey in zip(
                bounds, jax.random.split(coords_key, len(bounds))
            )
        ]
        self.coords = jnp.stack(axes, axis=1)

        # Random amplitudes
        key_r, key_theta = jax.random.split(amplitude_key)
        r = jnp.sqrt(2 * jax.random.uniform(key_r, (num_sources,)))
        theta = 2 * jnp.pi * jax.random.uniform(key_theta, (num_sources,))
        self.amplitudes = r * jnp.exp(1j * theta) / num_sources

    def __call__(self, probe):
        """Vectorized computation with coordinates clipped to bounds."""
        probe = jnp.array(probe)

        # Clip coords to bounds along each axis
        clipped_coords = jnp.stack(
            [
                jnp.clip(self.coords[:, i], self.bounds[i][0], self.bounds[i][1])
                for i in range(len(self.bounds))
            ],
            axis=1,
        )

        # Vectorized point source contributions
        def point_source(A, x, xx, f):
            R = jnp.sqrt(jnp.sum(jnp.abs(x - xx) ** 2))
            k = (2 * jnp.pi * f) / self.wave_speed
            return A * (jnp.exp(-1j * k * R) / R)

        contribs = jax.vmap(point_source, in_axes=(0, 0, None, None))(
            self.amplitudes, clipped_coords, probe, self.frequency
        )
        return jnp.sum(contribs)
