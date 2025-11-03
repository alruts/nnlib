from functools import partial
from typing import Callable

import jax
from jax import numpy as jnp
from jaxtyping import Array, Complex

from nnlib.misc import split_real_and_imaginary_activation


def split_tanh(z: Complex) -> Complex:
    """Applies tanh separately to the real and imaginary parts of a complex number."""
    return split_real_and_imaginary_activation(jax.nn.tanh)(z)


def cardioid(z: Complex) -> Complex:
    """Scales a complex number by a cardioid-shaped function based on its angle."""
    return 0.5 * (1 + jnp.cos(jnp.angle(z))) * z


def rotating_cardioid(z: Complex, b: Complex) -> Complex:
    """Applies a cardioid function with rotation determined by a complex bias."""
    arg = jnp.angle(z) + jnp.angle(b)
    cos_arg = jnp.cos(arg)
    return 0.5 * (1.0 + cos_arg) * z


def sin_activation(x: Array, angular_frequency: float) -> Array:
    return jnp.sin(angular_frequency * x)


def make_sin_at(angular_frequency: float) -> Callable:
    """Return sin function at `angular_frequency`"""
    return partial(sin_activation, angular_frequency=angular_frequency)


def split_periodic_activation(z: Complex, angular_frequency: float) -> Complex:
    return jnp.cos(angular_frequency * z.real) + 1j * jnp.sin(
        angular_frequency * z.imag
    )


def identity_activation(x: Array) -> Array:
    return x
