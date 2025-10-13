import jax
from jax import numpy as jnp
from jaxtyping import Complex

from nnlib.misc import split_activation


def complex_tanh(z: Complex) -> Complex:
    """Applies tanh separately to the real and imaginary parts of a complex number."""
    return split_activation(jax.nn.tanh)(z)


def cardioid(z: Complex) -> Complex:
    """Scales a complex number by a cardioid-shaped function based on its angle."""
    return 0.5 * (1 + jnp.cos(jnp.angle(z))) * z


def rotating_cardioid(z: Complex, b: Complex) -> Complex:
    """Applies a cardioid function with rotation determined by a complex bias."""
    arg = jnp.angle(z) + jnp.angle(b)
    cos_arg = jnp.cos(arg)
    return 0.5 * (1.0 + cos_arg) * z
