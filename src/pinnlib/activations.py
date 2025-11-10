import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Complex, Float

from pinnlib.misc import split_real_and_imaginary_activation


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


class LearnableSplitTanhLinear(eqx.Module):
    """Applies a sine activation scaled by the given angular frequency."""

    scalar_re: Array
    scalar_im: Array

    def __call__(self, z: Float) -> Float:
        return self.scalar_re * jax.nn.tanh(z.real) + 1j * self.scalar_im * jax.nn.tanh(
            z.imag
        )


class SinActivation(eqx.Module):
    """Applies a sine activation scaled by the given angular frequency."""

    angular_frequency: float

    def __call__(self, x: Float) -> Float:
        return jnp.sin(self.angular_frequency * x)


class SplitSinActivation(eqx.Module):
    """Applies a sine-based activation to complex inputs using the given
    angular frequency."""

    angular_frequency: float

    def __call__(self, z: Complex) -> Complex:
        return jnp.sin(self.angular_frequency * z.real) + 1j * jnp.sin(
            self.angular_frequency * z.imag
        )


class LearnableSinActivation(eqx.Module):
    """Applies a sine-based activation to complex inputs using the given
    angular frequency."""

    angular_frequency: float

    def __call__(self, z: Complex) -> Complex:
        return jnp.sin(self.angular_frequency * z.real) + 1j * jnp.sin(
            self.angular_frequency * z.imag
        )


def identity_activation(x: Array) -> Array:
    return x
