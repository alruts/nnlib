import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Complex, Float

from pinnlib.complex_utils import split_real_and_imaginary_activation


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


class LearnableSplitTanh(eqx.Module):
    """Applies a split tanh activation with each part scaled by a and b respectively."""

    a: Float
    b: Float

    def __call__(self, z: Float) -> Float:
        return self.a * jax.nn.tanh(z.real) + 1j * self.b * jax.nn.tanh(z.imag)


class LearnableTanh(eqx.Module):
    """Applies a tanh activation scaled by a."""

    a: Float

    def __call__(self, z: Float) -> Float:
        return self.a * jax.nn.tanh(z)


class SinActivation(eqx.Module):
    """Applies a sine activation scaled by the given angular frequency."""

    angular_frequency: Float

    def __call__(self, x: Float) -> Float:
        return jnp.sin(self.angular_frequency * x)


class SplitSinActivation(eqx.Module):
    """Applies a sine-based activation to complex inputs using the given
    angular frequency."""

    angular_frequency: Float

    def __call__(self, z: Complex) -> Complex:
        return jnp.sin(self.angular_frequency * z.real) + 1j * jnp.sin(
            self.angular_frequency * z.imag
        )


class WaveletActivation(eqx.Module):
    a: Array
    b: Array

    def __call__(self, x):
        return self.a * jnp.sin(x) + self.b * jnp.cos(x)


def identity_activation(x: Array) -> Array:
    return x
