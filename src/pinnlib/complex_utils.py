from collections.abc import Callable

from jax import numpy as jnp


def split_real_and_imaginary_activation(fn: Callable):
    """Split activation to apply separately to real and imaginary part."""
    return lambda z: fn(jnp.real(z)) + 1j * fn(jnp.imag(z))


def split_real_and_imaginary_metric(fn: Callable):
    """Split metric to apply separately to real and imaginary part."""
    return lambda x, xx: fn(x.real, xx.real) + 1j * fn(x.imag, xx.imag)


def split_real_and_imaginary_loss(fn: Callable):
    """Split loss to apply separately to real and imaginary part."""
    return lambda x, xx: fn(x.real, xx.real) + fn(x.imag, xx.imag)
