from collections.abc import Callable

from jax import numpy as jnp
from jaxtyping import Array, Complex


def split_real_and_imaginary_activation(fn: Callable) -> Callable[[Complex], Complex]:
    """
    Split activation function to apply separately to real and imaginary parts.

    Take a real-valued activation function and extend it to work with complex
    inputs by applying the function independently to the real and imaginary
    components.

    Args:
        fn: A real-valued activation function (e.g., jax.nn.tanh, jax.nn.relu)

    Returns:
        A function that applies the activation separately to real and imaginary parts

    Example:
        >>> import jax.nn as jnn
        >>> complex_tanh = split_real_and_imaginary_activation(jnn.tanh)
        >>> z = 1.0 + 2.0j
        >>> result = complex_tanh(z)
        >>> # Equivalent to: jnn.tanh(1.0) + 1j * jnn.tanh(2.0)
    """
    return lambda z: fn(jnp.real(z)) + 1j * fn(jnp.imag(z))


def split_real_and_imaginary_metric(
    fn: Callable,
) -> Callable[[Complex, Complex], Complex]:
    """
    Split metric function to apply separately to real and imaginary parts.

    Extend real-valued metric functions to work with complex inputs by
    computing the metric separately for real and imaginary components and
    combining them into a complex result.

    Args:
        fn: A real-valued metric function taking two arguments

    Returns:
        A function that computes the metric separately for real and imaginary parts

    Example:
        >>> def mse(x, y): return jnp.mean((x - y) ** 2)
        >>> complex_mse = split_real_and_imaginary_metric(mse)
        >>> z1, z2 = 1.0 + 2.0j, 1.1 + 1.9j
        >>> result = complex_mse(z1, z2)
    """
    return lambda x, xx: fn(x.real, xx.real) + 1j * fn(x.imag, xx.imag)


def split_real_and_imaginary_loss(fn: Callable) -> Callable[[Complex, Complex], Array]:
    """
    Split loss function to apply separately to real and imaginary parts.

    Extend real-valued loss functions to work with complex inputs by computing
    the loss separately for real and imaginary components and summing them to
    produce a real-valued loss.

    Args:
        fn: A real-valued loss function taking two arguments

    Returns:
        A function that computes the loss separately for real and imaginary parts
        and returns their sum as a real number

    Example:
        >>> def mse(x, y): return jnp.mean((x - y) ** 2)
        >>> complex_mse_loss = split_real_and_imaginary_loss(mse)
        >>> z1, z2 = 1.0 + 2.0j, 1.1 + 1.9j
        >>> loss = complex_mse_loss(z1, z2)
        >>> # Returns: mse(1.0, 1.1) + mse(2.0, 1.9)
    """
    return lambda x, xx: fn(x.real, xx.real) + fn(x.imag, xx.imag)
