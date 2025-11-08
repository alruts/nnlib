import jax
import jax.numpy as jnp


def args_to_array(f):
    """
    Wraps a function f(*args) to f_array(x_array) where x_array is a 1D array of all arguments.
    Returns a function that splits x_array into individual arguments internally.
    """

    def wrapper(x_array):
        # Convert 1D array to tuple of scalars for f
        args = tuple(x_array)
        return f(*args)

    return wrapper


# todo: remove args syntax


def complex_laplacian(f):
    def wrapper(*args):
        # Convert to real vector: [Re(f), Im(f)]
        @args_to_array
        def f_realvec(*args):
            val = f(*args)
            return jnp.array([val.real, val.imag])

        # Hessian: jacobian of the gradient
        hess_split = jax.hessian(f_realvec)(jnp.array(args))
        hessian = hess_split[0] + 1j * hess_split[1]
        laplacian = jnp.trace(hessian)

        return laplacian

    return wrapper


# ---------------- Example usage ----------------
def complex_function(x):
    return jnp.exp(1j * x)


f_wrapped = laplacian(complex_function)

x = jnp.pi / 5
y = 2.0
hessian = f_wrapped(x)

print(f"x = {x}, y = {y}")
print(f"Hessian (d²f/dargs²) = {hessian}")
