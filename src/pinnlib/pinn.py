from typing import Callable, Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import Dopri5, ODETerm, SaveAt, diffeqsolve
from jaxtyping import Complex, Float, PyTree

from pinnlib.architectures import (
    ModifiedMLP,
    PirateNet,
    make_modified_siren,
    make_siren,
)
from pinnlib.feature_maps import PeriodicFeatures, RandomFourierFeatures
from pinnlib.misc import (
    apply_model,
    default_medium_density,
    default_wave_speed,
)

arch_lib = {
    "modified_mlp": ModifiedMLP,
    "modified_siren": make_modified_siren,
    "mlp": eqx.nn.MLP,
    "siren": make_siren,
    "pirate_net": PirateNet,
}


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


class WavePINN(eqx.Module):
    """
    PINN for the acoustic wave equation with optional input embeddings.

    Models pressure fields, PDE residuals, and derived quantities
    (velocity, impedance) with support for batched grid predictions.
    """

    model: Callable
    embedding: PeriodicFeatures | RandomFourierFeatures | None = None
    wave_speed: float = default_wave_speed()
    medium_density: float = default_medium_density()

    @classmethod
    def create(
        cls,
        arch_name: Literal[
            "modified_mlp", "mlp", "pirate_net", "siren", "modified_siren"
        ],
        embedding: PeriodicFeatures | RandomFourierFeatures | None = None,
        wave_speed: float = default_wave_speed(),
        medium_density: float = default_medium_density(),
        pytree_transformation: Optional[Callable[[PyTree], PyTree]] = None,
        **arch_kwargs,
    ):
        match embedding:
            case PeriodicFeatures():
                arch_kwargs["in_size"] *= 2
            case RandomFourierFeatures():
                arch_kwargs["in_size"] = embedding.embed_dim
            case None:
                pass
            case _:
                raise ValueError(f"Unsupported embedding: {embedding}")

        arch = arch_lib[arch_name]
        model = (
            pytree_transformation(arch(**arch_kwargs))
            if pytree_transformation
            else arch(**arch_kwargs)
        )

        return cls(
            model=model,
            embedding=embedding,
            wave_speed=wave_speed,
            medium_density=medium_density,
        )

    def p_net(self, params: PyTree, *args: Float) -> Float:
        """Forward computation of pressure"""
        x = jnp.stack(args)

        if self.embedding:
            x = self.embedding(x)
        else:
            pass

        return apply_model(self.model, params, x)

    def r_net(self, params: PyTree, *args: Float) -> Float:
        """Computation of PDE residual for variable spatial dimensions.

        Validation with a polynomial model:
            >>> import jax.numpy as jnp
            >>> import equinox as eqx
            >>> polynomial = lambda x: x[0]**3 + x[1]**3
            >>> wave_speed = 2.0
            >>> pinn = WavePINN(model=polynomial, wave_speed=wave_speed)
            >>> params = eqx.filter(pinn.model, eqx.is_array)
            >>> print(pinn.r_net(params, 1.0, -1.0)) # 6(1) - (6/4)(-1) = 7.5
            7.5

        """

        def p_fn(*x):
            return self.p_net(params, *x)

        second_derivs = [
            jax.grad(lambda *x: jax.grad(p_fn, argnum)(*x), argnum)(*args)
            for argnum in range(len(args))
        ]

        p_tt = jnp.array(second_derivs[-1])
        laplacian = jnp.sum(jnp.array(second_derivs[:-1]))

        return laplacian - (1.0 / self.wave_speed**2) * p_tt

    def v_net(self, params: PyTree, *args: Float, saveat=SaveAt(t1=True)) -> Float:
        """Computation of PDE residual for variable spatial dimensions.

        Arguments:
            params: Model parameters
            *args: (coordinates..., tangents...) where
                coordinates = spatial coordinates (x, y, z, ...)
                tangents = unit normal vector components (nx, ny, nz, ...)

            saveat: specify time steps to return see `diffrax.SaveAt`. Defaults to t.

        >>> plane_wave = lambda x: jnp.sin(x[1] - x[0])
        >>> analytic = lambda x, t : -(-jnp.sin(t - x) - jnp.sin(x))
        >>> pinn = WavePINN(model=plane_wave, medium_density=1.0, wave_speed=1.0)
        >>> params = eqx.filter(pinn.model, eqx.is_array)

        >>> # It is safe to query at t=0
        >>> x0, t0 = 0.0, 0.0
        >>> print(jnp.isclose(pinn.v_net(params, x0, t0, 1.0), 0.0))
        True

        >>> # Test against analytic solution
        >>> x1, t1 = jnp.pi / 2, jnp.pi / 2
        >>> print(jnp.isclose(pinn.v_net(params, x1, t1, 1.0), analytic(x1, t1)))
        True

        # many points with saveat
        >>> ts = jnp.array([0, jnp.pi/4, jnp.pi/2])
        >>> vs = pinn.v_net(params, x1, t1, 1.0, saveat=SaveAt(ts=ts))
        >>> print(jnp.allclose(vs, analytic(x1, ts)))
        True


        """

        ndim = len(args) // 2

        # tangent has no time component
        coords, tangent = args[: ndim + 1], args[ndim + 1 :]
        *point, time = coords

        def p_fn(*x):
            return self.p_net(params, *x)

        def grad_p_fn(t, y, args):
            _, grad_p = jax.jvp(p_fn, (*point, t), (*tangent, 0.0))
            return grad_p

        # time integrate grad_p
        term = ODETerm(grad_p_fn)
        solver = Dopri5()
        y0 = 0.0  # initial condition

        solution = diffeqsolve(
            term, solver, t0=0, t1=time, dt0=0.01, y0=y0, saveat=saveat
        )

        if solution.ys is not None:
            int_p_dt = solution.ys.squeeze()
            return -1.0 / self.medium_density * int_p_dt
        else:
            return RuntimeError("Numerical integration failed :(")

    def z_net(self, params: PyTree, *args: Float) -> Float:
        """
        Compute directional impedance normalized to the medium.

        Arguments:
            params: Model parameters
            *args: (coordinates..., tangents...) where
                coordinates = spatial coordinates (x, y, z, ...)
                tangents = unit normal vector components (nx, ny, nz, ...)
            saveat: specify time steps to return see `diffrax.SaveAt`. Defaults to t.
        """
        ndim = len(args) // 2
        coords = args[: ndim + 1]  # time has no tangent component

        Z = self.p_net(params, *coords) / self.v_net(params, *args)
        return Z / (self.medium_density * self.wave_speed)


class HelmholtzPINN(eqx.Module):
    """
    PINN for the acoustic wave equation with optional input embeddings.

    Models pressure fields, PDE residuals, and derived quantities
    (velocity, impedance) with support for batched grid predictions.
    """

    model: Callable
    frequency: float
    embedding: PeriodicFeatures | RandomFourierFeatures | None = None
    wave_speed: float = default_wave_speed()
    medium_density: float = default_medium_density()

    @classmethod
    def create(
        cls,
        arch_name: Literal[
            "modified_mlp", "mlp", "pirate_net", "siren", "modified_siren"
        ],
        embedding: PeriodicFeatures | RandomFourierFeatures | None,
        frequency: float,
        wave_speed: float = default_wave_speed(),
        medium_density: float = default_medium_density(),
        pytree_transformation: Optional[Callable[[PyTree], PyTree]] = None,
        **arch_kwargs,
    ):
        match embedding:
            case PeriodicFeatures():
                arch_kwargs["in_size"] *= 2
            case RandomFourierFeatures():
                arch_kwargs["in_size"] = embedding.embed_dim
            case None:
                pass
            case _:
                raise ValueError(f"Unsupported embedding: {embedding}")

        arch = arch_lib[arch_name]
        model = (
            pytree_transformation(arch(**arch_kwargs))
            if pytree_transformation
            else arch(**arch_kwargs)
        )

        return cls(
            model=model,
            embedding=embedding,
            frequency=frequency,
            wave_speed=wave_speed,
            medium_density=medium_density,
        )

    def p_net(self, params: PyTree, *args: Float) -> Complex:
        """Forward computation of pressure"""
        x = jnp.stack(args)
        x = jnp.asarray(x, dtype=jnp.result_type(x, *jax.tree.leaves(params)))

        if self.embedding:
            x = self.embedding(x)
        else:
            pass

        return apply_model(self.model, params, x)

    def r_net(self, params: PyTree, *args: Float) -> Complex:
        """Computation of PDE residual for variable spatial dimensions.

        Validation with a polynomial model:
            >>> import jax.numpy as jnp
            >>> import equinox as eqx
            >>> polynomial = lambda x: x[0]**3 + x[1]**3 * 1j
            >>> wave_speed = 2.0 * jnp.pi
            >>> pinn = HelmholtzPINN(model=polynomial, frequency=1, wave_speed=wave_speed)
            >>> params = eqx.filter(pinn.model, eqx.is_array)
            >>> print(pinn.r_net(params, 1.0, 1.0)) # (6 + 6j) + (1 + 1j)
            (7+7j)

        Validation with an analytical Helmholtz solution (residual should be ~0):
            >>> u_analytic = lambda x: jnp.exp(1j * x[0])  # 1D Helmholtz solution for k=1
            >>> wave_speed = 2.0 * jnp.pi
            >>> pinn = HelmholtzPINN(model=u_analytic, frequency=1, wave_speed=wave_speed)
            >>> params = eqx.filter(pinn.model, eqx.is_array)
            >>> r = pinn.r_net(params, 0.5)  # Evaluate at x=0.5
            >>> print(abs(r) < 1e-12)
            True
        """

        def p_fn(*x):
            return self.p_net(params, *x)

        laplacian = complex_laplacian(p_fn)(*args)
        k = (2 * jnp.pi * self.frequency) / self.wave_speed

        return laplacian + (k**2) * self.p_net(params, *args)

    def v_net(self, params: PyTree, *args: Float) -> Complex:
        """
        Compute directional velocity via Euler's equation of motion.

        Arguments:
            params: Model parameters
            *args: (coordinates..., tangents...) where
                coordinates = spatial coordinates (x, y, z, ...)
                tangents = unit normal vector components (nx, ny, nz, ...)

        Validation with polynomial:
            >>> import jax.numpy as jnp
            >>> import equinox as eqx
            >>> p_fn = lambda x: x[0]**2 + 1j * x[1]**2
            >>> pinn = HelmholtzPINN(model=p_fn, frequency=1.0/(2.0 * jnp.pi), medium_density=1.0)
            >>> params = eqx.filter(pinn.model, eqx.is_array)
            >>> print(pinn.v_net(params, 0.0, 1.0, 0.0, 1.0))
            (-2+0j)
            >>> print(pinn.v_net(params, 1.0, 0.0, 1.0, 0.0))
            2j
        """

        def p_fn_real(*x):
            return self.p_net(params, *x).real

        def p_fn_imag(*x):
            return self.p_net(params, *x).imag

        ndim = len(args) // 2
        coords, tangent = args[:ndim], args[ndim:]

        # directional derivative
        _, dpdn_real = jax.jvp(p_fn_real, coords, tangent)
        _, dpdn_imag = jax.jvp(p_fn_imag, coords, tangent)
        dpdn = jnp.array(dpdn_real) + 1j * jnp.array(dpdn_imag)

        return -1.0 / (1j * 2.0 * jnp.pi * self.frequency * self.medium_density) * dpdn

    def z_net(self, params: PyTree, *args: Float) -> Complex:
        """
        Compute directional impedance normalized to the medium.

        Arguments:
            params: Model parameters
            *args: (coordinates..., tangents...) where
                coordinates = spatial coordinates (x, y, z, ...)
                tangents = unit normal vector components (nx, ny, nz, ...)

        """
        ndim = len(args) // 2
        coords = args[:ndim]

        Z = self.p_net(params, *coords) / self.v_net(params, *args)

        return Z / (self.medium_density * self.wave_speed)
