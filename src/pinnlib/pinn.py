from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import Dopri5, ODETerm, SaveAt, diffeqsolve
from jaxtyping import Float, PyTree, Scalar

from pinnlib.feature_maps import PeriodicFeatures, RandomFourierFeatures
from pinnlib.misc import (
    apply_model,
    default_medium_density,
    default_wave_speed,
)
from pinnlib.nn import (
    ModifiedMLP,
    PirateNet,
    make_modified_siren,
    make_siren,
)

arch_lib = {
    "modified_mlp": ModifiedMLP,
    "modified_siren": make_modified_siren,
    "mlp": eqx.nn.MLP,
    "siren": make_siren,
    "pirate_net": PirateNet,
}


def wave_residual(
    model: Callable,
    params: PyTree,
    *args: Scalar,
    wave_speed: float,
    rhs: Callable = lambda *_: 0.0,
) -> Scalar:
    """
    Compute the wave-equation PDE residual:
        R = ∇² p − (1/c²) p_tt - rhs(x, t)

    >>> # Polynomial model
    >>> polynomial = lambda params, x, t: x**3 + t**3
    >>> params = {}  # no parameters
    >>> analytic = lambda x, t: 6 * x - (1/4) * 6 * t
    >>> wave_speed = 2.0
    >>> x0, t0 = 1.0, 1.0
    >>> print(
    ...     jnp.isclose(
    ...         wave_residual(polynomial, params, x0, t0, wave_speed=wave_speed),
    ...         analytic(x0, t0)
    ...     )
    ... )
    True

    """

    def p_fn(*x):
        return model(params, *x)

    second_derivs = [
        jax.jacrev(lambda *x: jax.jacfwd(p_fn, argnum)(*x), argnum)(*args)
        for argnum in range(len(args))
    ]

    p_tt = jnp.array(second_derivs[-1])
    laplacian = jnp.sum(jnp.array(second_derivs[:-1]))

    return laplacian - (1.0 / wave_speed**2) * p_tt - rhs(*args)


def wave_directional_velocity(
    model: Callable,
    params: PyTree,
    *args: Scalar,
    medium_density: float,
    saveat: SaveAt = SaveAt(t1=True),
) -> Scalar:
    """
    Compute directional particle velocity v(x,t) in the time domain.
    Arguments:
        *args = (x1, x2, ..., t, nx, ny, ...)

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from diffrax import SaveAt

    >>> # Trivial plane wave model
    >>> plane_wave = lambda params, x, t: jnp.sin(t - x)
    >>> medium_density = 1.0
    >>> analytic = lambda x, t: -(-jnp.sin(t - x) - jnp.sin(x))
    >>> params = {}  # no parameters

    >>> # Single point at t=0
    >>> x0, t0 = 0.0, 0.0
    >>> print(
    ...     jnp.isclose(
    ...         wave_directional_velocity(plane_wave, params, x0, t0, 1.0, medium_density=medium_density),
    ...         analytic(x0, t0)
    ...     )
    ... )
    True

    >>> # Single point at t=pi/2
    >>> x1, t1 = jnp.pi/2, jnp.pi/2
    >>> print(
    ...     jnp.isclose(
    ...     wave_directional_velocity(plane_wave, params, x1, t1, 1.0, medium_density=medium_density),
    ...     analytic(x1, t1)
    ...     )
    ... )
    True

    >>> # Multiple time points using SaveAt
    >>> ts = jnp.array([0.0, jnp.pi/4, jnp.pi/2])
    >>> vs = wave_directional_velocity(plane_wave, params, x1, t1, 1.0, saveat=SaveAt(ts=ts), medium_density=medium_density)
    >>> print(jnp.allclose(vs, analytic(x1, ts)))
    True
    """
    ndim = len(args) // 2
    coords, tangent = args[: ndim + 1], args[ndim + 1 :]
    *spatial_point, time = coords

    def p_fn(*x):
        return model(params, *x)

    def dpdn_time_derivative(t, y, _):
        _, dpdn = jax.jvp(p_fn, (*spatial_point, t), (*tangent, 0.0))
        return dpdn

    term = ODETerm(dpdn_time_derivative)
    solver = Dopri5()

    sol = diffeqsolve(term, solver, t0=0.0, t1=time, dt0=0.01, y0=0.0, saveat=saveat)

    if sol.ys is None:
        raise RuntimeError("Directional velocity integration failed.")

    return -sol.ys.squeeze() / medium_density


def wave_impedance(
    params: PyTree,
    model: Callable,
    *args: Scalar,
    wave_speed: float,
    medium_density: float,
) -> Scalar:
    """Normalized acoustic impedance Z / (ρc) in the time domain."""
    ndim = len(args) // 2
    coords = args[: ndim + 1]

    p = model(params, *coords)
    v = wave_directional_velocity(model, params, *args, medium_density=medium_density)

    return p / (v * medium_density * wave_speed)


def helmholtz_residual(
    model: Callable,
    params: PyTree,
    *args: Scalar,
    wave_speed: float,
    frequency: float,
    rhs: Callable = lambda *_: 0.0,
) -> Scalar:
    """Computation of PDE residual for variable spatial dimensions.

    Validation with a polynomial model:

    >>> import jax.numpy as jnp
    >>> import equinox as eqx
    >>> frequency = 1.0
    >>> wave_speed = 1.0
    >>> k = 2 * jnp.pi * frequency / wave_speed
    >>> params = {}
    >>> polynomial = lambda params, x, y: x**3 + y**3 * 1j
    >>> analytic = lambda x, y: (6 * x + 6 * y * 1j) + k**2 * polynomial(params, x, y)
    >>> x0, y0 = 1.0, 1.0
    >>> print(
    ...     jnp.isclose(
    ...     helmholtz_residual(polynomial, params, x0, y0, wave_speed=1.0, frequency=1.0),
    ...     analytic(x0, y0)
    ...     )
    ... )
    True
    """

    def p_fn(*x):
        return jnp.array([model(params, *x).real, model(params, *x).imag])

    # Forward-over-reverse AD for laplacian
    second_derivs = [
        jax.jacfwd(lambda *x: jax.jacrev(p_fn, argnum)(*x), argnum)(*args)
        for argnum in range(len(args))
    ]

    laplacian = jnp.sum(jnp.array(second_derivs), axis=-1)
    laplacian = laplacian[0] + 1j * laplacian[1]

    k = (2 * jnp.pi * frequency) / wave_speed

    return laplacian + (k**2) * model(params, *args) - rhs(*args)


def helmholtz_directional_velocity(
    model: Callable,
    params: PyTree,
    *args: Scalar,
    frequency: float,
    medium_density: float,
) -> Scalar:
    """
    Compute directional velocity via Euler's equation of motion.

    Arguments:
        params: Model parameters
        *args: (coordinates..., tangents...) where
            | coordinates = spatial coordinates x, y, z, ...
            | tangents = unit normal vector components nx, ny, nz, ...

    Validation with polynomial:

    >>> import jax.numpy as jnp
    >>> import equinox as eqx
    >>> medium_density = 1.0
    >>> frequency = 1.0
    >>> params = {}
    >>> p_fn = lambda params, x, y: x**2 + 1j * y**2
    >>> analytic_x = lambda x, y: -1.0 / (1j * 2.0 * jnp.pi * frequency * medium_density) * 2*x
    >>> analytic_y = lambda x, y: -1.0 / (1j * 2.0 * jnp.pi * frequency * medium_density) * 2j*y
    >>> x0, y0 = 1.0, 1.0
    >>> print(
    ...     jnp.isclose(
    ...     helmholtz_directional_velocity(p_fn, params, x0, y0, 1.0, 0.0, medium_density=medium_density, frequency=frequency),
    ...     analytic_x(x0, y0)
    ...     )
    ... )
    True

    >>> print(
    ...     jnp.isclose(
    ...     helmholtz_directional_velocity(p_fn, params, x0, y0, 0.0, 1.0, medium_density=medium_density, frequency=frequency),
    ...     analytic_y(x0, y0)
    ...     )
    ... )
    True

    """

    # map return type from C to R^2
    def p_fn(*x):
        return model(params, *x)

    # number of spatial dimensions
    ndim = len(args) // 2
    coords, tangent = args[:ndim], args[ndim:]

    # directional derivative via Jacobi-vector product
    _, dpdn = jax.jvp(p_fn, coords, tangent)

    return -1.0 / (1j * 2.0 * jnp.pi * frequency * medium_density) * dpdn


def helmholtz_impedance(
    model: Callable,
    params: PyTree,
    *args: Scalar,
    frequency: float,
    wave_speed: float,
    medium_density: float,
) -> Scalar:
    """
    Compute directional impedance normalized to the medium.

    Arguments:
      params: Model parameters
      args: (coordinates..., tangents...) where
        - coordinates = spatial coordinates x, y, z, ...
        - tangents = unit normal vector components nx, ny, nz, ...

    """
    ndim = len(args) // 2
    coords = args[:ndim]

    Z = model(params, *coords) / helmholtz_directional_velocity(
        params, *args, medium_density=medium_density, frequency=frequency
    )

    return Z / (medium_density * wave_speed)


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
        pytree_transformation: Callable[[PyTree], PyTree] | None = None,
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

    def __call__(self, params: PyTree, *args: Float) -> Float:
        """Forward computation of pressure"""
        x = jnp.stack(args)

        if self.embedding:
            x = self.embedding(x)
        else:
            pass

        return apply_model(self.model, params, x)

    def residual(self, params: PyTree, *args: Scalar, rhs=lambda *_: 0.0) -> Scalar:
        return wave_residual(
            params, self.model, *args, rhs=rhs, wave_speed=self.wave_speed
        )

    def velocity(self, params: PyTree, *args: Scalar) -> Scalar:
        return wave_directional_velocity(
            params, self.model, *args, medium_density=self.medium_density
        )

    def impedance(self, params: PyTree, *args: Scalar) -> Scalar:
        return wave_impedance(
            params,
            self.model,
            *args,
            wave_speed=self.wave_speed,
            medium_density=self.medium_density,
        )


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
        pytree_transformation: Callable[[PyTree], PyTree] | None = None,
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

    def __call__(self, params: PyTree, *args: Scalar) -> Scalar:
        """Forward computation of pressure"""
        x = jnp.stack(args)
        x = jnp.asarray(x, dtype=jnp.result_type(x, *jax.tree.leaves(params)))

        if self.embedding:
            x = self.embedding(x)
        else:
            pass

        return apply_model(self.model, params, x)

    def residual(self, params: PyTree, *args: Scalar, rhs=lambda *_: 0.0) -> Scalar:
        return helmholtz_residual(
            self,
            params,
            *args,
            rhs=rhs,
            wave_speed=self.wave_speed,
            frequency=self.frequency,
        )

    def velocity(self, params: PyTree, *args: Scalar) -> Scalar:
        return helmholtz_directional_velocity(
            self,
            params,
            *args,
            medium_density=self.medium_density,
            frequency=self.frequency,
        )

    def impedance(self, params: PyTree, *args: Scalar) -> Scalar:
        return helmholtz_impedance(
            self,
            params,
            *args,
            wave_speed=self.wave_speed,
            medium_density=self.medium_density,
            frequency=self.frequency,
        )
