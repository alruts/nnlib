from typing import Callable, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, PyTree

from nnlib.architectures import (
    ModifiedMLP,
    PirateNet,
    make_modified_siren,
    make_siren,
)
from nnlib.feature_maps import PeriodicFeatures, RandomFourierFeatures
from nnlib.misc import (
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
        wave_speed: float | None = None,
        medium_density: float | None = None,
        **arch_kwargs,
    ):
        model_cls = arch_lib[arch_name]

        match embedding:
            case PeriodicFeatures():
                arch_kwargs["in_size"] *= 2
            case RandomFourierFeatures():
                arch_kwargs["in_size"] = embedding.embed_dim
            case None:
                pass
            case _:
                raise ValueError(f"Unsupported embedding: {embedding}")

        model = model_cls(**arch_kwargs)

        # defaults
        wave_speed = wave_speed if wave_speed is not None else default_wave_speed()
        medium_density = (
            medium_density if medium_density is not None else default_medium_density()
        )

        return cls(
            model=model,
            embedding=embedding,
            wave_speed=wave_speed,
            medium_density=medium_density,
        )

    def p_net(self, params, *args):
        """Forward computation of pressure"""
        x = jnp.stack(args)

        if self.embedding:
            x = self.embedding(x)
        else:
            pass

        return apply_model(self.model, params, x)

    def r_net(self, params, *args):
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
        wave_speed: float | None = None,
        medium_density: float | None = None,
        **arch_kwargs,
    ):
        model_cls = arch_lib[arch_name]

        match embedding:
            case PeriodicFeatures():
                arch_kwargs["in_size"] *= 2
            case RandomFourierFeatures():
                arch_kwargs["in_size"] = embedding.embed_dim
            case eqx.nn.Identity():
                pass
            case _:
                raise ValueError(f"Unsupported embedding: {embedding}")

        model = model_cls(**arch_kwargs)

        # defaults
        wave_speed = wave_speed if wave_speed is not None else default_wave_speed()
        medium_density = (
            medium_density if medium_density is not None else default_medium_density()
        )

        return cls(
            model=model,
            embedding=embedding,
            frequency=frequency,
            wave_speed=wave_speed,
            medium_density=medium_density,
        )

    def p_net(self, params, *args):
        """Forward computation of pressure"""
        x = jnp.stack(args)

        if self.embedding:
            x = self.embedding(x)
        else:
            pass

        return apply_model(self.model, params, x)

    def r_net(self, params, *args):
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
        """

        def p_fn_real(*x):
            return self.p_net(params, *x).real

        def p_fn_imag(*x):
            return self.p_net(params, *x).imag

        second_derivs_real = [
            jax.grad(lambda *x: jax.grad(p_fn_real, argnum)(*x), argnum)(*args)
            for argnum in range(len(args))
        ]

        second_derivs_imag = [
            jax.grad(lambda *x: jax.grad(p_fn_imag, argnum)(*x), argnum)(*args)
            for argnum in range(len(args))
        ]

        laplacian = jnp.sum(
            jnp.array(second_derivs_real) + 1j * jnp.array(second_derivs_imag)
        )
        k = (2 * jnp.pi * self.frequency) / self.wave_speed

        return laplacian + (k**2) * self.p_net(params, *args)
