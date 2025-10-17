from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vmap

from nnlib.architectures import ModifiedMLP, PirateNet, make_modified_siren, make_siren
from nnlib.feature_maps import PeriodicFeatures, RandomFourierFeatures
from nnlib.misc import (
    apply_model,
    default_medium_density,
    default_wave_speed,
    lift_to_args,
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

    model: eqx.Module
    embedding: PeriodicFeatures | RandomFourierFeatures | eqx.nn.Identity
    wave_speed: float = default_wave_speed()
    medium_density: float = default_medium_density()

    @classmethod
    def create(
        cls,
        arch_name: Literal[
            "modified_mlp", "mlp", "pirate_net", "siren", "modified_siren"
        ],
        embedding: PeriodicFeatures | RandomFourierFeatures | eqx.nn.Identity,
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
            wave_speed=wave_speed,
            medium_density=medium_density,
        )

    def p_net(self, params, *args):
        """Forward computation of pressure"""
        # apply embeddings
        input_arr = lift_to_args(self.embedding)(*args)
        return apply_model(self.model, params, input_arr)

    def r_net(self, params, *args):
        """Computation of PDE residual for variable spatial dimensions."""

        def p_fn(*coords):
            return self.p_net(params, *coords)

        inputs = jnp.stack(args)
        hess_p = jax.jacrev(lambda x: jax.jacrev(p_fn)(*x))(inputs)
        diag_hess = jnp.diag(hess_p)

        # Assume spatial first, then time
        laplacian = jnp.sum(diag_hess[:-1])
        p_tt = diag_hess[-1]

        return p_tt - (1.0 / self.wave_speed**2) * laplacian


class HelmholtzPINN(eqx.Module):
    """
    PINN for the acoustic wave equation with optional input embeddings.

    Models pressure fields, PDE residuals, and derived quantities
    (velocity, impedance) with support for batched grid predictions.
    """

    model: eqx.Module
    embedding: PeriodicFeatures | RandomFourierFeatures | eqx.nn.Identity
    frequency: float
    wave_speed: float = default_wave_speed()
    medium_density: float = default_medium_density()

    @classmethod
    def create(
        cls,
        arch_name: Literal[
            "modified_mlp", "mlp", "pirate_net", "siren", "modified_siren"
        ],
        embedding: PeriodicFeatures | RandomFourierFeatures | eqx.nn.Identity,
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
        # apply embeddings
        input_arr = lift_to_args(self.embedding)(*args)
        return apply_model(self.model, params, input_arr)

    def r_net(self, params, *args):
        """Computation of PDE residual for variable spatial dimensions."""

        def p_fn_real(x):
            return jnp.real(self.p_net(params, *x)).astype(jnp.float32)

        def p_fn_imag(x):
            return jnp.imag(self.p_net(params, *x)).astype(jnp.float32)

        k = (2 * jnp.pi * self.frequency) / self.wave_speed

        inputs = jnp.array(args)  # shape: (n_args,)
        hess_real = jax.jacrev(jax.jacrev(p_fn_real))(inputs)
        hess_imag = jax.jacrev(jax.jacrev(p_fn_imag))(inputs)

        diag_hess = jnp.diag(hess_real + 1j * hess_imag)
        laplacian = jnp.sum(diag_hess)

        return laplacian + (k**2) * self.p_net(params, *args)


# toy net
pinn = HelmholtzPINN.create(
    embedding=eqx.nn.Identity(),
    arch_name="modified_mlp",
    frequency=1,
    in_size=2,
    out_size="scalar",
    width_size=4,
    dtype=jax.numpy.complex64,
    depth=3,
    key=jax.random.PRNGKey(0),
)
params, _ = eqx.partition(pinn.model, filter_spec=eqx.is_array)
print(pinn.r_net(params, 1.0, 1.0))
