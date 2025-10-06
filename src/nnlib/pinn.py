from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vmap

from nnlib.architectures import ModifiedMLP, PirateNet
from nnlib.feature_maps import PeriodicFeatures, RandomFourierFeatures
from nnlib.misc import (
    apply_model,
    default_medium_density,
    default_wave_speed,
    lift_to_args,
)
from nnlib.reparametrize import make_modified_siren, make_siren

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

    def vn_net(self, params, *args):
        """Forward computation of normal velocity"""
        return NotImplementedError()

    def zn_net(self, params, *args):
        """Forward computation of normal impedance"""
        return NotImplementedError()

    def r_net(self, params, *args):
        """Computation of PDE residual for variable spatial dimensions."""

        def p_fn(*coords):
            return self.p_net(params, *coords)

        # Stack inputs along a new axis
        inputs = jnp.stack(args)

        # Hessian: second derivatives w.r.t. all coordinates
        hess_p = jax.jacrev(lambda x: jax.jacrev(p_fn)(*x))(inputs)
        diag_hess = jnp.diag(hess_p)

        # Assume spatial first, then time
        laplacian = jnp.sum(diag_hess[:-1])
        p_tt = diag_hess[-1]
        residual = p_tt - (self.wave_speed**2) * laplacian

        return residual

    def _batch_vmap(self, pred_fn, params, *coords):
        """
        Vectorized mapping over all points in coords.
        coords: tuple of arrays of shape (batch, ...) or (grid_dim, ...)
        """
        # Determine which axes to map over: assume last axis is batch/grid
        n_coords = len(coords)
        # `params` not mapped, `coords` mapped over axis 0
        in_axes = (None,) + (0,) * n_coords
        return vmap(pred_fn, in_axes=in_axes)(params, *coords)

    def pressure_pred_fn(self, params, *coords):
        """Predict pressure over a grid of any spatial dimension."""
        return self._batch_vmap(self.p_net, params, *coords)

    def norm_velocity_pred_fn(self, params, *coords):
        """Predict particle velocity over a grid of any spatial dimension."""
        return self._batch_vmap(self.vn_net, params, *coords)

    def norm_impedance_pred_fn(self, params, *coords):
        """Predict impedance over a grid of any spatial dimension."""
        return self._batch_vmap(self.zn_net, params, *coords)

    @eqx.filter_jit
    def update(self, params, opt_state, opt, batches):
        return params, opt_state
