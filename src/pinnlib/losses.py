from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from equinox import filter_jit
from jaxtyping import Array, PyTree

from pinnlib.metrics import aggregated_metrics
from pinnlib.pinn import HelmholtzPINN, WavePINN


def data_loss(
    params: PyTree,
    model: WavePINN | HelmholtzPINN,
    coords_vals: tuple[Array],
    criterion: Callable = aggregated_metrics["mse"],
    *args,
) -> float:
    *coords, vals = coords_vals
    n_dim = len(coords)

    # vectorize and parallelize
    batched_p_net = jax.vmap(model.p_net, in_axes=(None, *[0] * n_dim))
    parallel_p_net = jax.pmap(batched_p_net, in_axes=(None, *[0] * n_dim))

    # make prediction
    pred = parallel_p_net(params, *coords)
    return criterion(pred, vals)  # error measure


def hom_pde_loss(
    params: PyTree,
    model: WavePINN,
    coords: tuple[Array],
    criterion: Callable = aggregated_metrics["mse"],
    *args,
) -> float:
    n_dim = len(coords)

    # vectorize and parallelize
    batched_r_net = jax.vmap(model.r_net, in_axes=(None, *[0] * n_dim))
    parallel_r_net = jax.pmap(batched_r_net, in_axes=(None, *[0] * n_dim))

    # make prediction
    pred = parallel_r_net(params, *coords)
    return criterion(pred, 0.0)  # error measure


def pressure_model_loss(
    forward_params: PyTree,
    forward_model: HelmholtzPINN,
    inverse_params: PyTree,
    inverse_model: HelmholtzPINN,
    coords: tuple[Array],
    criterion: Callable = aggregated_metrics["mse"],
    *args,
) -> float:
    n_dim = len(coords)

    # vectorize and parallelize
    batched_fwd_model = jax.vmap(forward_model.p_net, in_axes=(None, *[0] * n_dim))
    parallel_fwd_model = jax.pmap(batched_fwd_model, in_axes=(None, *[0] * n_dim))

    batched_inv_model = jax.vmap(inverse_model.p_net, in_axes=(None, *[0] * n_dim))
    parallel_inv_model = jax.pmap(batched_inv_model, in_axes=(None, *[0] * n_dim))

    fwd_pred = parallel_fwd_model(forward_params, *coords)
    inv_pred = parallel_inv_model(inverse_params, *coords)

    return criterion(fwd_pred, inv_pred)


def compute_loss(
    params: PyTree,
    model: WavePINN,
    batch: dict[str, tuple[Array]],
    losses: dict[str, Callable] = {"pde": hom_pde_loss, "data": data_loss},
    criterion: Callable = aggregated_metrics["mse"],
    extra_args: dict[str, Any] = {},
) -> tuple[float, dict[str, float]]:
    """
    Compute the total loss and auxiliary per-loss values.
    """
    computed_losses: dict[str, float] = {
        term: l_fn(params, model, batch[term], criterion, *extra_args.get(term, ()))
        for term, l_fn in losses.items()
    }
    total_loss = jax.tree.reduce(lambda x, y: x + y, computed_losses)
    return total_loss, computed_losses


def compute_weighted_loss(
    params: PyTree,
    model: WavePINN,
    weights: dict[str, float],
    batch: dict[str, tuple[Array]],
    losses: dict[str, Callable],
    criterion: Callable,
    extra_args: dict[str, Any] = {},
) -> tuple[float, dict[str, float]]:
    computed_losses: dict[str, float] = {
        term: l_fn(params, model, batch[term], criterion, *extra_args.get(term, ()))
        for term, l_fn in losses.items()
    }
    weighted_losses = jax.tree.map(lambda x, y: x * y, computed_losses, weights)
    total_loss = jax.tree.reduce(lambda x, y: x + y, weighted_losses)
    return total_loss, computed_losses


@filter_jit
def compute_weights(
    params,
    model,
    batch: dict[str, tuple],
    losses: dict[str, Callable],
    criterion=aggregated_metrics["mse"],
    extra_args: dict[str, Any] = {},
):
    """Compute grad-norm-based weights for each loss in `losses` dict, supporting complex numbers."""

    def grad_norm(loss_fn, term):
        """Compute L2 norm of gradients for a single loss term."""

        def loss_scalar(p):
            return loss_fn(p, model, batch[term], criterion, *extra_args.get(term, ()))

        y, vjp_fn = jax.vjp(loss_scalar, params)
        (grads,) = vjp_fn(jnp.ones_like(y))

        # Compute squared L2 norm over all parameters
        leaves = jax.tree.leaves(grads)
        norm_sq = jnp.sum(jnp.stack([jnp.sum(jnp.abs(g) ** 2) for g in leaves]))
        return jnp.sqrt(norm_sq)

    # Vectorize over the loss dict
    grad_norms = {term: grad_norm(loss_fn, term) for term, loss_fn in losses.items()}

    mean_grad_norm = jnp.mean(jnp.array(list(grad_norms.values())))

    # Compute weights
    weights = {term: mean_grad_norm / norm for term, norm in grad_norms.items()}

    return weights


@filter_jit
def update_weights(
    momentum: float, old_weights: dict[str, Array], new_weights: dict[str, Array]
):
    """Updates `weights` using running average with momentum."""

    def running_average(old_w, new_w):
        return old_w * momentum + (1 - momentum) * new_w

    weights = jax.tree.map(running_average, old_weights, new_weights)
    weights = jax.lax.stop_gradient(weights)
    return weights
