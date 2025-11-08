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


def pressure_source_loss(
    params: PyTree,
    model: HelmholtzPINN,
    source_params: PyTree,
    source_model: Callable,
    coords: tuple[Array],
    criterion: Callable = aggregated_metrics["mse"],
    *args,
) -> float:
    n_dim = len(coords)

    # vectorize and parallelize
    batched_fwd_model = jax.vmap(model.p_net, in_axes=(None, *[0] * n_dim))
    parallel_fwd_model = jax.pmap(batched_fwd_model, in_axes=(None, *[0] * n_dim))

    # vectorize and parallelize
    batched_unknown_model = jax.vmap(source_model, in_axes=(None, *[0] * n_dim))
    parallel_unknown_model = jax.pmap(
        batched_unknown_model, in_axes=(None, *[0] * n_dim)
    )

    # make predictions
    fwd_pred = parallel_unknown_model(params, *coords)
    unknown_pred = parallel_unknown_model(source_params, *coords)

    return criterion(fwd_pred, unknown_pred)


def flow_source_loss(
    params: PyTree,
    model: HelmholtzPINN,
    source_params: PyTree,
    source_model: Callable,
    coords_normals: tuple[Array],
    criterion: Callable = aggregated_metrics["mse"],
) -> float:
    n_dim = len(coords_normals)

    # vectorize and parallelize
    batched_fwd_model = jax.vmap(model.v_net, in_axes=(None, *[0] * n_dim))
    parallel_fwd_model = jax.pmap(batched_fwd_model, in_axes=(None, *[0] * n_dim))

    # vectorize and parallelize
    batched_unknown_model = jax.vmap(source_model, in_axes=(None, *[0] * n_dim))
    parallel_unknown_model = jax.pmap(
        batched_unknown_model, in_axes=(None, *[0] * n_dim)
    )

    # make predictions
    fwd_pred = parallel_unknown_model(params, *coords_normals)
    unknown_pred = parallel_unknown_model(source_params, *coords_normals)

    return criterion(fwd_pred, unknown_pred)


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
        key: func(params, model, batch[key], criterion, *extra_args.get(key, ()))
        for key, func in losses.items()
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
) -> tuple[float, dict[str, float]]:
    computed_losses = jax.tree.map(
        lambda loss_fn, batch_data: loss_fn(params, model, batch_data, criterion),
        losses,
        batch,
    )
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
):
    """Compute grad-norm-based weights for each loss in `losses` dict, supporting complex numbers."""

    grad_norms = {}
    for term, loss_fn in losses.items():
        # partially applied loss
        def loss_scalar(params):
            return loss_fn(params, model, batch[term], criterion)

        # `vjp` works for both real and complex cases
        y, vjp_fn = jax.vjp(loss_scalar, params)
        (grads,) = vjp_fn(jnp.ones_like(y))

        # Flatten the gradients and compute norms
        flat_grads = jnp.concatenate([g.ravel() for g in jax.tree.leaves(grads)])
        grad_norms[term] = jnp.sqrt(
            jnp.sum(jnp.abs(flat_grads) ** 2)
        )  # valid for complex numbers

    mean_grad_norm = jnp.mean(jnp.array(jax.tree.leaves(grad_norms)))

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
