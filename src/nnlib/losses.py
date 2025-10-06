from typing import Callable

import jax
import jax.numpy as jnp
from equinox import filter_jit
from jaxtyping import Array, PyTree

from nnlib.pinn import WavePINN

criteria = {
    "mse": lambda x, y, axis=None: jnp.mean((x - y) ** 2, axis),
    "mae": lambda x, y, axis=None: jnp.mean(jnp.abs(x - y), axis),
}


def data_loss(
    params: PyTree,
    model: WavePINN,
    coords_vals: tuple[Array],
    criterion: Callable = criteria["mse"],
) -> float:
    *coords, vals = coords_vals
    n_dim = len(coords)

    # vectorize and parallelize
    batched_p_net = jax.vmap(model.p_net, in_axes=(None, *[0] * n_dim))
    parallel_p_net = jax.pmap(batched_p_net, in_axes=(None, *[0] * n_dim))

    # make prediction
    pred = parallel_p_net(params, *coords)
    return criterion(pred, vals)  # error measure


def pde_loss(
    params: PyTree,
    model: WavePINN,
    coords: tuple[Array],
    criterion: Callable = criteria["mse"],
) -> float:
    n_dim = len(coords)

    # vectorize and parallelize
    batched_p_net = jax.vmap(model.r_net, in_axes=(None, *[0] * n_dim))
    parallel_p_net = jax.pmap(batched_p_net, in_axes=(None, *[0] * n_dim))

    # make prediction
    pred = parallel_p_net(params, *coords)
    return criterion(pred, 0.0)  # error measure


def compute_loss(
    params: PyTree,
    model: WavePINN,
    batch: dict[str, tuple[Array]],
    losses: dict[str, Callable] = {"pde": pde_loss, "data": data_loss},
    criterion: Callable = criteria["mse"],
) -> tuple[float, dict[str, float]]:
    """
    Compute the total loss and auxiliary per-loss values.
    """
    computed_losses: dict[str, float] = {
        key: func(params, model, batch[key], criterion) for key, func in losses.items()
    }
    total_loss = jax.tree.reduce(lambda x, y: x + y, computed_losses)
    return total_loss, computed_losses


def compute_weighted_loss(
    params: PyTree,
    model: WavePINN,
    weights: dict[str, float],
    batch: dict[str, tuple[Array]],
    losses: dict[str, Callable] = {"pde": pde_loss, "data": data_loss},
    criterion: Callable = criteria["mse"],
) -> tuple[float, dict[str, float]]:
    computed_losses: dict[str, float] = {
        key: func(params, model, batch[key], criterion) for key, func in losses.items()
    }
    weighted_losses = jax.tree.map(lambda x, y: x * y, computed_losses, weights)
    total_loss = jax.tree.reduce(lambda x, y: x + y, weighted_losses)
    return total_loss, computed_losses


@filter_jit
def compute_weights(
    params: PyTree,
    model: WavePINN,
    batch: dict[str, tuple[Array]],
    losses: dict[str, Callable],
    criterion: Callable = criteria["mse"],
) -> dict[str, Array]:
    """Compute grad-norm-based weights for each loss in `losses` dict."""

    # compute gradient norms from each loss term
    grad_norms = {}
    for term, loss_fn in losses.items():
        grads = jax.jacrev(loss_fn)(params, model, batch[term], criterion)
        flat_grads = jax.tree.leaves(grads)[0].ravel()
        grad_norms[term] = jnp.linalg.norm(flat_grads)

    # aggregate all terms via mean
    mean_grad_norm = jnp.mean(jnp.stack(jax.tree.leaves(grad_norms)))

    # compute weights by mean/this (map to dict)
    weights = jax.tree.map(lambda this: (mean_grad_norm / this), grad_norms)

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
