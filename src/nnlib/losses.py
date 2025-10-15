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
    batched_r_net = jax.vmap(model.r_net, in_axes=(None, *[0] * n_dim))
    parallel_r_net = jax.pmap(batched_r_net, in_axes=(None, *[0] * n_dim))

    # make prediction
    pred = parallel_r_net(params, *coords)
    return criterion(pred, 0.0)  # error measure


def impedance_loss(
    params: PyTree,
    model: WavePINN,
    impedance_params: PyTree,
    impedance_model: Callable,  # callable pytree
    coords: tuple[Array],
    normals: tuple[Array],
    criterion: Callable = criteria["mse"],
) -> float:
    # I have separate models for the impedance and the sound field
    # In the ...

    # the pressure at the boundary surely will give me the waveform??

    n_dim = len(coords)

    # vectorize and parallelize
    batched_p_net = jax.vmap(model.r_net, in_axes=(None, *[0] * n_dim))
    parallel_p_net = jax.pmap(batched_p_net, in_axes=(None, *[0] * n_dim))

    batched_vn_net = jax.vmap(model.vn_net, in_axes=(None, *[0] * n_dim * 2))
    parallel_vn_net = jax.pmap(batched_vn_net, in_axes=(None, *[0] * n_dim * 2))

    # make pinn prediction
    p_pred = parallel_p_net(params, *coords)
    vn_pred = parallel_vn_net(params, *coords, *normals)
    pred = p_pred / vn_pred

    # impedance model prediction
    impedance_model_pred = ...

    return criterion(pred, impedance_model_pred)  # error measure


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
    params,
    model,
    batch: dict[str, tuple],
    losses: dict[str, Callable],
    criterion=criteria["mse"],
):
    """Compute grad-norm-based weights for each loss in `losses` dict."""
    # Compute gradient norms for each loss term
    grad_norms = {}
    for term, loss_fn in losses.items():
        grads = jax.jacrev(loss_fn)(params, model, batch[term], criterion)
        flat_grads = jnp.concatenate([g.ravel() for g in jax.tree.leaves(grads)])
        grad_norms[term] = jnp.linalg.norm(flat_grads)

    grad_norm_values = jnp.stack(list(grad_norms.values()))

    # Compute mean grad norm
    mean_grad_norm = jnp.mean(grad_norm_values)

    # Compute weights
    weights = jax.tree.map(lambda x: (mean_grad_norm / x), grad_norms)

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
