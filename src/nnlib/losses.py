from typing import Callable

import jax
import jax.numpy as jnp
from equinox import filter_jit
from jaxtyping import Array, PyTree

from nnlib.pinn import WavePINN

point_wise_metrics = {
    # Element-wise error maps
    "abs_err": lambda p_ref, p_pred: jnp.abs(p_pred - p_ref),
    "sq_err": lambda p_ref, p_pred: (p_pred - p_ref) ** 2,
    "rel_err": lambda p_ref, p_pred, eps=1e-8: jnp.abs(p_pred - p_ref)
    / (jnp.abs(p_ref) + jnp.abs(p_pred) + eps),
    "log_err": lambda p_ref, p_pred: jnp.abs(
        jnp.log1p(jnp.abs(p_ref)) - jnp.log1p(jnp.abs(p_pred))
    ),
    "diff": lambda p_ref, p_pred: p_pred - p_ref,
}

aggregated_metrics = {
    "mse": lambda p_ref, p_pred: jnp.mean((p_pred - p_ref) ** 2),
    "rmse": lambda p_ref, p_pred: jnp.sqrt(jnp.mean((p_pred - p_ref) ** 2)),
    "mae": lambda p_ref, p_pred: jnp.mean(jnp.abs(p_pred - p_ref)),
    "nrmse_range": lambda p_ref, p_pred, eps=1e-8: (
        jnp.sqrt(jnp.mean((p_pred - p_ref) ** 2)) / (p_ref.max() - p_ref.min() + eps)
    ),
    "nrmse_std": lambda p_ref, p_pred, eps=1e-8: (
        jnp.sqrt(jnp.mean((p_pred - p_ref) ** 2)) / (jnp.std(p_ref) + eps)
    ),
    "mean_rel_err": lambda p_ref, p_pred, eps=1e-8: (
        jnp.mean(jnp.abs(p_pred - p_ref) / (jnp.abs(p_ref) + jnp.abs(p_pred) + eps))
    ),
    "split_mse": lambda x, y, axis=None: jnp.mean(
        (x.real - y.real) ** 2 + (x.imag - y.imag) ** 2, axis=axis
    ).real,
    "split_mae": lambda x, y, axis=None: jnp.mean(
        jnp.abs(x.real - y.real) + jnp.abs(x.imag - y.imag), axis=axis
    ).real,
    "mag_phase": lambda x, y, axis=None, alpha=1.0, beta=1.0: (
        alpha * jnp.mean((jnp.abs(x) - jnp.abs(y)) ** 2, axis=axis)
        + beta * jnp.mean(jnp.angle(jnp.exp(1j * (x - y))) ** 2, axis=axis)
    ),
}


def data_loss(
    params: PyTree,
    model: WavePINN,
    coords_vals: tuple[Array],
    criterion: Callable = aggregated_metrics["mse"],
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
    criterion: Callable = aggregated_metrics["mse"],
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
    criterion: Callable = aggregated_metrics["mse"],
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
    criterion: Callable = aggregated_metrics["mse"],
) -> tuple[float, dict[str, float]]:
    """
    Compute the total loss and auxiliary per-loss values.
    """
    computed_losses: dict[str, float] = {
        key: func(params, model, batch[key], criterion) for key, func in losses.items()
    }
    total_loss = jax.tree.reduce(lambda x, y: x + y, computed_losses)
    return total_loss, computed_losses


def make_loss(data_loss: Callable, criterion: Callable):
    """
    Returns a new loss function with the given criterion partially applied
    to the provided data_loss function.

    Usage:
        loss_fn = make_loss(data_loss, criteria["mse"])
        value = loss_fn(params, model, coords_vals)
    """

    def loss_fn(params: PyTree, model, coords_vals: tuple[Array]) -> float:
        return data_loss(params, model, coords_vals, criterion=criterion)

    return loss_fn


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
        grads = vjp_fn(jnp.ones_like(y))[0]

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
