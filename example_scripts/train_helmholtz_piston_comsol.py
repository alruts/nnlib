import pickle
from collections.abc import Callable
from pathlib import Path

import equinox as eqx
import jax
import optax
import trimesh
from jax import Array
from jax import numpy as jnp
from jax import random as jrandom
from jaxtyping import PyTree
from matplotlib import pyplot as plt
from soap_jax import soap
from tqdm import tqdm

from pinnlib import feature_maps
from pinnlib.activations import (
    LearnableSplitTanh,
    SplitSinActivation,
)
from pinnlib.data_utils import (
    DataPointGenerator,
    MeshGenerator,
    PointCloud,
    UniformGenerator,
    pc_utils,
)
from pinnlib.logger import TensorboardLogger
from pinnlib.losses import (
    aggregated_metrics,
    compute_mask,
    compute_weighted_loss,
    compute_weights,
    data_loss,
    hom_pde_loss,
    update_weights,
)
from pinnlib.metrics import sq_error
from pinnlib.misc import (
    default_complex_dtype,
    default_wave_speed,
    split_real_and_imaginary_loss,
    split_real_and_imaginary_metric,
)
from pinnlib.pinn import HelmholtzPINN

seed_key = jrandom.PRNGKey(0)
data_key, subsample_key, net_key, emb_key, dom_key, bnd_key = jrandom.split(seed_key, 6)

# load data structure from .pkl file
data_path = Path("./data/baffled_piston.pkl")
with open(data_path, "rb") as f:
    data: tuple[PointCloud, PointCloud, dict] = pickle.load(f)
    pressure_pc, velocity_pc, meta_data = data

# ...
frequency = meta_data["frequency"]
piston_radius = meta_data["piston_radius"]
piston_velocity = 0.02
wave_speed = default_wave_speed()

# Derived acoustic quantities
angular_frequency = 2 * jnp.pi * frequency
wavenumber = angular_frequency / wave_speed
wavelength = 2 * jnp.pi / wavenumber

# Observation grid
grid_extent = 0.5 * wavelength + piston_radius
lower_bound = 1e-9  # lower bound for hom pde loss

# data pipeline
data_pipe = pc_utils.pipe(
    pc_utils.filter_points(lambda c, _: c[-1] <= 0.5 * wavelength),
    pc_utils.filter_points(lambda c, _: c[-1] > 1e-2),
    pc_utils.sample_points(subsample_key, 128),
)
data_pc = data_pipe(pressure_pc)

data_generator = DataPointGenerator(
    point_cloud=data_pc,
    batch_size=128,
    key=data_key,
)

domain_generator = UniformGenerator(
    [
        (-grid_extent, grid_extent),
        (-grid_extent, grid_extent),
        (lower_bound, 0.5 * wavelength),
    ],
    batch_size=256,
    key=dom_key,
)

mesh = trimesh.load_mesh("./data/baffle.stl")
mesh_generator = MeshGenerator(mesh, batch_size=32, key=bnd_key)


# Filter evaluation data on surface plane
zero_z_filter = pc_utils.filter_points(lambda coord, _: coord[2] < 1e-12)
eval_pressure_pc = zero_z_filter(pressure_pc)
eval_velocity_pc = zero_z_filter(velocity_pc)

# Here we define which loss functions to use during training
# The losses should be parallelized and vectorized via `pmap` and `vmap`


def boundary_velocity_loss(
    fwd_params: PyTree,
    fwd_model: HelmholtzPINN,
    coords_normals: tuple[Array, Array],
    criterion: Callable,
    inv_params: PyTree,
    inv_model: Callable,
):
    pts, tangents = coords_normals
    n_dim = len(pts)

    # vectorize and parallelize
    batched_fwd_model = jax.vmap(fwd_model.velocity, in_axes=(None, *[0] * n_dim * 2))
    parallel_fwd_model = jax.pmap(batched_fwd_model, in_axes=(None, *[0] * n_dim * 2))

    batched_inv_model = jax.vmap(inv_model, in_axes=(None, *[0] * n_dim * 2))
    parallel_inv_model = jax.pmap(batched_inv_model, in_axes=(None, *[0] * n_dim * 2))

    fwd_pred = parallel_fwd_model(fwd_params, *pts, *tangents)
    inv_pred = parallel_inv_model(inv_params, *pts, *tangents)

    return criterion(fwd_pred, inv_pred)


losses = {"data": data_loss, "pde": hom_pde_loss, "bnd": boundary_velocity_loss}
update_weights_every = 1000

# Initialize the weights for adaptive grad norm, these are updated after the first step
# and every `update_weights_every` steps after that
loss_weights = {key: jnp.array(1.0) for key in losses.keys()}


# Helpers for logging
@eqx.filter_jit
def predict_pressure(params, pinn):
    """Evaluates pressure over the same grid as original dataset"""
    p_fn = jax.vmap(pinn, [None, 0, 0, None])
    x, y, _ = eval_pressure_pc.coords
    return p_fn(params, x, y, 0.0)


@eqx.filter_jit
def predict_velocity(params, pinn):
    """Evaluates pressure over the same grid as original dataset"""
    v = jax.vmap(pinn.velocity, [None, 0, 0, None, None, None, None])
    x, y, _ = eval_pressure_pc.coords
    return v(params, x, y, 0.0, 0.0, 0.0, 1.0)


def plot_pred(pressure):
    fig, ax = plt.subplots()
    x, y, _ = eval_pressure_pc.coords
    pc = ax.scatter(x, y, c=pressure, cmap="seismic", s=5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.colorbar(pc, ax=ax)

    # Draw a dashed circle using plt.plot
    theta = jnp.linspace(0, 2 * jnp.pi, 200)
    x = piston_radius * jnp.cos(theta)
    y = piston_radius * jnp.sin(theta)
    ax.plot(x, y, "k--", linewidth=2)  # 'w--' = white dashed line

    ax.set_aspect("equal", "box")
    return fig


# These are infinitely iterable
infinite_data_loader = iter(data_generator)
infinite_point_loader = iter(domain_generator)
infinite_disk_loader = iter(mesh_generator)

# Initialize logger
logger = TensorboardLogger(experiment_name=f"test-run_{len(data_generator)}")
logger.log_plot("plots/p_gt_mag", plot_pred, jnp.abs(eval_pressure_pc.vals), 0)
logger.log_plot("plots/p_gt_phase", plot_pred, jnp.angle(eval_pressure_pc.vals), 0)
logger.log_plot("plots/v_gt_mag", plot_pred, jnp.abs(eval_velocity_pc.vals), 0)
logger.log_plot("plots/v_gt_phase", plot_pred, jnp.angle(eval_pressure_pc.vals), 0)
log_every = 1000

# Random Fourier features for input coordinates helps with low-frequency bias
rff = feature_maps.RandomFourierFeatures(
    embed_scale=1 / jnp.sqrt(wavelength), embed_dim=128, in_dim=3, key=emb_key
)

# build PINN
pinn = HelmholtzPINN.create(
    embedding=rff,
    arch_name="siren",
    frequency=frequency,
    in_size=3,
    out_size="scalar",
    width_size=16,
    depth=3,
    dtype=default_complex_dtype(),
    first_activation=SplitSinActivation(float(wavenumber)),
    activation=SplitSinActivation(30.0),
    final_activation=LearnableSplitTanh(jnp.array(1.0), jnp.array(1.0)),
    key=net_key,
)


# v_n is a circular 'step function'
def velocity_model(params, *x):
    alpha, velocity = params
    # unpack coordinates
    x = jnp.array(x)
    r = jnp.linalg.norm(x[:2])  # radius in the xy-plane

    # smooth indicator
    inside_smooth = jax.nn.sigmoid(-(r - piston_radius) * alpha)

    # velocity is params inside the radius, 0 outside, smooth transition
    return inside_smooth * velocity


velocity_params = (jnp.array(50.0), jnp.array(piston_velocity))

# Extract the trainable parameters of the neural-net as a `PyTree`
pinn_params, _ = eqx.partition(pinn.model, filter_spec=eqx.is_array)
pinn_params = jax.tree.map(
    lambda x: x / jnp.sqrt(2), pinn_params
)  # scaling due to complex


# Initialize optimizers
learning_rate = optax.schedules.exponential_decay(1e-3, 2000, 0.9)
fwd_optimizer = optax.contrib.split_real_and_imaginary(
    soap(learning_rate, precondition_frequency=2)
)
learning_rate = optax.schedules.exponential_decay(1e-2, 10_000, 0.9)
inv_optimizer = optax.contrib.split_real_and_imaginary(optax.sgd(learning_rate))

fwd_opt_state = fwd_optimizer.init(pinn_params)
inv_opt_state = inv_optimizer.init(velocity_params)


# Define a single training step, `filter_jit` compiles this code to
# machine-code


@eqx.filter_jit
def fwd_train_step(model, params, inv_model, inv_params, opt_state, weights, batch):
    # extra arguments for boundary loss
    extra_args = {"bnd": (inv_params, inv_model)}
    (total, each_term), grads = jax.value_and_grad(compute_weighted_loss, has_aux=True)(
        params,
        model=model,
        batch=batch,
        weights=weights,
        losses=losses,
        criterion=split_real_and_imaginary_loss(aggregated_metrics["mse"]),
        extra_args=extra_args,
    )
    grads_conj = jax.tree.map(jnp.conj, grads)
    updates, opt_state = fwd_optimizer.update(grads_conj, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, (total, each_term)


@eqx.filter_jit
def inv_train_step(model, params, inv_model, inv_params, opt_state, batch):
    # extra arguments for boundary loss
    val, grads = jax.value_and_grad(boundary_velocity_loss, argnums=4)(
        params,
        model,
        batch["bnd"],
        split_real_and_imaginary_loss(aggregated_metrics["mse"]),
        inv_params,
        inv_model,
    )
    grads_conj = jax.tree.map(jnp.conj, grads)
    updates, opt_state = inv_optimizer.update(grads_conj, opt_state, inv_params)
    inv_params = optax.apply_updates(inv_params, updates)
    return inv_params, opt_state, val


# Training loop: all the optimization work is done here + logging
total_steps = int(50e3) + 1
for step, data_batch, pde_batch, bnd_batch in tqdm(
    zip(
        range(total_steps),
        infinite_data_loader,
        infinite_point_loader,
        infinite_disk_loader,
    ),
    total=total_steps,
):
    # The batch dictionary should have the same structure as `losses`
    batch = {"data": data_batch, "pde": pde_batch, "bnd": bnd_batch}

    # Do step
    pinn_params, fwd_opt_state, (loss, individual_losses) = fwd_train_step(
        pinn,
        pinn_params,
        velocity_model,
        velocity_params,
        fwd_opt_state,
        loss_weights,
        batch,
    )

    # adaptive inverse model
    velocity_params, inv_opt_state, inv_loss = inv_train_step(
        pinn,
        pinn_params,
        velocity_model,
        velocity_params,
        inv_opt_state,
        batch,
    )

    # Update adaptive weights
    extra_args = {"bnd": (velocity_params, velocity_model)}
    if step % update_weights_every == 0:
        new_weights = compute_weights(
            pinn_params, pinn, batch, losses, extra_args=extra_args
        )
        loss_weights = update_weights(0.9, loss_weights, new_weights)

    # Boundary condition masking
    filtered_loss_terms = {k: v for k, v in individual_losses.items() if k != "bnd"}
    filtered_loss_sum = sum(filtered_loss_terms.values())
    if step == 0:
        b = filtered_loss_sum  # save first step to calibrate masking fn
    loss_weights["bnd"] *= compute_mask(filtered_loss_sum, 1.0, b, 1e-3)  # pyright: ignore

    if step % log_every == 0:
        logger.log_scalar("loss/total", loss, step)
        logger.log_scalar("inv/loss", inv_loss, step)
        logger.log_scalar(
            "inv/percent-relative-velocity-error",
            100 * (jnp.abs(velocity_params[1] - piston_velocity) / piston_velocity),
            step,
        )

        logger.log_scalar(
            "inv/v_pred-minus-v_gt",
            velocity_params[1] - piston_velocity,
            step,
        )

        for term, weight in loss_weights.items():
            logger.log_scalar(f"weight/{term}", weight, step)

        for term, loss in individual_losses.items():
            logger.log_scalar(f"loss/{term}", loss / loss_weights.get(term), step)

        # make a prediction
        p_pred = predict_pressure(pinn_params, pinn)
        v_pred = predict_velocity(pinn_params, pinn)

        logger.log_plot("plots/p_pred_mag", plot_pred, jnp.abs(p_pred), step)
        logger.log_plot("plots/p_pred_phase", plot_pred, jnp.angle(p_pred), step)
        logger.log_plot("plots/v_pred_mag", plot_pred, jnp.abs(v_pred), step)
        logger.log_plot("plots/v_pred_phase", plot_pred, jnp.angle(v_pred), step)

        # compute errors
        p_pred_error = split_real_and_imaginary_metric(aggregated_metrics["mse"])(
            p_pred, eval_pressure_pc.vals
        )

        v_pred_error = split_real_and_imaginary_metric(aggregated_metrics["mse"])(
            v_pred, eval_pressure_pc.vals
        )

        logger.log_scalar("errors/p_mse_re", p_pred_error.real, step)
        logger.log_scalar("errors/p_mse_im", p_pred_error.imag, step)
        logger.log_scalar("errors/v_mse_re", v_pred_error.real, step)
        logger.log_scalar("errors/v_mse_im", v_pred_error.imag, step)

        # compute point-wise metrics
        p_error = split_real_and_imaginary_metric(sq_error)(
            p_pred, eval_pressure_pc.vals
        )
        logger.log_plot("errors/p_sq_error_re", plot_pred, jnp.real(p_error), step)
        logger.log_plot("errors/p_sq_error_im", plot_pred, jnp.imag(p_error), step)

        v_error = split_real_and_imaginary_metric(sq_error)(
            v_pred, eval_velocity_pc.vals
        )
        logger.log_plot("errors/v_sq_error_re", plot_pred, jnp.real(v_error), step)
        logger.log_plot("errors/v_sq_error_im", plot_pred, jnp.imag(v_error), step)

        # what the 'velcity model' looks like
        v_inv_pred = jax.vmap(velocity_model, [None, 0, 0, 0])(
            velocity_params, *eval_velocity_pc.coords
        )
        logger.log_plot("plots/v_inv_pred_mag", plot_pred, jnp.abs(v_inv_pred), step)
        logger.log_plot(
            "plots/v_inv_pred_phase", plot_pred, jnp.angle(v_inv_pred), step
        )
