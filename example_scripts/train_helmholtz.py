from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from soap_jax import soap
from tqdm import tqdm

import pinnlib as pl
from pinnlib.data import (
    DataPointGenerator,
    GridDiscretisationND,
    UniformGenerator,
)
from pinnlib.data import pc_utils as pcu
from pinnlib.metrics import mse

# Set random seed
seed = jr.PRNGKey(42)
rng_keys = iter(jr.split(seed, 7))

# Problem parameters
frequency = 1000.0  # Hz
wave_speed = pl.default_wave_speed()
angular_frequency = 2 * jnp.pi * frequency
wavenumber = angular_frequency / wave_speed
wavelength = 2 * jnp.pi / wavenumber

# Domain setup - square domain of side length 0.25 m
domain_size = 0.25
domain_bounds = [
    (-domain_size / 2, domain_size / 2),
    (-domain_size / 2, domain_size / 2),
]

# Three point sources on the boundary
source_locations = [
    (-0.1 + -domain_size / 2, 0.0),
    (-0.1 + -domain_size / 2, 0.05),
    (-0.1 + -domain_size / 2, -0.05),
]

# Source amplitudes (complex)
source_amplitudes = [
    1.0 + 0.0j,
    0.8 - 0.2j,
    0.6 + 0.1j,
]


def point_source(coords, origin, amplitude, k):
    """Analytical solution for point sources in 2D Helmholtz equation"""
    x, y = coords
    r = jnp.sqrt((x - origin[0]) ** 2 + (y - origin[1]) ** 2)
    return amplitude * jnp.exp(1j * k * r) / jnp.sqrt(r)


# Create ground truth field
gt_fields = [
    GridDiscretisationND.discretise_fn(
        fn=partial(point_source, origin=loc, amplitude=A, k=wavenumber),
        bounds=domain_bounds,
        n_points=[128, 128],
    )
    for A, loc in zip(source_amplitudes, source_locations)
]
gt_field = sum(gt_fields[1:], start=gt_fields[0])  # sum points source fields

# Create point cloud from dataset
dataset = gt_field.as_point_cloud()

subset = pcu.grid_sample_points(grid_size=(5, 5))(dataset)
# Make data generators
data_generator = DataPointGenerator(
    point_cloud=subset,
    batch_size=len(subset.vals),  # full-batch
    key=next(rng_keys),
)

domain_generator = UniformGenerator(
    domain_bounds,
    batch_size=128,
    key=next(rng_keys),
)

infinite_data_loader = iter(data_generator)
infinite_domain_loader = iter(domain_generator)

# Losses are defined in a dictionary like so, arbitrary losses can be added
losses = {"data": pl.data_loss, "pde": pl.hom_pde_loss}

# Initialize the weights for adaptive grad norm, these are updated after the first step
# and every `update_weights_every` steps after that
update_weights_every = 1000
loss_weights = {key: jnp.array(1.0) for key in losses.keys()}

# Build PINN
pinn = pl.pinn.HelmholtzPINN.create(
    embedding=None,
    arch_name="modified_siren",
    frequency=frequency,
    in_size=2,
    out_size="scalar",
    width_size=32,
    depth=3,
    dtype=pl.default_complex_dtype(),
    first_activation=pl.SplitSinActivation(30),
    activation=pl.SplitSinActivation(30),
    final_activation=pl.identity_activation,
    key=next(rng_keys),
)

# Extract the trainable parameters of the neural-net as a `PyTree`
params, _ = eqx.partition(pinn.model, filter_spec=eqx.is_array)

# Initialize optimizers
learning_rate = optax.schedules.exponential_decay(1e-3, 2000, 0.9)
optimizer = optax.contrib.split_real_and_imaginary(soap(learning_rate))
opt_state = optimizer.init(params)


@eqx.filter_jit
def train_step(model, params, opt_state, weights, batch):
    (total, each_term), grads = jax.value_and_grad(
        pl.compute_weighted_loss, has_aux=True
    )(
        params,
        model=model,
        batch=batch,
        weights=weights,
        losses=losses,
        criterion=pl.split_real_and_imaginary_loss(mse),
    )
    grads_conj = jax.tree.map(jnp.conj, grads)
    updates, opt_state = optimizer.update(grads_conj, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, (total, each_term)


# Helpers for logging and evaluation
@eqx.filter_jit
def predict_pressure(params, pinn, x, y):
    """Evaluates pressure over given coordinates"""
    p_fn = jax.vmap(pinn, [None, 0, 0])
    return p_fn(params, x, y)


def plot_solution(pressure, title=""):
    """Plot solution on 2D domain"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Get coordinates from ground truth field
    x, y = gt_field.coordinate_arrays

    # Plot magnitude
    im = ax.pcolormesh(x, y, jnp.abs(pressure), shading="auto", cmap="viridis")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_aspect("equal")

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Magnitude")

    # Mark sensors
    plt.scatter(*subset.coords, c="k", label="Sensor")

    # Mark source locations
    for i, source_loc in enumerate(source_locations):
        ax.plot(
            source_loc[0],
            source_loc[1],
            "r*",
            markersize=15,
            label="Source" if i == 0 else None,
        )

    if len(source_locations) > 0:
        ax.legend()

    return fig


# Training loop: all the optimization work is done here + logging
total_steps = int(8e3) + 1
for step, data_batch, pde_batch in tqdm(
    zip(range(total_steps), infinite_data_loader, infinite_domain_loader),
    total=total_steps,
):
    # The batch dictionary should have the same structure as `losses`
    batch = {"data": data_batch, "pde": pde_batch}

    # Do step
    params, opt_state, (loss, individual_losses) = train_step(
        pinn,
        params,
        opt_state,
        loss_weights,
        batch,
    )

    # Update adaptive weights
    if step % update_weights_every == 0:
        new_weights = pl.compute_weights(
            params, pinn, batch, losses, criterion=pl.split_real_and_imaginary_loss(mse)
        )
        loss_weights = pl.update_weights(0.9, loss_weights, new_weights)

    if step % 1000 == 0:
        print(f"Step {step}: Loss = {loss:.6f}")
        print(f"Individual losses: {individual_losses}")

# plot final solution
print("Generating final solution plot...")

# Create evaluation grid using ground truth field coordinates
x_eval, y_eval = gt_field.coordinate_arrays
x_flat = x_eval.flatten()
y_flat = y_eval.flatten()

# Predict pressure field using helper function
predicted_pressure = predict_pressure(params, pinn, x_flat, y_flat)
predicted_pressure = predicted_pressure.reshape(x_eval.shape)

# Plot ground truth using helper function
fig_gt = plot_solution(gt_field.vals, title="Ground Truth Solution")
plt.savefig("ground_truth.png", bbox_inches="tight")
plt.show()

# Plot predicted solution using helper function
fig_pred = plot_solution(predicted_pressure, title="PINN Predicted Solution")
plt.savefig("predicted_solution.png", bbox_inches="tight")
plt.show()

# Plot error
error = jnp.abs(predicted_pressure - gt_field.vals)
fig_error, ax_error = plt.subplots(figsize=(8, 6))
im_error = ax_error.pcolormesh(x_eval, y_eval, error, shading="auto", cmap="hot")
ax_error.set_xlabel("X (m)")
ax_error.set_ylabel("Y (m)")
ax_error.set_title("Absolute Error")
ax_error.set_aspect("equal")
plt.colorbar(im_error, ax=ax_error, label="Error")
plt.savefig("error.png", bbox_inches="tight")
plt.show()

print("Plots saved: ground_truth.png, predicted_solution.png, error.png")
