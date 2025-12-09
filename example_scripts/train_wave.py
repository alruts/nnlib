from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from soap_jax import soap
from tqdm import tqdm

import pinnlib as pl
from pinnlib.data import (
    DataPointGenerator,
    GridDiscretisationND,
    PointCloud,
    UniformGenerator,
)
from pinnlib.data import pc_utils as pcu
from pinnlib.metrics import mse
from pinnlib.misc import args_to_array

# Set random seed
seed = jr.PRNGKey(42)
rng_keys = iter(jr.split(seed, 7))


def animate_comparison(ground_truth, prediction, sensor_locs):
    """
    Animate comparison between ground truth, prediction, and error with a
    common colorbar.
    """
    x, y, t = ground_truth.coordinate_arrays
    x, y, t = map(jnp.unique, (x, y, t))

    # Compute error
    error_vals = jnp.abs(ground_truth.vals - prediction.vals)

    # Set up figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Ground Truth", "Prediction", "Error"]
    datasets = [ground_truth.vals, prediction.vals, error_vals]

    # Determine common color limits across all datasets
    all_data = jnp.concatenate([d.reshape(-1) for d in datasets])
    vmin, vmax = all_data.min(), all_data.max()

    im_list = []

    for ax, data, title in zip(axes, datasets, titles):
        im = ax.imshow(
            data[:, :, 0],
            cmap="seismic",
            origin="lower",
            extent=(x.min(), x.max(), y.min(), y.max()),
            vmin=vmin,
            vmax=vmax,  # Use common color limits
        )
        ax.scatter(*sensor_locs.T, c="k")
        ax.set_title(title)
        im_list.append(im)

    # Add a single colorbar for all subplots
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = fig.colorbar(im_list[0], cax=cax)
    cbar.set_label("Pressure (Pa)")

    # Update function for animation
    def update(frame):
        for im, data in zip(im_list, datasets):
            im.set_data(data[:, :, frame])
        for ax, title in zip(axes, titles):
            ax.set_title(f"{title} - Time: {t[frame] * 1e3:.2f} ms")
        return im_list

    # Create animation
    _ = animation.FuncAnimation(
        fig, update, frames=ground_truth.vals.shape[2], interval=100, blit=False
    )
    fig.tight_layout()

    plt.show()


def plane_wave(xs, A=1.0, theta=0.0, f=1000.0, c=pl.default_wave_speed()):
    x, y, t = xs
    k = 2 * jnp.pi * f / c
    return A * jnp.cos(
        k * (x * jnp.cos(theta) + y * jnp.sin(theta)) - 2 * jnp.pi * f * t
    )


# Discretise the plane wave with random waves
n_waves = 12
random_angles = jr.uniform(next(rng_keys), (n_waves,), minval=-jnp.pi, maxval=jnp.pi)
random_freqs = jr.uniform(next(rng_keys), (n_waves,), minval=500, maxval=4000)

# 4 cycles
cycle_len = 4 / random_freqs.min()
sample_rate = 4 * random_freqs.max()

waves = [
    GridDiscretisationND.discretise_fn(
        fn=partial(plane_wave, theta=θ, f=f),
        bounds=[(-0.25, 0.25), (-0.25, 0.25), (0.0, float(cycle_len))],
        n_points=[128, 128, int(cycle_len * sample_rate)],
    )
    for θ, f in zip(random_angles, random_freqs)
]

# Sum and normalize all waves
gt_field: GridDiscretisationND = sum(waves[1:], start=waves[0])
gt_field *= 1 / float(jnp.max(gt_field.vals))

# Create point cloud from dataset
dataset = PointCloud(
    tuple(x.flatten() for x in gt_field.coordinate_arrays), gt_field.vals.flatten()
)

# Make random sensor location filter (keep dense time axis for each (x,y) pair)
n_sensors = 16
x, y, _ = dataset.coords
sensor_locs = jr.choice(
    next(rng_keys), jnp.stack([x, y], axis=-1), (n_sensors,), replace=False
)

filter_x = pcu.filter_points(lambda c, _: jnp.isin(c[0], sensor_locs[:, 0]))
filter_y = pcu.filter_points(lambda c, _: jnp.isin(c[1], sensor_locs[:, 1]))
spatial_filter = pcu.pipe(filter_x, filter_y)

filtered_pc = spatial_filter(dataset)

# Make data generators
data_generator = DataPointGenerator(
    point_cloud=filtered_pc,
    batch_size=len(filtered_pc.vals) // 8,
    key=next(rng_keys),
)

domain_generator = UniformGenerator(
    gt_field.bounds,
    batch_size=2048,
    key=next(rng_keys),
)

infinite_data_loader = iter(data_generator)
infinite_domain_loader = iter(domain_generator)

# Losses are defined in a dictionary like so, arbitrary losses can be added
# but they must be vmapped and pmapped
losses = {"data": pl.data_loss, "pde": pl.hom_pde_loss}

# Initialize the weights for adaptive grad norm, these are updated after the first step
# and every `update_weights_every` steps after that
update_weights_every = 1000
loss_weights = {key: jnp.array(1.0) for key in losses.keys()}

# Build PINN
pinn = pl.WavePINN.create(
    arch_name="modified_siren",
    in_size=3,
    out_size="scalar",
    width_size=64,
    depth=3,
    first_activation=pl.SinActivation(10.0),
    activation=pl.SinActivation(30.0),
    final_activation=pl.LearnableTanh(jnp.array(1.0)),
    key=next(rng_keys),
)

# Extract the trainable parameters of the neural-net as a `PyTree`
params, _ = eqx.partition(pinn.model, filter_spec=eqx.is_array)

# Initialize optimizers
learning_rate = optax.schedules.exponential_decay(1e-3, 2000, 0.9)
# optimizer = optax.adam(learning_rate)
optimizer = soap(learning_rate, precondition_frequency=2)
opt_state = optimizer.init(params)


@eqx.filter_jit
def train_step(model, params, opt_state, weights, batch):
    # extra arguments for boundary loss
    (total, each_term), grads = jax.value_and_grad(
        pl.compute_weighted_loss, has_aux=True
    )(
        params,
        model=model,
        batch=batch,
        weights=weights,
        losses=losses,
        criterion=mse,
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, (total, each_term)


# Training loop: all the optimization work is done here + logging
total_steps = int(8e3) + 1
log_every = 1000
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
        new_weights = pl.compute_weights(params, pinn, batch, losses, criterion=mse)
        loss_weights = pl.update_weights(0.9, loss_weights, new_weights)

    if step % log_every == 0:
        print(individual_losses)


# Animate results
make_pred = lambda *xs: pinn(params, *xs)
predicted_field = GridDiscretisationND.discretise_fn(
    fn=args_to_array(make_pred),
    bounds=gt_field.bounds,
    n_points=gt_field.vals.shape,
)
animate_comparison(gt_field, predicted_field, sensor_locs)
