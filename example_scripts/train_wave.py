from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import optax
from soap_jax import soap
from tqdm import tqdm

import pinnlib as pl
from pinnlib import data_utils
from pinnlib.data_utils import pc_utils as pcu
from pinnlib.data_utils.point_cloud import GridDiscretisationND
from pinnlib.metrics import mse
from pinnlib.misc import args_to_array, default_wave_speed


# helper to animate
def animate(field):
    # Set up figure
    fig, ax = plt.subplots()
    im = ax.imshow(field.vals[:, :, 0], cmap="seismic", origin="lower")
    ax.set_title("Acoustic Wave")
    fig.colorbar(im, ax=ax)

    # Update function for animation
    def update(frame):
        im.set_data(field.vals[:, :, frame])
        ax.set_title(f"Time step: {frame}")
        return [im]

    # Create animation
    _ = animation.FuncAnimation(
        fig, update, frames=field.vals.shape[2], interval=1, blit=False
    )

    # Show animation
    plt.show()


# set random seed
seed = jr.PRNGKey(0)
keys = iter(jr.split(seed, 13))


def plane_wave(xs, A=1.0, theta=0.0, f=1000.0, c=pl.default_wave_speed()):
    x, y, t = xs
    k = 2 * jnp.pi * f / c
    return A * jnp.cos(
        k * (x * jnp.cos(theta) + y * jnp.sin(theta)) - 2 * jnp.pi * f * t
    )


# discretise the plane wave with random waves
n_waves = 3
angles = jr.uniform(next(keys), (n_waves,), minval=-jnp.pi, maxval=jnp.pi)
waves = [
    data_utils.GridDiscretisationND.discretise_fn(
        fn=partial(plane_wave, theta=θ),
        bounds=[(-0.25, 0.25), (-0.25, 0.25), (0.0, 0.1)],
        n_points=[128, 128, 96],
    )
    for θ in angles
]

# Sum and normalize all waves
gt_field: GridDiscretisationND = sum(waves[1:], start=waves[0])
gt_field *= 1 / float(jnp.max(gt_field.vals))

# Create point cloud from dataset
dataset = data_utils.PointCloud(
    tuple(x.flatten() for x in gt_field.coordinate_arrays), gt_field.vals.flatten()
)

# Make random sensor location filter
n_sensors = 32
x, y, _ = dataset.coords
sensor_locs = jr.choice(next(keys), jnp.stack([x, y], axis=-1), (32,), replace=False)

filter_x = pcu.filter_points(lambda c, _: jnp.isin(c[0], sensor_locs[:, 0]))
filter_y = pcu.filter_points(lambda c, _: jnp.isin(c[1], sensor_locs[:, 1]))
spatial_filter = pcu.pipe(filter_x, filter_y)

# Make infinite generators for training points
data_generator = data_utils.DataPointGenerator(
    point_cloud=spatial_filter(dataset),
    batch_size=1024,
    key=next(keys),
)

domain_generator = data_utils.UniformGenerator(
    gt_field.bounds,
    batch_size=256,
    key=next(keys),
)

infinite_data_loader = iter(data_generator)
infinite_domain_loader = iter(domain_generator)

# Losses are defined in a dictionary like so, arbitrary losses can be added
# but they must be vmapped and pmapped
losses = {"data": pl.data_loss, "pde": pl.hom_pde_loss}

# Initialize the weights for adaptive grad norm, these are updated after the first step
# and every `update_weights_every` steps after that
update_weights_every = jnp.inf
loss_weights = {key: jnp.array(1.0) for key in losses.keys()}


# build PINN
pinn = pl.WavePINN.create(
    embedding=None,
    arch_name="siren",
    in_size=3,
    out_size="scalar",
    first_activation=pl.SinActivation(10.0),
    activation=pl.SinActivation(30.0),
    final_activation=lambda x: x,
    width_size=32,
    depth=3,
    key=next(keys),
    wave_speed=default_wave_speed(),
)

# Extract the trainable parameters of the neural-net as a `PyTree`
params, _ = eqx.partition(pinn.model, filter_spec=eqx.is_array)

# Initialize optimizers
learning_rate = optax.schedules.exponential_decay(1e-3, 1000, 0.9)
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
total_steps = int(5e3) + 1
print_every = 1000
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

    if step % print_every == 0:
        print(individual_losses)

# Finalized model
final_model = lambda *xs: pinn(params, *xs)

# Make prediction
predicted_field = GridDiscretisationND.discretise_fn(
    fn=args_to_array(final_model),  # fn needs to take in arrays
    bounds=[(-0.25, 0.25), (-0.25, 0.25), (0.0, 0.1)],
    n_points=[128, 128, 96],
)

# Visualize!
animate(predicted_field)
animate(gt_field)


# todo: add particle velocity visualization
