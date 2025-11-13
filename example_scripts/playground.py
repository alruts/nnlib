import equinox as eqx
import jax
import optax
import trimesh
from jax import numpy as jnp
from jax import random as jrandom
from matplotlib import pyplot as plt
from soap_jax import soap
from tqdm import tqdm

from pinnlib.data_utils.generators import MeshGenerator

jax.config.update("jax_enable_x64", True)


from pinnlib.activations import (
    SplitSinActivation,
)
from pinnlib.data_utils import (
    DataPointGenerator,
    UniformGenerator,
    subsample,
)
from pinnlib.data_utils.data_structures import GridDiscretisationND
from pinnlib.logger import TensorboardLogger
from pinnlib.losses import (
    aggregated_metrics,
    compute_loss,
    data_loss,
    hom_pde_loss,
)
from pinnlib.metrics import point_wise_metrics
from pinnlib.misc import (
    default_complex_dtype,
    default_medium_density,
    default_wave_speed,
    grid_map,
    split_real_and_imaginary_loss,
    split_real_and_imaginary_metric,
)
from pinnlib.pinn import HelmholtzPINN
from pinnlib.simulate_data.rayleigh_disk import RayleighDiskInBaffle

# setup random keys
seed_key = jrandom.PRNGKey(0)
data_key, data_subsample_key, seed_key = jrandom.split(seed_key, 3)
dom_key, bnd_key, seed_key = jrandom.split(seed_key, 3)
field_key, source_key = jrandom.split(seed_key, 2)

# Construct infinite data generators for PDE collocation points
# and to load random batches of data points
#

disk_integrator = RayleighDiskInBaffle(
    medium_density=default_medium_density(),
    wave_speed=default_wave_speed(),
    frequency=400.0,
    disk_radius=0.1,
    surface_impedance=0.3,
    piston_velocity=1.0,
    points_per_wavelength=7,
)

# Derived acoustic quantities
angular_frequency = 2 * jnp.pi * disk_integrator.frequency
wavenumber = angular_frequency / disk_integrator.wave_speed
wavelength = 2 * jnp.pi / wavenumber

# Observation grid
grid_extent = 1.5 * wavelength
points_per_wavelength_obs = 8
dx_obs = wavelength / points_per_wavelength_obs

n_points_x = int(2 * grid_extent / dx_obs)
n_points_y = int(2 * grid_extent / dx_obs)
n_points_z = int((2 * wavelength) / dx_obs)

data = GridDiscretisationND.discretise_fn(
    bounds=[
        (-grid_extent, grid_extent),
        (-grid_extent, grid_extent),
        (0.01, wavelength),
    ],
    fn=disk_integrator,
    n_points=[n_points_x, n_points_y, n_points_z],
)

dataset = DataPointGenerator(
    point_cloud=subsample.full_data(
        data,
    ),  # returns `PointCloud` with random samples
    batch_size=128,
    key=data_key,
)

domain_sampler = UniformGenerator(
    [
        (-grid_extent, grid_extent),
        (-grid_extent, grid_extent),
        (0.0, wavelength),
    ],
    batch_size=256,
    key=dom_key,
)

# load in a mesh
mesh: trimesh.Trimesh = trimesh.load_mesh("~/Documents/disk.stl")
disk_sampler = MeshGenerator(mesh, batch_size=128, key=bnd_key)

# Here we define which loss functions to use during training
# The losses should be parallelized and vectorized via `pmap` and `vmap`
#

losses = {"data": data_loss, "pde": hom_pde_loss}

# Initialize the weights for adaptive grad norm, these are updated after the first step
# and every `update_weights_every` steps after that
loss_weights = {key: jnp.array(1.0) for key in losses.keys()}


# Helper to make predictions for logging
@eqx.filter_jit
def compute_pressure(params, pinn):
    """Evaluates pressure over the same grid as original dataset"""

    # Map p_net to accept mesh-grids for x and t
    p = grid_map(pinn.p_net, axis_mask=(0, 1, 1, 1))
    return p(params, *data.coordinate_arrays)


#
# Train the each setup in a loop
#

# These are infinitely iterable
infinite_dataloader = iter(dataset)
infinite_point_generator = iter(domain_sampler)
infinite_disk_generator = iter(disk_sampler)


# Helper to make plots for logging
def plot_pred(pressure):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(*data.coordinate_arrays, c=pressure, alpha=0.1, cmap="seismic")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.colorbar(sc, ax=ax)
    ax.set_aspect("equal", "box")
    return fig


# Initialize logger
logger = TensorboardLogger(experiment_name=f"test-run_{len(dataset)}")
logger.log_plot("plots/gt_abs", plot_pred, jnp.abs(data.vals), 0)  # to visually compare
logger.log_plot("plots/gt_angle", plot_pred, jnp.angle(data.vals), 0)
log_every = 1000

# Build PINN
pinn = HelmholtzPINN.create(
    embedding=None,
    arch_name="siren",
    frequency=disk_integrator.frequency,
    in_size=3,
    out_size="scalar",
    width_size=32,
    depth=3,
    dtype=default_complex_dtype(),
    activation=SplitSinActivation(30.0),
    first_activation=SplitSinActivation(30.0),
    key=field_key,
)

# Build source model
aux = HelmholtzPINN.create(
    embedding=None,
    arch_name="siren",
    frequency=disk_integrator.frequency,
    in_size=3,
    out_size="scalar",
    width_size=8,
    depth=3,
    dtype=default_complex_dtype(),
    activation=SplitSinActivation(30.0),
    first_activation=SplitSinActivation(30.0),
    key=field_key,
)


# Extract the trainable parameters of the neural-net as a `PyTree`
params, _ = eqx.partition(pinn.model, filter_spec=eqx.is_array)
params = jax.tree.map(lambda x: x / jnp.sqrt(2), params)  # scaling due to complex

# ...
aux_params, _ = eqx.partition(aux.model, filter_spec=eqx.is_array)
aux_params = jax.tree.map(lambda x: x / jnp.sqrt(2), aux_params)


# Initialize the Adam optimizer with learning rate scheduler
learning_rate = optax.schedules.exponential_decay(1e-3, 2000, 0.9)
opt_pinn = optax.contrib.split_real_and_imaginary(
    soap(learning_rate, precondition_frequency=2)
)
opt_state = opt_pinn.init(params)  # Running state of the optimizer

# ...
aux_opt = optax.contrib.split_real_and_imaginary(optax.adam(1e-3))
aux_opt_state = opt_pinn.init(aux_params)  # Running state of the optimizer

#
# Define a single training step, `filter_jit` compiles this code to
# machine-code
#


@eqx.filter_jit
def train_step(model, params, aux_model, aux_params, opt_state, weights, batch):
    # this only needs to include extra args for terms that need it

    (total, each_term), grads = jax.value_and_grad(compute_loss, has_aux=True)(
        params,
        model=model,
        batch=batch,
        losses=losses,
        criterion=split_real_and_imaginary_loss(aggregated_metrics["mse"]),
    )

    total_grads_conj = jax.tree.map(jnp.conj, grads)
    updates, opt_state = opt_pinn.update(total_grads_conj, opt_state, params)
    params = optax.apply_updates(params, updates)  # # pyright: ignore
    return params, opt_state, (total, each_term)


#
# Training loop: all the optimization work is done here + logging
#

total_steps = int(4e3) + 1
for step, data_batch, pde_batch, disk_batch in tqdm(
    zip(
        range(total_steps),
        infinite_dataloader,
        infinite_point_generator,
        infinite_disk_generator,
    ),
    total=total_steps,
):
    # The batch dictionary should have the same structure as `losses`
    batch = {"data": data_batch, "pde": pde_batch, "aux": disk_batch}

    # Do step
    params, opt_state, (loss, individual_losses) = train_step(
        pinn, params, aux, aux_params, opt_state, loss_weights, batch
    )

    # Write to logger
    if step % log_every == 0:
        logger.log_scalar("loss/total", loss, step)

        for term, weight in loss_weights.items():
            logger.log_scalar(f"weight/{term}", weight, step)

        for term, loss in individual_losses.items():
            logger.log_scalar(f"loss/{term}", loss / loss_weights.get(term), step)

        # make a prediction
        pred = compute_pressure(params, pinn)

        pred_error = split_real_and_imaginary_metric(aggregated_metrics["mse"])(
            pred, data.vals
        )
        logger.log_scalar("errors/mse_re", pred_error.real, step)
        logger.log_scalar("errors/mse_im", pred_error.imag, step)
        logger.log_plot("plots/pred_mag", plot_pred, jnp.abs(pred), step)
        logger.log_plot("plots/pred_phase", plot_pred, jnp.angle(pred), step)

        # compute point-wise metrics
        error = split_real_and_imaginary_metric(point_wise_metrics["sq_error"])(
            pred, data.vals
        )
        logger.log_plot("errors/sq_error_mag", plot_pred, jnp.abs(error), step)
        logger.log_plot("errors/sq_error_phase", plot_pred, jnp.angle(error), step)
