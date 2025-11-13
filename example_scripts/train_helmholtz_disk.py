import equinox as eqx
import jax
import optax
from jax import numpy as jnp
from jax import random as jrandom
from matplotlib import pyplot as plt
from soap_jax import soap
from tqdm import tqdm

from pinnlib import feature_maps
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
    compute_weighted_loss,
    compute_weights,
    data_loss,
    hom_pde_loss,
    update_weights,
)
from pinnlib.metrics import sq_error
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

seed_key = jrandom.PRNGKey(0)
data_key, subsample_key, net_key, emb_key, dom_key = jrandom.split(seed_key, 5)

#
# Construct infinite data generators for PDE collocation points
# and to load random batches of data points
#


disk_in_baffle = RayleighDiskInBaffle(
    medium_density=default_medium_density(),
    wave_speed=default_wave_speed(),
    frequency=1000.0,
    disk_radius=0.1,
    surface_impedance=2,
    piston_velocity=1.0,
    points_per_wavelength=10,
)

# Derived acoustic quantities
angular_frequency = 2 * jnp.pi * disk_in_baffle.frequency
wavenumber = angular_frequency / disk_in_baffle.wave_speed
wavelength = 2 * jnp.pi / wavenumber

# Observation grid
grid_extent = 0.75 * wavelength + disk_in_baffle.disk_radius
points_per_wavelength_obs = 128 // 2
dx_obs = wavelength / points_per_wavelength_obs

n_points_x = int(2 * grid_extent / dx_obs)
n_points_y = int(2 * grid_extent / dx_obs)
n_points_z = int((2 * wavelength) / dx_obs)

data = GridDiscretisationND.discretise_fn(
    bounds=[
        (-grid_extent, grid_extent),
        (-grid_extent, grid_extent),
        (0.0001, wavelength),
    ],
    fn=disk_in_baffle,
    n_points=[n_points_x, n_points_y, n_points_z],
)

eval_data = data.slice(z=0)


datasets = [
    DataPointGenerator(
        point_cloud=subsample.random_sample(
            data,
            num_points=n,
            key=subsample_key,
        ),  # returns `PointCloud` with random samples
        batch_size=128,
        key=data_key,
    )
    for n in {128}
]

domain_sampler = UniformGenerator(
    [
        (-grid_extent, grid_extent),
        (-grid_extent, grid_extent),
        (0.0, wavelength),
    ],
    batch_size=256,
    key=dom_key,
)

# Here we define which loss functions to use during training
# The losses should be parallelized and vectorized via `pmap` and `vmap`
#

losses = {"data": data_loss, "pde": hom_pde_loss}
update_weights_every = 1000

# Initialize the weights for adaptive grad norm, these are updated after the first step
# and every `update_weights_every` steps after that
loss_weights = {key: jnp.array(1.0) for key in losses.keys()}


# Helper to make predictions for logging
@eqx.filter_jit
def compute_pressure(params, pinn):
    """Evaluates pressure over the same grid as original dataset"""
    p = grid_map(pinn.p_net, axis_mask=(0, 1, 1, 0))
    return p(params, *eval_data.coordinate_arrays, 0.0)


# Helper to make plots for logging
def plot_pred(pressure):
    from matplotlib import colors

    divnorm = colors.TwoSlopeNorm(vmin=-3.14, vcenter=0.0, vmax=3.14)

    fig, ax = plt.subplots()
    pc = ax.pcolormesh(
        *eval_data.coordinate_arrays, pressure, cmap="seismic", norm=divnorm
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    plt.colorbar(pc, ax=ax)

    ax.set_aspect("equal", "box")
    return fig


#
# Train the each setup in a loop
#

for dataset in datasets:
    # These are infinitely iterable
    infinite_dataloader = iter(dataset)
    infinite_point_generator = iter(domain_sampler)

    # initialize logger
    logger = TensorboardLogger(experiment_name=f"test-run_{len(dataset)}")
    logger.log_plot(
        "plots/gt_mag", plot_pred, jnp.abs(eval_data.vals), 0
    )  # to visually compare
    logger.log_plot("plots/gt_phase", plot_pred, jnp.angle(eval_data.vals), 0)
    log_every = 1000

    # Random Fourier features for input coordinates helps with low-frequency bias
    rff = feature_maps.RandomFourierFeatures(
        embed_scale=1 / jnp.sqrt(wavelength), embed_dim=128, in_dim=3, key=emb_key
    )

    # build PINN
    pinn = HelmholtzPINN.create(
        embedding=rff,
        arch_name="siren",
        frequency=disk_in_baffle.frequency,
        in_size=3,
        out_size="scalar",
        width_size=16,
        depth=3,
        dtype=default_complex_dtype(),
        first_activation=SplitSinActivation(float(wavenumber)),
        activation=SplitSinActivation(30.0),
        # final_activation=LearnableSplitTanh(jnp.array(1.0), jnp.array(1.0)),
        key=net_key,
    )

    # Extract the trainable parameters of the neural-net as a `PyTree`
    params, _ = eqx.partition(pinn.model, filter_spec=eqx.is_array)
    params = jax.tree.map(lambda x: x / jnp.sqrt(2), params)  # scaling due to complex

    # Initialize the Adam optimizer with learning rate scheduler
    learning_rate = optax.schedules.exponential_decay(1e-3, 2000, 0.9)
    optimizer = optax.contrib.split_real_and_imaginary(
        soap(learning_rate, precondition_frequency=2)
    )
    opt_state = optimizer.init(params)  # Running state of the optimizer

    #
    # Define a single training step, `filter_jit` compiles this code to
    # machine-code
    #

    @eqx.filter_jit
    def train_step(model, params, opt_state, weights, batch):
        (total, each_term), grads = jax.value_and_grad(
            compute_weighted_loss, has_aux=True
        )(
            params,
            model=model,
            batch=batch,
            weights=weights,
            losses=losses,
            criterion=split_real_and_imaginary_loss(aggregated_metrics["mse"]),
        )
        grads_conj = jax.tree.map(jnp.conj, grads)
        updates, opt_state = optimizer.update(grads_conj, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, (total, each_term)

    #
    # Training loop: all the optimization work is done here + logging
    #

    total_steps = int(5e3) + 1
    for step, data_batch, pde_batch in tqdm(
        zip(range(total_steps), infinite_dataloader, infinite_point_generator),
        total=total_steps,
    ):
        # The batch dictionary should have the same structure as `losses`
        batch = {"data": data_batch, "pde": pde_batch}

        # Do step
        params, opt_state, (loss, individual_losses) = train_step(
            pinn, params, opt_state, loss_weights, batch
        )

        # Update adaptive weights
        if step % update_weights_every == 0:
            new_weights = compute_weights(params, pinn, batch, losses)
            loss_weights = update_weights(0.9, loss_weights, new_weights)

        if step % log_every == 0:
            logger.log_scalar("loss/total", loss, step)

            for term, weight in loss_weights.items():
                logger.log_scalar(f"weight/{term}", weight, step)

            for term, loss in individual_losses.items():
                logger.log_scalar(f"loss/{term}", loss / loss_weights.get(term), step)

            # make a prediction
            pred = compute_pressure(params, pinn)

            pred_error = split_real_and_imaginary_metric(aggregated_metrics["mse"])(
                pred, eval_data.vals
            )

            logger.log_scalar("errors/mse_re", pred_error.real, step)
            logger.log_scalar("errors/mse_im", pred_error.imag, step)
            logger.log_plot("plots/pred_mag", plot_pred, jnp.abs(pred), step)
            logger.log_plot("plots/pred_phase", plot_pred, jnp.angle(pred), step)

            # compute point-wise metrics
            error = split_real_and_imaginary_metric(sq_error)(pred, eval_data.vals)
            logger.log_plot("errors/sq_error_re", plot_pred, jnp.real(error), step)
            logger.log_plot("errors/sq_error_im", plot_pred, jnp.imag(error), step)
