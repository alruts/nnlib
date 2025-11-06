import equinox as eqx
import jax
import optax
from jax import numpy as jnp
from jax import random as jrandom
from matplotlib import pyplot as plt
from soap_jax import soap
from tqdm import tqdm

from nnlib.activations import (
    SplitSinActivation,
)
from nnlib.data_utils import (
    DataPointGenerator,
    UniformGenerator,
    subsample,
)
from nnlib.data_utils.data_structures import GridDiscretisationND
from nnlib.logger import TensorboardLogger
from nnlib.losses import (
    aggregated_metrics,
    compute_weighted_loss,
    compute_weights,
    data_loss,
    hom_pde_loss,
    update_weights,
)
from nnlib.metrics import point_wise_metrics
from nnlib.misc import (
    default_complex_dtype,
    default_wave_speed,
    grid_map,
    split_real_and_imaginary_loss,
    split_real_and_imaginary_metric,
)
from nnlib.pinn import HelmholtzPINN

seed_key = jrandom.PRNGKey(0)
data_key, subsample_key, net_key, emb_key, dom_key = jrandom.split(seed_key, 5)

#
# Construct infinite data generators for PDE collocation points
# and to load random batches of data points
#

FREQUENCY = 10


def point_source(x, x0=jnp.array([1.0, 1.0]), A=1 + 1j, f=FREQUENCY):
    R = jnp.sqrt(jnp.sum(jnp.abs(x - x0) ** 2))
    k = (2 * jnp.pi * f) / default_wave_speed()
    return A * (jnp.exp(-1j * k * R) / R)


data = GridDiscretisationND.discretise_fn(
    [(-0.5, 0.5), (-0.5, 0.5)], n_points=[256, 256], fn=point_source
)

datasets = [
    DataPointGenerator(
        point_cloud=subsample.random_sample(
            data,
            num_points=n,
            key=subsample_key,
        ),  # returns `PointCloud` with random samples
        batch_size=n,
        key=data_key,
    )
    for n in [8, 16, 32, 64, 128, 256, 512]
]

domain_sampler = UniformGenerator([(-1, 1), (-1, 1)], batch_size=128, key=dom_key)

# Here we define which loss functions to use during training
# The losses should be parallelized and vectorized via `pmap` and `vmap`
#

losses = {"data": data_loss, "pde": hom_pde_loss}
update_weights_every = jnp.inf

# Initialize the weights for adaptive grad norm, these are updated after the first step
# and every `update_weights_every` steps after that
loss_weights = {key: jnp.array(1.0) for key in losses.keys()}


# Helper to make predictions for logging
@eqx.filter_jit
def compute_pressure(params, pinn):
    """Evaluates pressure over the same grid as original dataset"""
    X, T = data.coordinate_arrays

    # Map p_net to accept mesh-grids for x and t
    p = grid_map(pinn.p_net, axis_mask=(0, 1, 1))
    return p(params, X, T)


#
# Train the each setup in a loop
#

for dataset in datasets:
    # These are infinitely iterable
    infinite_dataloader = iter(dataset)
    infinite_point_generator = iter(domain_sampler)

    # Helper to make plots for logging
    def plot_pred(pressure):
        fig = plt.figure(figsize=(6, 5))
        plt.pcolormesh(
            *data.coordinate_arrays,
            pressure,
            shading="auto",
            cmap="jet",
        )
        plt.colorbar(label="Pressure")
        plt.scatter(*dataset.point_cloud.coords.T, color="k")
        plt.xlabel("x")
        plt.ylabel("y")
        return fig

    # initialize logger
    logger = TensorboardLogger(experiment_name=f"test-run_{len(dataset)}")
    logger.log_plot("plots/gt_re", plot_pred, data.vals.real, 0)  # to visually compare
    logger.log_plot("plots/gt_im", plot_pred, data.vals.imag, 0)  # to visually compare
    log_every = 1000

    # build PINN
    pinn = HelmholtzPINN.create(
        embedding=None,
        arch_name="siren",
        frequency=FREQUENCY,
        in_size=2,
        out_size="scalar",
        width_size=32,
        depth=3,
        dtype=default_complex_dtype(),
        activation=SplitSinActivation(30.0),
        first_activation=SplitSinActivation(30.0),
        key=net_key,
    )

    # Extract the trainable parameters of the neural-net as a `PyTree`
    params, _ = eqx.partition(pinn.model, filter_spec=eqx.is_array)
    params = jax.tree.map(lambda x: x / jnp.sqrt(2), params)  # scaling due to complex

    # Initialize the Adam optimizer with learning rate scheduler
    learning_rate = optax.schedules.exponential_decay(1e-3, 2000, 0.9)
    # optimizer = optax.contrib.split_real_and_imaginary(optax.adam(learning_rate))
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

    total_steps = int(4e3) + 1
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

        # Write to logger
        if step % log_every == 0:
            logger.log_scalar("loss/total", loss, step)

            for term, weight in loss_weights.items():
                logger.log_scalar(f"weight/{term}", weight, step)

            for term, loss in individual_losses.items():
                logger.log_scalar(f"loss/{term}", loss / loss_weights.get(term), step)

            # make a prediction
            pred = compute_pressure(params, pinn)

            # aggregated metrics
            pred_error = split_real_and_imaginary_metric(aggregated_metrics["mse"])(
                pred, data.vals
            )
            logger.log_scalar("error/mse_re", pred_error.real, step)
            logger.log_scalar("error/mse_im", pred_error.imag, step)

            # compute point-wise metrics
            for metric, fn in point_wise_metrics.items():
                error = split_real_and_imaginary_metric(fn)(pred, data.vals)
                logger.log_plot(f"errors/{metric}_re", plot_pred, error.real, step)
                logger.log_plot(f"errors/{metric}_im", plot_pred, error.imag, step)

            logger.log_plot("plots/pred_re", plot_pred, pred.real, step)
            logger.log_plot("plots/pred_im", plot_pred, pred.imag, step)
