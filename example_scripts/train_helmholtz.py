import warnings
from pathlib import Path

import equinox as eqx
import jax
import optax
from jax import numpy as jnp
from jax import random as jrandom
from matplotlib import pyplot as plt
from tqdm import tqdm

from nnlib import feature_maps
from nnlib.activations import cardioid, split_sin_activation
from nnlib.data_utils import (
    DataPointSampler,
    UniformSampler,
    subsample,
)
from nnlib.data_utils.data_structures import GridDiscretisationND
from nnlib.logger import TensorboardLogger
from nnlib.losses import (
    compute_weighted_loss,
    compute_weights,
    criteria,
    data_loss,
    make_loss,
    pde_loss,
    update_weights,
)
from nnlib.misc import default_wave_speed, grid_map
from nnlib.pinn import HelmholtzPINN

warnings.filterwarnings("error")
seed_key = jrandom.PRNGKey(0)
data_key, subsample_key, net_key, emb_key, dom_key = jrandom.split(seed_key, 5)

#
# Construct infinite data generators for PDE collocation points
# and to load random batches of data points
#

FREQUENCY = 440


def point_source(x, x0=jnp.array([0.0, 0.0]), f=FREQUENCY):
    r, r0 = jnp.linalg.norm(x), jnp.linalg.norm(x0)
    R = jnp.abs(r - r0)
    k = (2 * jnp.pi * f) / default_wave_speed()
    return jnp.exp(-1j * k * R) / R


data = GridDiscretisationND.discretise_fn(
    [(-0.1, 0.1), (-0.1, 0.1)], n_points=[256, 256], fn=point_source
)


dataset = DataPointSampler(
    point_cloud=subsample.random_sample(
        data,
        num_points=int(data.n_points * 0.01),  # take 1% of data
        key=subsample_key,
    ),  # returns `PointCloud` with random samples
    batch_size=32,
    key=data_key,
)

domain_sampler = UniformSampler([(-1, 1), (-1, 1)], batch_size=128, key=dom_key)

# These are infinitely iterable
infinite_dataloader = iter(dataset)
infinite_point_generator = iter(domain_sampler)

# Random Fourier features for input coordinates helps with low-frequency bias
rff_emb = feature_maps.RandomFourierFeatures(
    embed_scale=10.0, embed_dim=256, in_dim=2, key=emb_key
)
id_emb = eqx.nn.Identity()  # simply does nothing

# Pairs of embeddings and architectures for loop
setup = (
    ["mlp", rff_emb],
    ["modified_mlp", rff_emb],
    ["pirate_net", rff_emb],
    ["siren", id_emb],
    ["modified_siren", id_emb],
)

#
# Here we define which loss functions to use during training
# The losses should be parallelized and vectorized via `pmap` and `vmap`
#

losses = {"data": data_loss, "pde": pde_loss}
update_weights_every = 100

# Arbitrary loss terms can be added, single loss also works
# losses = {"data": data_loss}
# update_weights_every = jnp.inf # hack to avoid using adaptive weights

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


# Helper to make plots for logging
def plot_pred(pressure):
    fig = plt.figure(figsize=(6, 5))
    plt.pcolormesh(
        *data.coordinate_arrays,
        pressure.real,
        shading="auto",
        cmap="jet",
    )
    plt.colorbar(label="Pressure")
    plt.xlabel("x")
    plt.ylabel("t")
    return fig


#
# Train the each setup in a loop
#

for arch, emb in setup:
    # initialize logger
    logger = TensorboardLogger(experiment_name=f"test-run_{arch}")
    log_every = 1000

    # build PINN
    pinn = HelmholtzPINN.create(
        embedding=emb,
        arch_name=arch,
        frequency=FREQUENCY,
        in_size=2,
        out_size="scalar",
        width_size=32,
        depth=3,
        key=net_key,
        dtype=jnp.complex64,
        # activation=split_sin_activation,
    )

    # Extract the trainable parameters of the neural-net as a `PyTree`
    params, _ = eqx.partition(pinn.model, filter_spec=eqx.is_array)

    # Initialize the adam optimizer with learning rate scheduler
    learning_rate = optax.schedules.exponential_decay(1e-3, 2000, 0.9)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)  # Running state of the optimizer

    #
    # Define a single training step, `filter_jit` compiles this code to
    # machine-code
    #

    @eqx.filter_jit
    def train_step(model, params, opt_state, weights, batch):
        (total, each_term), grads = jax.value_and_grad(
            compute_weighted_loss, has_aux=True, holomorphic=True
        )(
            params,
            model=model,
            weights=weights,
            batch=batch,
            losses=losses,
            criterion=criteria["split_mse"],
        )

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, (total, each_term)

    #
    # Training loop: all the optimization work is done here + logging
    #

    total_steps = int(40e3)
    for step, data_batch, pde_batch in tqdm(
        zip(range(total_steps), infinite_dataloader, infinite_point_generator),
        total=total_steps,
    ):
        # The batch dictionary should have the same structure as `losses`
        batch = {"data": data_batch, "pde": pde_batch}

        # Convert all float arrays to complex64
        batch = jax.tree.map(
            lambda x: x.astype(jnp.complex64)
            if jnp.issubdtype(x.dtype, jnp.floating)
            else x,
            batch,
        )

        # Do step
        params, opt_state, (loss, individual_losses) = train_step(
            pinn, params, opt_state, loss_weights, batch
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
            pred_error = jnp.mean(jnp.square(data.vals - pred))

            logger.log_scalar("error/mse", pred_error, step)
            logger.log_plot("plots/pred", plot_pred, pred, step)
