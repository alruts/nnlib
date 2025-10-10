import pickle
from pathlib import Path

import equinox as eqx
import jax
import optax
from jax import numpy as jnp
from jax import random as jrandom
from matplotlib import pyplot as plt
from tqdm import tqdm

from nnlib import feature_maps
from nnlib.data_utils import (
    DataPointSampler,
    UniformSampler,
    grid_sample,
)
from nnlib.losses import (
    compute_weighted_loss,
    compute_weights,
    data_loss,
    pde_loss,
    update_weights,
)
from nnlib.pinn import WavePINN

# load data structure from .pkl file
data_path = Path("./data/gt_data.pkl")
with open(data_path, "rb") as f:
    data = pickle.load(f)

seed_key = jrandom.PRNGKey(0)
data_key, subsample_key, net_key, emb_key, dom_key = jrandom.split(seed_key, 5)

dataset = DataPointSampler(
    point_cloud=grid_sample(data, (50, 50)),  # returns `PointCloud`
    batch_size=1024,
    key=data_key,
)

domain_sampler = UniformSampler([(-1, 1), (-1, 1)], batch_size=256, key=dom_key)

infinite_dataloader = iter(dataset)
infinite_point_generator = iter(domain_sampler)

# Define architecture
rff_emb = feature_maps.RandomFourierFeatures(
    embed_scale=20.0, embed_dim=32, in_dim=2, key=emb_key
)
id_emb = eqx.nn.Identity()

setups = (
    ["mlp", rff_emb],
    ["modified_mlp", rff_emb],
    ["pirate_net", rff_emb],
    ["siren", id_emb],
    ["modified_siren", id_emb],
)

loss_dict = {"data": data_loss, "pde": pde_loss}
weight_dict = {"data": jnp.array(1.0), "pde": jnp.array(1.0)}

loss_fn = compute_weighted_loss
update_weights_every = 100

for arch, emb in setups:
    pinn = WavePINN.create(
        embedding=emb,
        arch_name=arch,
        in_size=2,
        out_size="scalar",
        width_size=64,
        depth=3,
        key=net_key,
    )

    params, static = eqx.partition(pinn.model, filter_spec=eqx.is_array)
    learning_rate = optax.schedules.exponential_decay(1e-3, 1000, 0.9)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @eqx.filter_jit
    def train_step(model, params, opt_state, weights, batch):
        (total, seperate), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params,
            model=model,
            weights=weights,
            batch=batch,
            losses=loss_dict,
        )

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, (total, seperate)

    total_steps = 8000
    for step, data_points, pde_points in tqdm(
        zip(range(total_steps), infinite_dataloader, infinite_point_generator),
        total=total_steps,
    ):
        batch = {"data": data_points, "pde": pde_points}

        if step % update_weights_every == 0:
            new_weights = compute_weights(params, pinn, batch, loss_dict)
            weight_dict = update_weights(0.9, weight_dict, new_weights)

        params, opt_state, loss = train_step(
            pinn, params, opt_state, weight_dict, batch
        )

    # ~~ Plot ~~
    X, Y = data.coordinate_arrays
    x, y = (X.ravel(), Y.ravel())

    # Evaluate on the grid of the original full data
    pressure = pinn.pressure_pred_fn(params, x, y)
    pressure = pressure.reshape(data.vals.shape)

    plt.figure(figsize=(6, 5))
    plt.imshow(pressure.T, origin="lower", extent=(-1, 1, -1, 1), cmap="viridis")
    plt.colorbar(label="Pressure")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f"{arch}_{emb}.png")
    print(f"saved: {arch}_{emb}.png")
    plt.show()
