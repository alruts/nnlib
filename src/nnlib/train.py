import pickle
from pathlib import Path

import equinox as eqx
import jax
import optax
from jax import numpy as jnp
from jax import random as jrandom
from matplotlib import pyplot as plt
from numpy import identity
from tqdm import tqdm

from nnlib import embeddings
from nnlib.dataload import subset
from nnlib.dataload.sampling import DataPointSampler
from nnlib.misc import get_parameters
from nnlib.pinn import WavePINN

# load data structure from .pkl file
data_path = Path("./data/gt_data.pkl")
with open(data_path, "rb") as f:
    data = pickle.load(f)

_key = jrandom.PRNGKey(0)
data_key, subsample_key, net_key, emb_key = jrandom.split(_key, 4)

data = subset.full_data(data=data)

dataset = DataPointSampler(
    point_cloud=data,
    batch_size=1024,
    key=data_key,
)

dataloader = iter(dataset)

# define architecture
emb = embeddings.RandomFourierEmbedding(
    embed_scale=1.0, embed_dim=16, in_dim=2, key=emb_key
)
# emb = eqx.nn.Identity()
pinn = WavePINN.create(
    embedding=emb,
    arch_name="modified_siren",
    in_size=2,
    out_size="scalar",
    width_size=64,
    depth=5,
    key=net_key,
)


params, static = eqx.partition(pinn.model, filter_spec=eqx.is_array)


criteria = {
    "mse": lambda x, y, axis=None: jnp.mean((x - y) ** 2, axis),
    "mae": lambda x, y, axis=None: jnp.mean(jnp.abs(x - y), axis),
}


learning_rate = 1e-4
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)


@eqx.filter_jit
def train_step(model, params, opt_state, batch):
    def data_loss(model, params, batch, criterion=criteria["mse"]):
        *coords, vals = batch
        n_dim = len(coords)

        # vectorize and parallelize
        batched_p_net = jax.vmap(model.p_net, in_axes=(None, *[0] * n_dim))
        parallel_p_net = jax.pmap(batched_p_net, in_axes=(None, *[0] * n_dim))

        # make prediction
        pred = parallel_p_net(params, *coords)
        return criterion(pred, vals)  # error measure

    loss, grads = jax.value_and_grad(data_loss, argnums=1)(model, params, batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


train_steps = 20_000
for step, batch in tqdm(zip(range(train_steps), dataloader), total=train_steps):
    params, opt_state, loss = train_step(pinn, params, opt_state, batch)


# Define grid
nx, ny = 300, 300  # resolution
x = jnp.linspace(-1, 1, nx)
y = jnp.linspace(-1, 1, ny)
X, Y = jnp.meshgrid(x, y, indexing="ij")  # shape: (nx, ny)
coords = (X.ravel(), Y.ravel())  # flatten to (nx*ny,)

# Evaluate on the grid
pressure = pinn.pressure_pred_fn(params, *coords)  # shape: (nx*ny,)
pressure = pressure.reshape((nx, ny))  # reshape back to 2D

plt.figure(figsize=(6, 5))
plt.imshow(pressure.T, origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
plt.colorbar(label="Pressure")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Pressure Field")
plt.show()
