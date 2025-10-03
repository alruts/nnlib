import pickle
from pathlib import Path
from typing import Iterator

import equinox as eqx
import jax
import optax
from jax import numpy as jnp
from jax import random as jrandom
from jaxtyping import Array, Float, Int, PyTree
from test_bench.discretize import Point2d

from nnlib import architectures, embeddings
from nnlib.dataload import subset
from nnlib.dataload.sampling import DataPointSampler
from nnlib.misc import get_parameters
from nnlib.pinn import WavePINN

# load data structure from .pkl file
data_path = Path("./data/gt_data.pkl")
with open(data_path, "rb") as f:
    data = pickle.load(f)

key = jrandom.PRNGKey(0)
data_key, subsample_key, net_key = jrandom.split(key, 3)

data = subset.full_data(data=data)

dataset = DataPointSampler(
    point_cloud=data,
    batch_size=4,
    key=data_key,
)

dataloader = iter(dataset)

# define architecture
emb = embeddings.PeriodicEmbedding(periods=(30.0, 40.0))
pinn = WavePINN.create(
    embedding=emb,
    arch_name="modified_mlp",
    in_size=2,
    out_size="scalar",
    width_size=8,
    depth=3,
    key=net_key,
)

params = get_parameters(pinn.model)
print(params)
print(pinn.p_net(params, 2.0, 1.0))


criteria = {
    "mse": lambda x, y, axis=None: jnp.mean((x - y) ** 2, axis),
    "mae": lambda x, y, axis=None: jnp.mean(jnp.abs(x - y), axis),
}


# define loss
def data_loss(model, params, batch, criterion=criteria["mse"]):
    *coords, vals = batch
    n_dim = len(coords)
    batched_p_net = jax.vmap(model.p_net, in_axes=(None, *[0] * n_dim))
    parallel_p_net = jax.pmap(batched_p_net, in_axes=(None, *[0] * n_dim))
    pred = parallel_p_net(params, *coords)
    return criterion(pred, vals)


# tester loop
train_steps = 100
for step, batch in zip(range(100), dataloader):
    loss, grad = jax.value_and_grad(data_loss, argnums=1)(pinn, params, batch)
    print(f"step {step}: {loss}")
