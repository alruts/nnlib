# training
#  in -> [embedding] -> [neural net] -> d_fn \
#                                   -> r_fn  > out

import pickle
from pathlib import Path
from typing import Iterator

import equinox as eqx
import optax
from jax import random as jrandom
from jaxtyping import Array, Float, Int, PyTree
from test_bench.discretize import Point2d

from nnlib.dataload.sampling import DataPointSampler
from nnlib.geometry import subsample
from nnlib.pinn import WavePINN

# load data structure from .pkl file
data_path = Path("./data/gt_data.pkl")
with open(data_path, "rb") as f:
    data = pickle.load(f)

key = jrandom.PRNGKey(0)
data_key, subsample_key, *other = jrandom.split(key, 3)

data = subsample.full_data(
    data=data,
    coord_structure=Point2d,
)

dataset = DataPointSampler(
    data=data,
    batch_size=4,
    key=data_key,
)

dataloader = iter(dataset)
from pprint import pprint

for _ in range(1):
    batch = next(dataloader)
    pprint(batch)


# to-do: overhaul the data structures used into these `pointcloud` things for massive speedup
