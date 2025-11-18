import pickle
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import mph
from jax import numpy as jnp

from pinnlib.data_utils import PointCloud

# Start client
print("starting client...")
client = mph.start()  # pyright: ignore

# Load model
path = Path("/home/sn/ws/comsol-scripts/sq.mph")
print(f"loading model from {path}...")
model = client.load(path)

# Get variables of interest
data = model.evaluate(
    ["x", "y", "z", "acpr.p_t", "acpr.vz"],
)

x, y, z, p, vz = data

# convert coordinates into reals
x, y, z = map(lambda x: x.real, (x, y, z))
coords = jnp.array([x, y, z])

pressure_dataset = PointCloud(coords, jnp.array(p))
velocity_dataset = PointCloud(coords, jnp.array(vz))

save_path = Path("./data/baffled_piston.pkl")
with open(save_path, "wb") as f:
    pickle.dump((pressure_dataset, velocity_dataset), f)

print(f" data saved to path {save_path}")
