import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.scipy.special import erf
from matplotlib import pyplot as plt

from nnlib.data_utils.data_structures import GridDiscretisationND
from nnlib.data_utils.subset import grid_sample, random_sample
from nnlib.misc import default_wave_speed

c = default_wave_speed()


def acoustic_point_source_1d(coord):
    """
    1D acoustic point source u(x,t) with a Gaussian pulse.
    Assumes constant wave speed c.

    Gaussian pulse: S(t) = exp(-(t-t0)^2 / (2*sigma^2))
    """
    x, t = coord
    sigma = 0.05  # pulse width
    t0 = 0.3

    # Retarded time
    t_ret = t - jnp.abs(x) / c
    t_ret = jnp.maximum(t_ret, 0.0)

    # Analytical integral of Gaussian
    u = (sigma * jnp.sqrt(jnp.pi / 2) / (2 * c)) * (
        erf((t_ret - t0) / (jnp.sqrt(2) * sigma)) - erf(-t0 / (jnp.sqrt(2) * sigma))
    )

    return u * 100


# generate 'grid' data using acoustic point source function
spatial_discretisation = GridDiscretisationND.discretise_fn(
    [(0.0, 1.0), (0.0, 1.0)], [256, 256], acoustic_point_source_1d
)

# Serialize the data object
save_path = Path("./data/gt_data.pkl")
with open(save_path, "wb") as f:
    pickle.dump(spatial_discretisation, f)
print(f"GridDiscretisationND object saved to {save_path}")

# To load it back later:
with open(save_path, "rb") as f:
    spatial_discretisation = pickle.load(f)

# Plot using the grid
plt.figure(figsize=(6, 5))

# Original
plt.pcolormesh(
    *spatial_discretisation.coordinate_arrays,
    spatial_discretisation.vals,
    shading="auto",
    cmap="jet",
)
plt.title("Original Image")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.tight_layout()

plt.show()

# ===
key = jax.random.PRNGKey(0)

# Sample random points
num_points = 20_000
subset = random_sample(spatial_discretisation, num_points=num_points, key=key)

# Extract coordinates and values
coords = subset.coords
vals = subset.vals

# Scatter plot
plt.figure(figsize=(6, 5))
sc = plt.scatter(
    subset.coords[:, 0],  # x coordinates
    subset.coords[:, 1],  # y coordinates
    c=vals,  # values mapped to color
    cmap="jet",
    s=1,  # marker size
    edgecolor="none",  # optional: black edge around points
)
plt.colorbar(sc, label="Value")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Random {num_points} Sampled Points")
plt.axis("equal")
plt.show()

# Grid
num_points = (100, 100)
# Sample random points
subset = grid_sample(spatial_discretisation, num_points)

# Extract coordinates and values
coords = subset.coords
vals = subset.vals

# Scatter plot
plt.figure(figsize=(6, 5))
sc = plt.scatter(
    subset.coords[:, 0],  # x coordinates
    subset.coords[:, 1],  # y coordinates
    c=vals,  # values mapped to color
    cmap="jet",
    s=1,  # marker size
    edgecolor="none",  # optional: black edge around points
)
plt.colorbar(sc, label="Value")
plt.xlabel("x")
plt.ylabel("t")
plt.title(f"Random {num_points} Sampled Points")
plt.axis("equal")
plt.show()
