import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from PIL import Image

from nnlib.data_utils.data_structures import SpatialDiscretisationND
from nnlib.data_utils.subset import grid_sample, random_sample

# Load image
file = Path(
    "./data/The-famous-Lena-image-often-used-as-an-example-in-image-processing.png"
)
img = Image.open(file)
img = jnp.array(img)[:, :, 0]  # take first channel
img = img / 255.0
img = jnp.rot90(img, k=-1)

# data structure to represent "grid" like data
data = SpatialDiscretisationND(
    [(-1.0, 1.0), (-1.0, 1.0)],
    vals=img,
)

# Serialize the data object
save_path = Path("./data/gt_data.pkl")
with open(save_path, "wb") as f:
    pickle.dump(data, f)
print(f"SpatialDiscretisationND object saved to {save_path}")

# To load it back later:
with open(save_path, "rb") as f:
    data = pickle.load(f)

# Plot using the grid
plt.figure(figsize=(6, 5))

# Original
plt.pcolormesh(*data.coordinate_arrays, data.vals, shading="auto", cmap="jet")
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
subset = random_sample(data, num_points=num_points, key=key)

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
subset = grid_sample(data, num_points)

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
