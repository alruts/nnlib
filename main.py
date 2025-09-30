import pickle
from pathlib import Path
from typing import Tuple, Type

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray
from matplotlib import pyplot as plt
from PIL import Image
from test_bench.discretize import Point2d, SpatialDiscretisationND

# Load image
file = Path(
    "./data/The-famous-Lena-image-often-used-as-an-example-in-image-processing.png"
)
img = Image.open(file)
img = jnp.array(img)[:, :, 0]  # take first channel
img = img / 255.0
img = jnp.rot90(img, k=-1)

data = SpatialDiscretisationND(
    [(-1.0, 1.0), (-1.0, 1.0)],
    vals=img,
)


def grid_sample(
    data: SpatialDiscretisationND,
    num_indices_per_dim: Tuple[int, ...],
    *,
    structure: Type[Point2d] = Point2d,
) -> list[tuple[Point2d, float]]:
    """Grid sample SpatialDiscretisationND object at specified number of points per dimension"""

    if len(num_indices_per_dim) != data.ndim:
        raise ValueError(
            "Length of num_indices_per_dim must match number of dimensions in data"
        )

    # generate indices for each dimension
    sampled_idxs = [
        jnp.linspace(0, dim_size - 1, n, dtype=int)
        for dim_size, n in zip(data.vals.shape, num_indices_per_dim)
    ]
    sampled_idxs = jnp.meshgrid(*sampled_idxs, indexing="ij")
    sampled_idxs = tuple(sampled_idxs)

    # gather values and coordinates
    sampled_points = jnp.stack(
        [c[sampled_idxs].ravel() for c in data.coordinate_arrays], axis=-1
    )
    sampled_vals = data.vals[sampled_idxs].ravel()

    return [(structure(*pt), v) for pt, v in zip(sampled_points, sampled_vals)]


def random_sample(
    data: SpatialDiscretisationND,
    num_points: int,
    *,
    structure: Type[Point2d] = Point2d,
    key: PRNGKeyArray,
) -> list[tuple[Point2d, float]]:
    """Randomly sample SpatialDiscretisationND object at a specified number of points"""

    # sample random indices `num_
    vals = data.vals.ravel()
    coords = [x.ravel() for x in data.coordinate_arrays]
    sampled_idxs = jrandom.choice(
        key, jnp.arange(vals.size), shape=(num_points,), replace=False
    )

    # gather values and coordinates
    sampled_points = jnp.stack([c[sampled_idxs] for c in coords], axis=-1)
    sampled_vals = vals[sampled_idxs]

    return [(structure(*pt), v) for pt, v in zip(sampled_points, sampled_vals)]


# Serialize the data object
save_path = Path("./data/gt_data.pkl")
with open(save_path, "wb") as f:
    pickle.dump(data, f)
print(f"SpatialDiscretisationND object saved to {save_path}")

# To load it back later:
with open(save_path, "rb") as f:
    data = pickle.load(f)

f_y, f_x = jnp.gradient(data.vals, *data.dxs)  # specify spacing from coordinates

f_yy = jnp.gradient(f_y, data.dxs[1], axis=1)
f_xx = jnp.gradient(f_x, data.dxs[0], axis=0)

# Sum derivatives
first_order_sum = f_x + f_y
second_order_sum = f_xx + f_yy

# Plot using the grid
plt.figure(figsize=(18, 5))

# Original
plt.subplot(1, 3, 1)
plt.pcolormesh(*data.coordinate_arrays, data.vals, shading="auto", cmap="jet")
plt.title("Original Image")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")

# First-order derivatives sum
plt.subplot(1, 3, 2)
plt.pcolormesh(*data.coordinate_arrays, first_order_sum, shading="auto", cmap="jet")
plt.title("Sum of First-Order Derivatives")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")

# Second-order derivatives sum
plt.subplot(1, 3, 3)
plt.pcolormesh(*data.coordinate_arrays, second_order_sum, shading="auto", cmap="jet")
plt.title("Sum of Second-Order Derivatives")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")

plt.tight_layout()
plt.show()

# ===

key = jax.random.PRNGKey(0)

# Sample random points
num_points = 2000
sampled = random_sample(data, num_points=num_points, key=key, structure=Point2d)

# Extract coordinates and values
coords = jnp.array([[*p] for p, _ in sampled])
vals = jnp.array([v for _, v in sampled])

# Create a "masked" grid where unsampled points are gray
# Start with a grid filled with NaNs
masked_vals = jnp.full_like(data.vals, jnp.nan)

# Convert coordinates to indices in the original grid
# Assuming data.coordinate_arrays[0] = y coords, [1] = x coords
y_coords, x_coords = data.coordinate_arrays
ny, nx = data.vals.shape

# Map sampled coordi]]nates back to nearest grid indices
x_idxs = jnp.searchsorted(x_coords[0, :], coords[:, 0])
y_idxs = jnp.searchsorted(y_coords[:, 0], coords[:, 1])

# Clip indices to avoid out-of-bounds
x_idxs = jnp.clip(x_idxs, 0, nx - 1)
y_idxs = jnp.clip(y_idxs, 0, ny - 1)

# Fill in the sampled values
masked_vals = masked_vals.at[y_idxs, x_idxs].set(vals)

# Plot
plt.figure(figsize=(6, 6))
plt.pcolormesh(
    *data.coordinate_arrays,
    masked_vals,
    shading="auto",
    cmap="jet",
)
plt.title(f"Random {num_points} Sampled Points")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()

# Grid

# Sample random points
num_points = 2000
sampled = grid_sample(data, (100, 10), structure=Point2d)

# Extract coordinates and values
coords = jnp.array([[*p] for p, _ in sampled])
vals = jnp.array([v for _, v in sampled])

# Create a "masked" grid where unsampled points are gray
# Start with a grid filled with NaNs
masked_vals = jnp.full_like(data.vals, jnp.nan)

# Convert coordinates to indices in the original grid
# Assuming data.coordinate_arrays[0] = y coords, [1] = x coords
y_coords, x_coords = data.coordinate_arrays
ny, nx = data.vals.shape

# Map sampled coordinates back to nearest grid indices
x_idxs = jnp.searchsorted(x_coords[0, :], coords[:, 0])
y_idxs = jnp.searchsorted(y_coords[:, 0], coords[:, 1])

# Clip indices to avoid out-of-bounds
x_idxs = jnp.clip(x_idxs, 0, nx - 1)
y_idxs = jnp.clip(y_idxs, 0, ny - 1)

# Fill in the sampled values
masked_vals = masked_vals.at[y_idxs, x_idxs].set(vals)

# Plot
plt.figure(figsize=(6, 6))
plt.pcolormesh(
    *data.coordinate_arrays,
    masked_vals,
    shading="auto",
    cmap="jet",
)
plt.title(f"Random {num_points} Sampled Points")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()
