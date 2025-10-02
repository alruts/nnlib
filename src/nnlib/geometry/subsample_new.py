from typing import Tuple

import jax.numpy as jnp
from jax import random as jrandom
from nn import PointCloudData, PRNGKeyArray, SpatialDiscretisationND


# --- Grid sample ---
def grid_sample(
    data: SpatialDiscretisationND,
    num_indices_per_dim: Tuple[int, ...],
) -> PointCloudData:
    """
    Grid sample SpatialDiscretisationND object at specified number of points per dimension.

    Returns:
        PointCloudData(coords=(N, D), values=(N,))
    """
    if len(num_indices_per_dim) != data.ndim:
        raise ValueError(
            "Length of num_indices_per_dim must match number of dimensions in data"
        )

    # sample indices in a grid
    get_at = [
        jnp.linspace(0, dim_size - 1, n, dtype=int)
        for dim_size, n in zip(data.vals.shape, num_indices_per_dim)
    ]
    get_at = jnp.meshgrid(*get_at, indexing="ij")
    get_at = tuple(get_at)  # tuple indexing

    # gather values and coordinates
    points = jnp.stack([c[get_at].ravel() for c in data.coordinate_arrays], axis=-1)
    vals = data.vals[get_at].ravel()

    return PointCloudData(coords=points, values=vals)


# --- Random sample ---
def random_sample(
    data: SpatialDiscretisationND,
    num_points: int,
    *,
    key: PRNGKeyArray,
) -> PointCloudData:
    """
    Randomly sample SpatialDiscretisationND object at a specified number of points.

    Returns:
        PointCloudData(coords=(N, D), values=(N,))
    """
    vals = data.vals.ravel()
    coords = [x.ravel() for x in data.coordinate_arrays]
    get_at = jrandom.choice(
        key, jnp.arange(vals.size), shape=(num_points,), replace=False
    )

    points = jnp.stack([c[get_at] for c in coords], axis=-1)
    vals = vals[get_at]

    return PointCloudData(coords=points, values=vals)


# --- Full data ---
def full_data(data: SpatialDiscretisationND) -> PointCloudData:
    """
    Return all points and values from a SpatialDiscretisationND object.

    Returns:
        PointCloudData(coords=(N, D), values=(N,))
    """
    vals = data.vals.ravel()
    coords = [x.ravel() for x in data.coordinate_arrays]
    points = jnp.stack(coords, axis=-1)

    return PointCloudData(coords=points, values=vals)
