from typing import Tuple, Type

import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray
from test_bench.discretize import Point, SpatialDiscretisationND


#! to-do: add doctest
def grid_sample(
    data: SpatialDiscretisationND,
    num_indices_per_dim: Tuple[int, ...],
    *,
    coord_structure: Type[Point],
) -> list[tuple[Point, float]]:
    """Grid sample SpatialDiscretisationND object at specified number of points per dimension"""

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
    get_at = tuple(get_at)  # indexing must use tuples

    # gather values and coordinates
    points = jnp.stack([c[get_at].ravel() for c in data.coordinate_arrays], axis=-1)
    vals = data.vals[get_at].ravel()

    return [(coord_structure(*pt), v) for pt, v in zip(points, vals)]


#! to-do: add doctest
def random_sample(
    data: SpatialDiscretisationND,
    num_points: int,
    *,
    coord_structure: Type[Point],
    key: PRNGKeyArray,
) -> list[tuple[Point, float]]:
    """Randomly sample SpatialDiscretisationND object at a specified number of points"""

    # sample random indices
    vals = data.vals.ravel()
    coords = [x.ravel() for x in data.coordinate_arrays]
    get_at = jrandom.choice(
        key, jnp.arange(vals.size), shape=(num_points,), replace=False
    )

    # gather values and coordinates
    points = jnp.stack([c[get_at] for c in coords], axis=-1)
    vals = vals[get_at]

    return [(coord_structure(*pt), v) for pt, v in zip(points, vals)]


def full_data(
    data: SpatialDiscretisationND,
    *,
    coord_structure: type[Point],
) -> list[tuple[Point, float]]:
    """Return all points and values from a SpatialDiscretisationND object"""

    # flatten values and coordinates
    vals = data.vals.ravel()
    coords = [x.ravel() for x in data.coordinate_arrays]

    # stack coordinates
    points = jnp.stack(coords, axis=-1)

    # return as list of (Point, value) tuples
    return [(coord_structure(*pt), v) for pt, v in zip(points, vals)]
