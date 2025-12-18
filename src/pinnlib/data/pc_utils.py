from collections.abc import Callable, Sequence
from functools import reduce

import jax
from jax import numpy as jnp

from pinnlib.data.point_cloud import PointCloud


def map_coords(fn):
    """
    Returns a transformation that applies fn to the unpacked coordinate arrays of a PointCloud.

    >>> import jax.numpy as jnp
    >>> pc = PointCloud(coords=(jnp.array([1., 3.]), jnp.array([2., 4.])),
    ...                 vals=jnp.array([10., 20.]))
    >>> translate = map_coords(lambda *coords: (coords[0] + 1, coords[1] - 1))
    >>> out = translate(pc)
    >>> x, y = translate(pc).coords
    >>> print(x, y)
    [2. 4.] [1. 3.]
    >>> print(jnp.all(out.vals == pc.vals))
    True
    """

    def _apply(pc):
        new_coords = fn(*pc.coords)
        return PointCloud(coords=new_coords, vals=pc.vals)

    return _apply


def map_vals(fn):
    """
    Returns a transformation that applies fn to vals.

    >>> import jax.numpy as jnp
    >>> pc = PointCloud(coords=(jnp.array([0., 1.]), jnp.array([0., 1.])),
    ...                 vals=jnp.array([1., 2.]))
    >>> double = map_vals(lambda v: v * 2)
    >>> out = double(pc)
    >>> print(out.vals)
    [2. 4.]
    >>> all(jnp.all(c == oc) for c, oc in zip(out.coords, pc.coords))
    True
    """

    def _apply(pc: PointCloud) -> PointCloud:
        return PointCloud(coords=pc.coords, vals=fn(pc.vals))

    return _apply


def filter_points(predicate):
    """
    Filter points by a predicate(coords, vals) → boolean mask.

    >>> import jax.numpy as jnp
    >>> pc = PointCloud(coords=(jnp.array([0., 1., 2.]), jnp.array([0., 1., 2.])),
    ...                 vals=jnp.array([5., -3., 7.]))
    >>> keep_positive = filter_points(lambda c, v: v > 0)
    >>> out = keep_positive(pc)
    >>> x, y = out.coords
    >>> print(x, y)
    [0. 2.] [0. 2.]
    >>> print(out.vals)
    [5. 7.]
    """

    def _apply(pc: PointCloud) -> PointCloud:
        mask = predicate(pc.coords, pc.vals)
        return PointCloud(coords=tuple(c[mask] for c in pc.coords), vals=pc.vals[mask])

    return _apply


def grid_sample_points(grid_size: tuple[int, ...]):
    """
    Sample points from a PointCloud using a uniform grid in arbitrary dimensions.

    Args:
        grid_size: Tuple defining the number of samples along each axis (e.g., (nx, ny, nz) for 3D).

    Returns:
        A function that takes a PointCloud and returns a subsampled PointCloud.
    """

    def _apply(pc):
        coords = pc.coords  # tuple of arrays, shape (n_points,)
        dims = len(coords)
        n_points = len(pc.vals)

        # Compute bounding box for each axis
        mins = jnp.array([jnp.min(c) for c in coords])
        maxs = jnp.array([jnp.max(c) for c in coords])

        # Build uniform grid
        linspaces = [jnp.linspace(mins[i], maxs[i], grid_size[i]) for i in range(dims)]
        mesh = jnp.meshgrid(*linspaces, indexing="ij")
        grid_points = jnp.stack(
            [m.ravel() for m in mesh], axis=-1
        )  # shape (n_grid_points, dims)

        # Stack coords for distance computation
        pc_points = jnp.stack(coords, axis=-1)  # shape (n_points, dims)

        # For each grid point, find the nearest point in the point cloud
        def nearest_idx(pt):
            dists = jnp.sum((pc_points - pt) ** 2, axis=-1)
            return jnp.argmin(dists)

        idxs = jax.vmap(nearest_idx)(grid_points)
        idxs = jnp.unique(idxs)  # remove duplicates

        return PointCloud(coords=tuple(c[idxs] for c in coords), vals=pc.vals[idxs])

    return _apply


def sample_points(key, n_samples):
    """
    Randomly sample n_samples points from a PointCloud.

    >>> import jax.random as jr
    >>> import jax.numpy as jnp
    >>> pc = PointCloud(coords=(jnp.array([0., 1., 2., 3.]), jnp.array([0., 1., 2., 3.])),
    ...                 vals=jnp.array([10., 20., 30., 40.]))
    >>> key = jr.PRNGKey(0)
    >>> sampler = sample_points(key, 2)
    >>> out = sampler(pc)
    >>> out.coords[0].shape[0]
    2
    """

    def _apply(pc: PointCloud) -> PointCloud:
        n_points = len(pc.vals)
        idx = jax.random.choice(key, n_points, shape=(n_samples,), replace=False)
        return PointCloud(coords=tuple(c[idx] for c in pc.coords), vals=pc.vals[idx])

    return _apply


def pipe(*funcs):
    """
    Compose multiple transformations into a pipeline.
    """

    def composed(x):
        return reduce(lambda v, f: f(v), funcs, x)

    return composed


def get_bounding_box(pc: PointCloud) -> list[tuple[float, float]]:
    """
    Get a bounding box for a PointCloud as [(xmin, xmax), (ymin, ymax), ...].

    >>> import jax.numpy as jnp
    >>> pc = PointCloud(coords=(jnp.array([-1., 1., 3.]), jnp.array([-3., 2., 4.])),
    ...                 vals=jnp.array([10., 20.]))
    >>> get_bounding_box(pc)
    [(-1.0, 3.0), (-3.0, 4.0)]
    """
    return [(float(c.min()), float(c.max())) for c in pc.coords]


def discretise_fn(
    bounds: Sequence[tuple[float, float]],
    n_points: Sequence[int],
    fn: Callable,
) -> PointCloud:
    """
    Discretise a function over a regular grid.

    >>> def f(x): return x[0] + x[1]
    >>> pc = discretise_fn(bounds=[(0, 1),(0, 1)], n_points=[2, 2], fn=f)
    >>> pc.vals.shape
    (4,)
    >>> get_bounding_box(pc)
    [(0.0, 1.0), (0.0, 1.0)]
    """
    if len(bounds) != len(n_points):
        raise ValueError("`bounds` and `n_points` must have same length")

    if any(n < 2 for n in n_points):
        raise ValueError("Each dimension must have at least 2 points")

    axes = [
        jnp.linspace(start, end, num) for (start, end), num in zip(bounds, n_points)
    ]

    mesh = jnp.meshgrid(*axes, indexing="ij")
    coords = jnp.stack(mesh, axis=-1).reshape(-1, len(bounds))
    vals_flat = jax.vmap(fn)(coords)
    return PointCloud(tuple(coords.T), vals_flat)
