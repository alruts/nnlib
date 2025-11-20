from functools import reduce
from typing import NamedTuple, Sequence

import jax
from jaxtyping import Array

# base data annotations
Coords = tuple[Array, ...]  # tuple of 1D arrays
Vals = Array


class PointCloud(NamedTuple):
    """
    Data structure for a point cloud where coords is a tuple of 1D arrays.
    """

    coords: Coords
    vals: Vals


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
        n_points = pc.coords[0].shape[0]
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
