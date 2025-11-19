from functools import reduce
from typing import NamedTuple

import jax
from jaxtyping import Array, Float


class PointCloud(NamedTuple):
    """
    Data structure for a point cloud.

    >>> import jax.numpy as jnp
    >>> pc = PointCloud(coords=jnp.array([[0.0, 1.0], [2.0, 3.0]]),
    ...                 vals=jnp.array([10.0, 20.0]))
    >>> pc.coords.shape
    (2, 2)
    """

    coords: Float[Array, "n_points n_dim"]
    vals: Float[Array, "n_points"]


def map_coords(fn):
    """
    Returns a transformation that applies fn to coords.

    >>> import jax.numpy as jnp
    >>> pc = PointCloud(coords=jnp.array([[1., 2.], [3., 4.]]),
    ...                 vals=jnp.array([10., 20.]))
    >>> translate = map_coords(lambda c: c + jnp.array([1., 0.]))
    >>> out = translate(pc)
    >>> print(out.coords)
    [[2. 2.]
     [4. 4.]]
    >>> print(jnp.all(out.vals == pc.vals))
    True
    """

    def _apply(pc: PointCloud) -> PointCloud:
        return PointCloud(coords=fn(pc.coords), vals=pc.vals)

    return _apply


def map_vals(fn):
    """
    Returns a transformation that applies fn to vals.

    >>> import jax.numpy as jnp
    >>> pc = PointCloud(coords=jnp.array([[0., 0.], [1., 1.]]),
    ...                 vals=jnp.array([1., 2.]))
    >>> double = map_vals(lambda v: v * 2)
    >>> out = double(pc)
    >>> print(out.vals)
    [2. 4.]
    >>> print(jnp.all(out.coords == pc.coords))
    True
    """

    def _apply(pc: PointCloud) -> PointCloud:
        return PointCloud(coords=pc.coords, vals=fn(pc.vals))

    return _apply


def filter_points(predicate):
    """
    Filter points by a predicate(coords, vals) → boolean mask.

    >>> import jax.numpy as jnp
    >>> pc = PointCloud(coords=jnp.array([[0., 0.], [1., 1.], [2., 2.]]),
    ...                 vals=jnp.array([5., -3., 7.]))
    >>> keep_positive = filter_points(lambda c, v: v > 0)
    >>> out = keep_positive(pc)
    >>> print(out.coords)
    [[0. 0.]
     [2. 2.]]
    >>> print(out.vals)
    [5. 7.]
    """

    def _apply(pc: PointCloud) -> PointCloud:
        mask = predicate(pc.coords, pc.vals)
        return PointCloud(coords=pc.coords[mask], vals=pc.vals[mask])

    return _apply


def sample_points(key, n_samples):
    """
    Returns a transformation that randomly samples n_samples points
    from a PointCloud using jax.random.choice.

    >>> import jax.random as jr
    >>> import jax.numpy as jnp
    >>> key = jr.PRNGKey(0)
    >>> pc = PointCloud(
    ...     coords=jnp.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.]]),
    ...     vals=jnp.array([10., 20., 30., 40.]),
    ... )
    >>> sampler = sample_points(key, 2)
    >>> out = sampler(pc)
    >>> out.coords.shape[0]
    2
    """

    def _apply(pc: PointCloud) -> PointCloud:
        n_points = pc.coords.shape[0]
        idx = jax.random.choice(key, n_points, shape=(n_samples,), replace=False)
        return PointCloud(coords=pc.coords[idx], vals=pc.vals[idx])

    return _apply


def pipe(*funcs):
    """
    Compose multiple transformations into a pipeline.

    >>> import jax.numpy as jnp
    >>> pc = PointCloud(
    ...     coords=jnp.array([[1., 1.], [2., 2.]]),
    ...     vals=jnp.array([10., 20.])
    ... )
    >>> scale = map_coords(lambda c: c * 0.5)
    >>> shift = map_coords(lambda c: c - 1.0)
    >>> pipeline = pipe(scale, shift)
    >>> out = pipeline(pc)
    >>> print(out.coords)
    [[-0.5 -0.5]
     [ 0.   0. ]]
    """

    def composed(x):
        return reduce(lambda v, f: f(v), funcs, x)

    return composed
