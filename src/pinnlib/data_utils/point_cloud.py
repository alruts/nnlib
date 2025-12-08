from collections.abc import Callable, Sequence
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar

# base data annotations
Coords = tuple[Array, ...]
Coord = tuple[Scalar, ...]
Vecs = tuple[Array, ...]
Vec = tuple[Scalar, ...]
CoordsVecs = tuple[Array, ...]
CoordVec = tuple[Scalar, ...]
Vals = Array


class PointCloud(NamedTuple):
    """
    Data structure for a point cloud where coords is a tuple of 1D arrays.
    """

    coords: Coords
    vals: Vals


class GridDiscretisationND(eqx.Module):
    bounds: Sequence[tuple[float, float]] = eqx.field(static=True)
    vals: Float[Array, "n_points ..."]

    @classmethod
    def discretise_fn(
        cls,
        bounds: Sequence[tuple[float, float]],
        n_points: Sequence[int],
        fn: Callable,
    ):
        """
        Discretise a function over a regular grid.

        >>> def f(x): return x[0] + x[1]
        >>> data = GridDiscretisationND.discretise_fn(bounds=[(0,1),(0,1)], n_points=[2,2], fn=f)
        >>> data.vals.shape
        (2, 2)
        >>> data.bounds
        [(0, 1), (0, 1)]
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
        vals = vals_flat.reshape(*n_points)
        return cls(bounds, vals)

    @property
    def coordinate_arrays(self):
        axes = [
            jnp.linspace(start, end, num)
            for (start, end), num in zip(self.bounds, self.vals.shape)
        ]
        return tuple(jnp.meshgrid(*axes, indexing="ij"))

    def as_point_cloud(self):
        coords = tuple(x.flatten() for x in self.coordinate_arrays)
        vals = self.vals.flatten()
        return PointCloud(coords, vals)

    def binop(self, other, fn):
        if isinstance(other, GridDiscretisationND):
            if self.bounds != other.bounds or self.vals.shape != other.vals.shape:
                raise ValueError("Mismatched spatial discretisations")
            other = other.vals
        return GridDiscretisationND(self.bounds, fn(self.vals, other))

    def __add__(self, other):
        return self.binop(other, lambda x, y: x + y)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.binop(other, lambda x, y: x - y)

    def __rsub__(self, other):
        return self.binop(other, lambda x, y: y - x)

    def __mul__(self, other):
        return self.binop(other, lambda x, y: x * y)

    def __rmul__(self, other):
        return self.__mul__(other)
