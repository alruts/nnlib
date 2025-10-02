from typing import Callable, NamedTuple, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from test_bench.discretize import Point


class SpatialDiscretisationND(eqx.Module):
    bounds: Sequence[tuple[float, float]] = eqx.field(static=True)
    vals: Float[Array, "n_points ..."]  # noqa: F722

    @classmethod
    def discretise_fn(
        cls,
        bounds: Sequence[tuple[float, float]],
        n_points: Sequence[int],
        fn: Callable,
    ):
        if len(bounds) != len(n_points):
            raise ValueError("`bounds` and `n_points` must have same length")

        if any(n < 2 for n in n_points):
            raise ValueError("Each dimension must have at least 2 points")

        # Generate coordinate arrays per dimension
        axes = [
            jnp.linspace(start, end, num) for (start, end), num in zip(bounds, n_points)
        ]

        # Create meshgrid of coordinates
        mesh = jnp.meshgrid(*axes, indexing="ij")

        # Stack to get array of shape (..., N), then reshape to (-1, N)
        coords = jnp.stack(mesh, axis=-1).reshape(-1, len(bounds))

        # Evaluate the function on all points
        # fn should accept a vector input of shape (ndim,)
        vals_flat = jax.vmap(fn)(coords)
        vals = vals_flat.reshape(*n_points)  # Reshape back to grid shape

        return cls(bounds, vals)

    @property
    def n_points(self):
        return self.vals.shape[0]

    @property
    def ndim(self):
        return len(self.bounds)

    @property
    def dxs(self):
        return jnp.array(
            [
                (end - start) / (n - 1)
                for (start, end), n in zip(self.bounds, self.vals.shape)
            ]
        )

    @property
    def coordinate_arrays(self):
        axes = [
            jnp.linspace(start, end, num)
            for (start, end), num in zip(self.bounds, self.vals.shape)
        ]
        return jnp.meshgrid(*axes, indexing="ij")  # Returns list of N arrays

    def locate_closest(self, point: Point):
        flat_coords, pt = jnp.stack(self.coordinate_arrays, -1), jnp.array(tuple(point))
        flat_idx = jnp.argmin(jnp.sum(flat_coords - pt) ** 2)
        return jnp.unravel_index(flat_idx, self.vals.shape)

    def binop(self, other, fn):
        if isinstance(other, SpatialDiscretisationND):
            if self.bounds != other.bounds or self.vals.shape != other.vals.shape:
                raise ValueError("Mismatched spatial discretisations")
            other = other.vals
        return SpatialDiscretisationND(self.bounds, fn(self.vals, other))

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


class UnstructuredDiscretisationND(eqx.Module):
    coords: Float[Array, "ndim n_points"]  # noqa: F722
    vals: Float[Array, "n_points"]

    @classmethod
    def discretise_fn(
        cls,
        coords: Array,
        fn: Callable,
    ):
        if coords.ndim != 2:
            raise ValueError("coords must be shape (N, ndim)")
        vals = jax.vmap(fn)(coords)  # Evaluate function at each point
        return cls(coords, vals)

    @property
    def ndim(self):
        return self.coords.shape[1]

    @property
    def n_points(self):
        return self.coords.shape[0]

    def locate_closest(self, point: Point):
        """Return index of closest point in the cloud."""
        pt = jnp.array(point)
        dists = jnp.sum((self.coords - pt) ** 2, axis=1)
        return jnp.argmin(dists)

    def binop(self, other, fn):
        if isinstance(other, UnstructuredDiscretisationND):
            if not jnp.allclose(self.coords, other.coords):
                raise ValueError(
                    "Mismatched point sets for unstructured discretisations"
                )
            other = other.vals
        return UnstructuredDiscretisationND(self.coords, fn(self.vals, other))

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


# Type hint for points
CartesianPoint = Float[Array, "n_dim"]
SpaceTimePoint = Float[Array, "n_dim + 1"]
SpaceFreqPoint = Float[Array, "n_dim + 1"]


class PointCloudData(NamedTuple):
    coords: Float[Array, "n_points n_dim"]
    values: Float[Array, "n_points"]


# idea for how to split into categories for more complex problems
class PINNPointCloudData(NamedTuple):
    interior: PointCloudData
    boundary: PointCloudData
    initial: PointCloudData
