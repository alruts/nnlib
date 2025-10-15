from typing import Callable, NamedTuple, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

# Type hint for points
CartesianPoint = Float[Array, "n_spatial"]
SpaceTimePoint = Float[Array, "n_spatial + 1"]
SpaceFreqPoint = Float[Array, "n_spatial + 1"]

Point = CartesianPoint | SpaceTimePoint | SpaceFreqPoint
Triangle = Float[jnp.ndarray, "3 n_spatial"]


class PointCloud(NamedTuple):
    """Data structure for a point cloud"""

    coords: Float[Array, "n_points n_dim"]
    vals: Float[Array, "n_points"]


class GridDiscretisationND(eqx.Module):
    bounds: Sequence[tuple[float, float]] = eqx.field(static=True)
    vals: Float[Array, "n_points ..."]  # noqa: F722

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
    def n_points(self):
        """
        >>> data = GridDiscretisationND.discretise_fn(bounds=[(0,1)], n_points=[5], fn=lambda x: x[0])
        >>> data.n_points
        5
        """
        return self.vals.shape[0]

    @property
    def ndim(self):
        """
        >>> data = GridDiscretisationND.discretise_fn(bounds=[(0,1),(0,2)], n_points=[2,3], fn=lambda x: x[0]+x[1])
        >>> data.ndim
        2
        """
        return len(self.bounds)

    @property
    def dxs(self):
        """
        >>> data = GridDiscretisationND.discretise_fn(bounds=[(0,1),(0,3)], n_points=[2,4], fn=lambda x: x[0]+x[1])
        >>> list(map(float, data.dxs))
        [1.0, 1.0]
        """
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
        return tuple(jnp.meshgrid(*axes, indexing="ij"))

    def locate_closest(self, point: Point):
        """
        Locate the index of the closest grid point.

        >>> data = GridDiscretisationND.discretise_fn(bounds=[(0,1),(0,1)], n_points=[2,2], fn=lambda x: x[0]+x[1])
        >>> data.locate_closest(jnp.array([0.1, 0.9]))
        (Array(0, dtype=int32), Array(1, dtype=int32))
        """
        flat_coords, pt = jnp.stack(self.coordinate_arrays, -1), point
        flat_idx = jnp.argmin(jnp.sum((flat_coords - pt) ** 2, axis=-1))
        return jnp.unravel_index(flat_idx, self.vals.shape)

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


class UnstructuredDiscretisationND(eqx.Module):
    coords: Float[Array, "n_points ndim"]  # noqa: F722
    vals: Float[Array, "n_points"]

    @classmethod
    def discretise_fn(
        cls,
        coords: Array,
        fn: Callable,
    ):
        """
        Evaluate a function at unstructured points.
        Example:
            >>> coords = jnp.array([[0,0],[1,1],[0.5,0.5]])
            >>> udf = UnstructuredDiscretisationND.discretise_fn(coords, lambda x: x[0]+x[1])
            >>> udf.vals
            Array([0., 2., 1.], dtype=float32)
        """
        if coords.ndim != 2:
            raise ValueError("coords must be shape (N, ndim)")
        vals = jax.vmap(fn)(coords)
        return cls(coords, vals)

    @property
    def ndim(self):
        """
        >>> coords = jnp.array([[0,0],[1,1]])
        >>> udf = UnstructuredDiscretisationND.discretise_fn(coords, lambda x: x[0]+x[1])
        >>> udf.ndim
        2
        """
        return self.coords.shape[1]

    @property
    def n_points(self):
        """
        >>> coords = jnp.array([[0,0],[1,1]])
        >>> udf = UnstructuredDiscretisationND.discretise_fn(coords, lambda x: x[0]+x[1])
        >>> udf.n_points
        2
        """
        return self.coords.shape[0]

    def locate_closest(self, point: Point):
        """
        Return index of closest point in the cloud.

        >>> coords = jnp.array([[0,0],[1,1],[0.5,0.5]])
        >>> udf = UnstructuredDiscretisationND.discretise_fn(coords, lambda x: x[0]+x[1])
        >>> print(udf.locate_closest(jnp.array([0.6,0.6])))
        2
        """
        dists = jnp.sum((self.coords - point) ** 2, axis=1)
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


# mesh datastructure
