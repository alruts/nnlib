from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random as jrandom
from jaxtyping import Array, Float, Int, PRNGKeyArray

from pinnlib.data_utils.point_cloud import PointCloud

# Type hint for points
CartesianPoint = Float[Array, "n_spatial"]
SpaceTimePoint = Float[Array, "n_spatial + 1"]
SpaceFreqPoint = Float[Array, "n_spatial + 1"]

Point = CartesianPoint | SpaceTimePoint | SpaceFreqPoint
Triangle = Float[jnp.ndarray, "3 n_spatial"]


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

    def transform(self, tx: Callable) -> "GridDiscretisationND":
        """
        Apply a transformation function to all values in the grid and return a new GridDiscretisationND.

        The transformation function should accept an array of the same shape as `vals` and return
        an array of the same shape.

        >>> import jax.numpy as jnp
        >>> data = GridDiscretisationND.discretise_fn(bounds=[(0,1)], n_points=[3], fn=lambda x: x[0])
        >>> print(data.vals)
        [0.  0.5 1. ]
        >>> squared = data.transform(lambda v: v**2)
        >>> print(squared.vals)
        [0.   0.25 1.  ]
        >>> squared.bounds
        [(0, 1)]
        """
        return GridDiscretisationND(self.bounds, tx(self.vals))

    def slice(self, **kwargs) -> "GridDiscretisationND":
        """
        Return a lower-dimensional slice of the grid.

        Specify one coordinate to slice along, e.g. x=0.5, y=0, z=0.3.

        >>> data = GridDiscretisationND.discretise_fn(bounds=[(0,1),(0,1)], n_points=[2,2], fn=lambda x: x[0]+x[1])
        >>> slice_xy = data.slice(x=0.0)  # slices along x=0
        >>> slice_xy.vals.shape
        (2,)
        """
        if len(kwargs) != 1:
            raise ValueError("You must specify exactly one coordinate to slice along.")

        coord_name, coord_val = next(iter(kwargs.items()))
        dim_map = {f"x{i}": i for i in range(self.ndim)}

        names = "xyzt"
        dim_map.update({name: i for i, name in enumerate(names[: self.ndim])})

        if coord_name not in dim_map:
            raise ValueError(f"Unknown coordinate '{coord_name}' for slicing.")

        dim = dim_map[coord_name]
        axis_vals = self.linspaces[dim]
        closest_idx = int(jnp.argmin(jnp.abs(axis_vals - coord_val)))

        # Slice the vals array along the chosen dimension
        new_vals = jnp.take(self.vals, closest_idx, axis=dim)

        # Remove the sliced dimension from bounds
        new_bounds = [b for i, b in enumerate(self.bounds) if i != dim]

        # If the resulting array is still multi-dimensional, keep it as is
        return GridDiscretisationND(new_bounds, new_vals)

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
    def linspaces(self):
        axes = [
            jnp.linspace(start, end, num)
            for (start, end), num in zip(self.bounds, self.vals.shape)
        ]
        return tuple(axes)

    @property
    def coordinate_arrays(self):
        return tuple(jnp.meshgrid(*self.linspaces, indexing="ij"))

    def locate_closest(self, point: Point) -> tuple[Int, ...]:
        """
        Locate the index of the closest grid point.

        >>> data = GridDiscretisationND.discretise_fn(bounds=[(0,1),(0,1)], n_points=[2,2], fn=lambda x: x[0]+x[1])
        >>> x_idx, y_idx = data.locate_closest(jnp.array([0.1, 0.9]))
        >>> print(f"({x_idx}, {y_idx})")
        (0, 1)
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


def grid_sample(
    data: GridDiscretisationND,
    num_indices_per_dim: tuple[int, ...],
) -> PointCloud:
    """
    Grid sample a GridDiscretisationND object.

    >>> import jax.numpy as jnp
    >>> from pinnlib.data_utils import GridDiscretisationND, PointCloud
    >>> data = GridDiscretisationND.discretise_fn(bounds=[(0,1),(0,1)], n_points=[2,2], fn=lambda x: 0.0)
    >>> pc = grid_sample(data, num_indices_per_dim=(2, 2))
    >>> isinstance(pc, PointCloud)
    True
    >>> pc.coords.shape
    (4, 2)
    >>> pc.vals.shape
    (4,)
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
    get_at = tuple(get_at)

    points = jnp.stack([c[get_at].ravel() for c in data.coordinate_arrays], axis=-1)
    vals = data.vals[get_at].ravel()

    return PointCloud(coords=points, vals=vals)


def random_sample(
    data: GridDiscretisationND,
    num_points: int,
    *,
    key: PRNGKeyArray,
) -> PointCloud:
    """
    RaNDomly sample points from a GridDiscretisationND object.

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from pinnlib.data_utils import GridDiscretisationND, PointCloud
    >>> key = jax.random.PRNGKey(0)
    >>> data = GridDiscretisationND.discretise_fn(bounds=[(0,1),(0,1)], n_points=[2,2], fn=lambda x: 0.0)
    >>> pc = random_sample(data, num_points=2, key=key)
    >>> isinstance(pc, PointCloud)
    True
    >>> pc.coords.shape
    (2, 2)
    >>> pc.vals.shape
    (2,)
    """
    vals_flat = data.vals.ravel()
    coords_flat = [c.ravel() for c in data.coordinate_arrays]
    indices = jrandom.choice(
        key, jnp.arange(vals_flat.size), shape=(num_points,), replace=False
    )

    points = jnp.stack([c[indices] for c in coords_flat], axis=-1)
    vals = vals_flat[indices]

    return PointCloud(coords=points, vals=vals)


def full_data(
    data: GridDiscretisationND,
) -> PointCloud:
    """
    Return all points from a GridDiscretisationND object as a PointCloud.

    >>> import jax.numpy as jnp
    >>> from pinnlib.data_utils import GridDiscretisationND, PointCloud
    >>> data = GridDiscretisationND.discretise_fn(bounds=[(0,1),(0,1)], n_points=[2,2], fn=lambda x: 0.0)
    >>> pc = full_data(data)
    >>> isinstance(pc, PointCloud)
    True
    >>> pc.coords.shape
    (4, 2)
    >>> pc.vals.shape
    (4,)
    """
    vals_flat = data.vals.ravel()
    coords_flat = [c.ravel() for c in data.coordinate_arrays]
    points = jnp.stack(coords_flat, axis=-1)

    return PointCloud(coords=points, vals=vals_flat)
