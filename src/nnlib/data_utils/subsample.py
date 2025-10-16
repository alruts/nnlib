import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray

from nnlib.data_utils.data_structures import GridDiscretisationND, PointCloud


def grid_sample(
    data: GridDiscretisationND,
    num_indices_per_dim: tuple[int, ...],
) -> PointCloud:
    """
    Grid sample a GridDiscretisationND object.

    >>> import jax.numpy as jnp
    >>> from nnlib.data_utils.data_structures import GridDiscretisationND, PointCloud
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
    >>> from nnlib.data_utils.data_structures import GridDiscretisationND, PointCloud
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
    >>> from nnlib.data_utils.data_structures import GridDiscretisationND, PointCloud
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
