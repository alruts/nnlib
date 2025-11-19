from collections.abc import Sequence
from functools import partial

import jax.numpy as jnp
import trimesh
from jax import Array, local_device_count, pmap, vmap
from jax import random as jrandom
from jaxtyping import PRNGKeyArray

from pinnlib.data_utils.point_cloud import PointCloud
from pinnlib.misc import default_floating_dtype


class BaseGenerator:
    """Base class for training point generators."""

    def __init__(self, batch_size, *, key=jrandom.PRNGKey(0)):
        self.batch_size = batch_size
        self.key = key
        self.num_devices = local_device_count()

    def __getitem__(self, index: int):
        self.key, subkey = jrandom.split(self.key)
        keys = jrandom.split(subkey, self.num_devices)
        batch = self.gen_data(key=keys)
        return batch

    def gen_data(self, *, key: PRNGKeyArray):
        raise NotImplementedError


class UniformGenerator(BaseGenerator):
    """
    Sample from a rectangular Uniform distribution

    >>> bounds = [(0, 1), (0, 1)]
    >>> Generator = UniformGenerator(bounds, 2)
    >>> x, y = Generator[0]
    >>> x.shape == (Generator.num_devices, 2)
    True
    >>> y.shape == (Generator.num_devices, 2)
    True
    >>> all(isinstance(arr, jnp.ndarray) for arr in [x, y])
    True
    """

    def __init__(
        self,
        bounds: Sequence[tuple[float, float]],
        batch_size: int,
        *,
        key: PRNGKeyArray = jrandom.PRNGKey(0),
    ):
        super().__init__(batch_size, key=key)
        self.bounds = bounds

    @partial(pmap, static_broadcasted_argnums=(0,))
    def gen_data(self, *, key: PRNGKeyArray):
        mins, maxs = zip(*self.bounds)
        coords = jrandom.uniform(
            key,
            shape=(self.batch_size, len(self.bounds)),
            minval=jnp.array(mins),
            maxval=jnp.array(maxs),
        )

        return tuple(coords.T)


class MeshGenerator(BaseGenerator):
    """
    Uniform Generator for a mesh.

    This Generator uniformly samples points on the surface of a given
    `trimesh.Trimesh` object using barycentric coordinates.

    >>> # Create a simple triangular mesh (a single triangle)
    >>> vertices = jnp.array([
    ...     [0.0, 0.0, 0.0],
    ...     [1.0, 0.0, 0.0],
    ...     [1.0, 1.0, 0.0],
    ...     [1.0, 1.0, 1.0],
    ... ])
    >>> faces = jnp.array([
    ...     [0, 1, 2],
    ...     [0, 1, 3],
    ...     [1, 2, 3],
    ...     [2, 0, 3],
    ... ])
    >>> mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    >>>
    >>> Generator = MeshGenerator(mesh, batch_size=2, key=jrandom.PRNGKey(0))
    >>> infinite_boundary_loader = iter(Generator)
    >>> (x, y, z), (nx, ny, nz) = next(infinite_boundary_loader)
    >>>
    >>> assert all(arr.shape == (Generator.num_devices, 2) for arr in [x, y, z, nx, ny, nz])
    >>> assert all(isinstance(arr, jnp.ndarray) for arr in [x, y, z, nx, ny, nz])
    """

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        batch_size: int,
        *,
        key: PRNGKeyArray = jrandom.PRNGKey(0),
    ):
        super().__init__(batch_size, key=key)
        self.mesh: trimesh.Trimesh = mesh
        self.triangles: Array = jnp.array(
            mesh.triangles, dtype=default_floating_dtype()
        )
        self.normals: Array = jnp.array(
            mesh.face_normals, dtype=default_floating_dtype()
        )

    @partial(pmap, static_broadcasted_argnums=(0,))
    def gen_data(self, *, key: PRNGKeyArray):
        def sample_point_on_triangle(tri: Array, *, key: PRNGKeyArray):
            key, subkey = jrandom.split(key)
            uv = jrandom.uniform(subkey, shape=(2,))
            u, v = uv[0], uv[1]

            # Reflect if outside the triangle
            u, v = jnp.where(u + v > 1, 1 - u, u), jnp.where(u + v > 1, 1 - v, v)
            w = 1 - u - v

            # Return barycentric combination
            return u * tri[0] + v * tri[1] + w * tri[2]

        key, *tri_keys = jrandom.split(key, self.batch_size + 1)

        num_tris = len(self.mesh.triangles)
        idxs = jrandom.randint(key, (self.batch_size,), minval=0, maxval=num_tris - 1)

        # Select random triangles
        these_triangles = self.triangles[idxs]
        these_normals = self.normals[idxs]

        # sub-sample each triangle
        coords = vmap(sample_point_on_triangle)(
            these_triangles, key=jnp.array(tri_keys)
        )

        return tuple(coords.T), tuple(these_normals.T)


class DataPointGenerator(BaseGenerator):
    """
     Randomly samples batches from a PointCloud and returns separate arrays for each dimension.

    >>> import jax
    >>> from jax import random as jrandom
    >>> import jax.numpy as jnp
    >>> key = jrandom.PRNGKey(42)
    >>> data = PointCloud(
    ...     coords=jnp.array([[1.0, 2.0, 3.0],
    ...                       [4.0, 5.0, 6.0],
    ...                       [7.0, 8.0, 9.0]]),
    ...     vals=jnp.array([10.0, 20.0, 30.0])
    ... )
    >>> Generator = DataPointGenerator(batch_size=2, point_cloud=data, key=key)
    >>> infinite_dataloader = iter(Generator)
    >>> (x, y, z), vals = next(infinite_dataloader)
    >>> x.shape == (Generator.num_devices, 2)
    True
    >>> y.shape == (Generator.num_devices, 2)
    True
    >>> z.shape == (Generator.num_devices, 2)
    True
    >>> vals.shape == (Generator.num_devices, 2)
    True
    >>> all(isinstance(arr, jnp.ndarray) for arr in [x, y, z, vals])
    True
    """

    def __init__(self, batch_size: int, point_cloud: PointCloud, *, key: PRNGKeyArray):
        super().__init__(batch_size=batch_size, key=key)
        self.point_cloud = point_cloud

    def __len__(self):
        return len(self.point_cloud.vals)

    @partial(pmap, static_broadcasted_argnums=(0,))
    def gen_data(self, *, key):
        idx = jrandom.randint(
            key,
            shape=(self.batch_size,),
            minval=0,
            maxval=self.point_cloud.coords.shape[0],
        )
        coords = self.point_cloud.coords[idx]
        vals = self.point_cloud.vals[idx]

        return (tuple(coords.T), vals)
