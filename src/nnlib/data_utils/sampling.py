from functools import partial
from typing import Sequence

import jax.numpy as jnp
from jax import local_device_count, pmap
from jax import random as jrandom
from jaxtyping import PRNGKeyArray

from nnlib.data_utils.data_structures import PointCloud


class BaseSampler:
    """Base class for coordinate sampling."""

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


class UniformSampler(BaseSampler):
    """
    Uniform sampler for a an rectangular domain

    Example
        >>> bounds = [(0, 1), (0, 1)]
        >>> sampler = UniformSampler(bounds, 2)
        >>> x, y = sampler[0]
        >>> x.shape == (sampler.num_devices, 2)
        True
        >>> y.shape == (sampler.num_devices, 2)
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

        # Split coordinates into x, y, z, ...
        coord_arrays = tuple(coords[:, i] for i in range(coords.shape[1]))

        return coord_arrays


class DataPointSampler(BaseSampler):
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
    >>> sampler = DataPointSampler(batch_size=2, point_cloud=data, key=key)
    >>> infinite_dataloader = iter(sampler)
    >>> x, y, z, vals = next(infinite_dataloader)
    >>> x.shape == (sampler.num_devices, 2)
    True
    >>> y.shape == (sampler.num_devices, 2)
    True
    >>> z.shape == (sampler.num_devices, 2)
    True
    >>> vals.shape == (sampler.num_devices, 2)
    True
    >>> all(isinstance(arr, jnp.ndarray) for arr in [x, y, z, vals])
    True
    """

    def __init__(self, batch_size: int, point_cloud: PointCloud, *, key: PRNGKeyArray):
        super().__init__(batch_size=batch_size, key=key)
        self.point_cloud = point_cloud

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

        # Split coordinates into x, y, z, ...
        coord_arrays = tuple(coords[:, i] for i in range(coords.shape[1]))

        return (*coord_arrays, vals)
