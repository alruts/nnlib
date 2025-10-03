from functools import partial
from typing import Sequence, Type

import jax.numpy as jnp
from jax import local_device_count, pmap
from jax import random as jrandom
from jaxtyping import PRNGKeyArray
from test_bench.discretize import Point
from torch.utils.data import Dataset

from nnlib.dataload.data_structures import PointCloud


class BaseSampler(Dataset):
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


class AbstractUniformSampler(BaseSampler):
    """
    Uniform sampler for a an abstract rectangular domain

    Example
        >>> from pprint import pprint
        >>> bounds = [(0, 1), (0, 1)]
        >>> sampler = AbstractUniformSampler(bounds, 3)
        >>> batch_points = sampler[0]
        >>> pprint(batch_points)
        Array([[[0.5788324 , 0.22059739],
                [0.10406339, 0.34068835],
                [0.15728986, 0.6127726 ]]], dtype=float32)
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
        batch = jrandom.uniform(
            key,
            shape=(self.batch_size, len(self.bounds)),
            minval=jnp.array(mins),
            maxval=jnp.array(maxs),
        )
        return batch


class DataPointSampler(BaseSampler):
    """
    Randomly samples batches from a PointCloud.

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
    >>> batch = sampler[0]
    >>> isinstance(batch, PointCloud)
    True
    >>> batch.coords.shape == (sampler.num_devices, 2, 3)
    True
    >>> batch.vals.shape == (sampler.num_devices, 2)
    True
    >>> all(isinstance(x, jnp.ndarray) for x in batch.coords)
    True
    >>> all(isinstance(x, float) or isinstance(x, jnp.ndarray) for x in batch.vals)
    True
    """

    def __init__(self, batch_size: int, point_cloud: PointCloud, *, key: PRNGKeyArray):
        super().__init__(batch_size=batch_size, key=key)
        self.point_cloud = point_cloud

    @partial(pmap, static_broadcasted_argnums=(0,))
    def gen_data(self, *, key: PRNGKeyArray):
        idx = jrandom.randint(
            key,
            shape=(self.batch_size,),
            minval=0,
            maxval=self.point_cloud.coords.shape[0],
        )
        coords = self.point_cloud.coords[idx]
        vals = self.point_cloud.vals[idx]
        return PointCloud(coords, vals)
