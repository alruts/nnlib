from functools import partial
from typing import NamedTuple, Sequence, Type

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, local_device_count, pmap
from jax import random as jrandom
from jaxtyping import PRNGKeyArray
from test_bench.discretize import Point, Point3d
from torch.utils.data import DataLoader, Dataset, default_collate


def numpy_collate(batch):
    return jax.tree.map(np.asarray, default_collate(batch))


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
        >>> from collections import namedtuple
        >>> Point2d = namedtuple("Point2d", ["x", "y"])
        >>> bounds = [(0, 1), (0, 1)]
        >>> sampler = AbstractUniformSampler(bounds, Point2d, 3)
        >>> batch_points = sampler[0]
        >>> pprint(batch_points)
        [Point2d(x=Array([0.5788324], dtype=float32), y=Array([0.22059739], dtype=float32)),
         Point2d(x=Array([0.10406339], dtype=float32), y=Array([0.34068835], dtype=float32)),
         Point2d(x=Array([0.15728986], dtype=float32), y=Array([0.6127726], dtype=float32))]
    """

    def __init__(
        self,
        bounds: Sequence[tuple[float, float]],
        structure: Type[Point],
        batch_size: int,
        *,
        key: PRNGKeyArray = jrandom.PRNGKey(0),
    ):
        super().__init__(batch_size, key=key)
        self.bounds = bounds
        self.structure = structure

    @partial(pmap, static_broadcasted_argnums=(0,))
    def gen_data(self, key: PRNGKeyArray):
        mins, maxs = zip(*self.bounds)
        batch = jrandom.uniform(
            key,
            shape=(self.batch_size, len(self.bounds)),
            minval=jnp.array(mins),
            maxval=jnp.array(maxs),
        )
        # Convert each row to a Point named tuple
        points = [self.structure(*row) for row in batch]
        return points


class BaseDataSampler(Dataset):
    """Base class for coordinate sampling."""

    def __init__(self, batch_size: int, data: list, *, key=jrandom.PRNGKey(0)):
        self.batch_size = batch_size
        self.data = data
        self.key = key
        self.num_devices = local_device_count()

    @classmethod
    def from_file(cls, batch_size: int, path_to_data, *, key=jrandom.PRNGKey(0)):
        import pickle

        with open(path_to_data, "rb") as f:
            data = pickle.load(f)
        return cls(batch_size=batch_size, data=data, key=key)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        self.key, subkey = jrandom.split(self.key)
        keys = jrandom.split(subkey, self.num_devices)
        batch = self.gen_data(key=keys)
        return batch

    def gen_data(self, *, key: PRNGKeyArray):
        raise NotImplementedError


class DataPointSampler(BaseDataSampler):
    """
    A simple data sampler that randomly samples batches of 3D points and associated values.

    >>> import jax
    >>> from jax import random
    >>> import jax.numpy as jnp
    >>> key = random.PRNGKey(42)
    >>> data = [(Point3d(1.0, 2.0, 3.0), 10.0),
    ...         (Point3d(4.0, 5.0, 6.0), 20.0),
    ...         (Point3d(7.0, 8.0, 9.0), 30.0)]
    >>> sampler = DataPointSampler(batch_size=2, data=data, key=key)
    >>> # Generate a batch (shape depends on number of devices)
    >>> batch = sampler[0]
    >>> isinstance(batch[0][0], Point3d)
    True
    >>> isinstance(batch[0][1], jnp.ndarray) or isinstance(batch[0][1], float)
    True
    """

    def __init__(
        self,
        batch_size: int,
        data: list[tuple[Point, float]],
        *,
        key=jrandom.PRNGKey(0),
    ):
        super().__init__(batch_size, data, key=key)
        points, vals = zip(*self.data)
        self.points = jnp.array(points)
        self.vals = jnp.array(vals)
        self.point_structure: Type[Point] = type(self.data[0][0])

    @partial(jax.pmap, static_broadcasted_argnums=(0,))
    def gen_data(self, key: jnp.ndarray):
        """
        Sample a random batch of datapoints in parallel across devices.
        Returns a list/array shaped [num_devices, batch_size, ...]
        """

        # Sample indices for this device
        idx = jrandom.randint(key, shape=(self.batch_size,), minval=0, maxval=len(self))
        points = self.points[idx]
        vals = self.vals[idx]
        return [(self.point_structure(*p), v) for p, v in zip(points, vals)]
