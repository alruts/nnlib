from functools import partial
from typing import Sequence

import jax.numpy as jnp
from jax import local_device_count, pmap, random
from jaxtyping import PRNGKeyArray
from test_bench.discretize import Point
from torch.utils.data import Dataset


class BaseSampler(Dataset):
    def __init__(self, batch_size, rng_key=random.PRNGKey(0)):
        self.batch_size = batch_size
        self.key = rng_key
        self.num_devices = local_device_count()

    def __getitem__(self, index: int):
        self.key, subkey = random.split(self.key)
        keys = random.split(subkey, self.num_devices)
        batch = self.gen_data(keys)
        return batch

    def gen_data(self, key: PRNGKeyArray):
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
        structure: Point,
        batch_size: int,
        rng_key: PRNGKeyArray = random.PRNGKey(0),
    ):
        super().__init__(batch_size, rng_key)
        self.bounds = bounds
        self.structure = structure

    @partial(pmap, static_broadcasted_argnums=(0,))
    def gen_data(self, key: PRNGKeyArray):
        mins, maxs = zip(*self.bounds)
        batch = random.uniform(
            key,
            shape=(self.batch_size, len(self.bounds)),
            minval=jnp.array(mins),
            maxval=jnp.array(maxs),
        )
        # Convert each row to a Point named tuple
        points = [self.structure(*row) for row in batch]  #! to-do: fix type error here
        return points
