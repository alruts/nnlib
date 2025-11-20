from collections.abc import Callable, Sequence
from functools import reduce
from typing import NamedTuple

import jax
from jax import numpy as jnp
from jaxtyping import Array

# base data annotations
Coords = tuple[Array, ...]  # tuple of 1D arrays
Vals = Array


class PointCloud(NamedTuple):
    """
    Data structure for a point cloud where coords is a tuple of 1D arrays.
    """

    coords: Coords
    vals: Vals
