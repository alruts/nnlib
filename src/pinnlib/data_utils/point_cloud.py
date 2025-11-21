from typing import NamedTuple

from jaxtyping import Array, Scalar

# base data annotations
Coords = tuple[Array, ...]
Coord = tuple[Scalar, ...]
Vecs = tuple[Array, ...]
Vec = tuple[Scalar, ...]
CoordsVecs = tuple[Array, ...]
CoordVec = tuple[Scalar, ...]
Vals = Array


class PointCloud(NamedTuple):
    """
    Data structure for a point cloud where coords is a tuple of 1D arrays.
    """

    coords: Coords
    vals: Vals
