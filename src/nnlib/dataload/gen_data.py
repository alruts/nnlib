from typing import Callable

import jax.numpy as jnp
from jaxtyping import Float
from test_bench.discretize import Point


def point_source_time(
    r: Point,
    t: Float,
    r0: Point,
    q: Callable[[Float], Float],
    c: float = 343.0,
):
    """
    Compute the time-domain acoustic pressure from a point source.

    Parameters:
    r: observation point (1, 2, or 3D)
    r0: source point (1, 2, or 3D)
    t: time array
    q: source function q(t)
    c: speed of sound (default 343 m/s)

    Returns:
    p : array, pressure at each time t
    """
    r_arr = jnp.array(r)
    r0_arr = jnp.array(r0)
    R = jnp.linalg.norm(r_arr - r0_arr)  # distance from source
    scale = 1 / (4 * jnp.pi * R)  # spherical spreading

    # retarded time
    t_retarded = t - R / c

    # pressure
    p = scale * q(t_retarded)
    return p


# # Example
# import jax
# def q(t: Float) -> jax.Array:
#     return jnp.sin(100 * (t - 0.1)) * jnp.exp(-1 * t)
#
#
# r0a = Point2d(0.0, 0.0)
# r0b = Point2d(0.05, 0.05)
#
#
# @jax.jit
# def fn(pt):
#     return point_source_time(
#         r=Point2d(pt[0], pt[1]),
#         r0=r0a,
#         t=pt[2],
#         q=q,
#     ) + point_source_time(
#         r=Point2d(pt[0], pt[1]),
#         r0=r0b,
#         t=pt[2],
#         q=q,
#     )
#
#
# d = SpatialDiscretisationND.discretise_fn(
#     [(-0.1, 0.1), (-0.1, 0.1), (0.0, 0.5)],
#     [1000, 1000, 1000],
#     fn=fn,
# )
#
# X, Y, T = d.coordinate_arrays
# vals = d.vals
#
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
#
# # Assume:
# # X, Y, T have shape (Nx, Ny, Nt)
# # vals has shape (Nx, Ny, Nt)
#
# fig, ax = plt.subplots(figsize=(6, 5))
#
# # Initial plot (first time step)
# pcm = ax.pcolormesh(
#     X[:, :, 0], Y[:, :, 0], vals[:, :, 0], shading="auto", cmap="viridis"
# )
# cbar = fig.colorbar(pcm, ax=ax, label="Value")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_title(f"Time = {T[0, 0, 0]:.3f} s")
#
#
# # Update function
# def update(frame):
#     pcm.set_array(vals[:, :, frame].ravel())  # flatten for pcolormesh
#     ax.set_title(f"Time = {T[0, 0, frame]:.3f} s")
#     return [pcm]
#
#
# anim = FuncAnimation(fig, update, frames=X.shape[2], interval=1, blit=False)
#
# plt.show()
