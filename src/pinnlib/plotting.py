from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
import pyvista as pv
import trimesh
from scipy.interpolate import griddata

from pinnlib.data_utils.point_cloud import PointCloud


def plot_batch(
    mesh: trimesh.Trimesh | None = None,
    pressure_batch: PointCloud | None = None,
    velocity_batch: PointCloud | None = None,
    domain_batch: tuple[jnp.ndarray, ...] | None = None,
    boundary_batch: tuple[tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...]]
    | None = None,
    gt_pressure: PointCloud | None = None,
    bounding_box: Sequence[tuple[float, float]] | None = None,
    point_size: int = 5,
    normal_scale: float = 0.05,
    gt_volume_res: int = 128,  # number of voxels per axis
):
    """
    Plot mesh, batches, and optionally a volumetric ground-truth pressure field.
    """
    plotter = pv.Plotter()

    # Mesh
    if mesh is not None:
        plotter.add_mesh(pv.wrap(mesh), color="lightgrey", opacity=0.5, label="Mesh")

    # Helper to flatten tuple of arrays (num_devices, batch_size) -> (num_points,)
    def flatten_coords(coords_tuple):
        return tuple(np.array(c).reshape(-1) for c in coords_tuple)

    # Pressure points
    if pressure_batch is not None:
        coords_np = np.stack(flatten_coords(pressure_batch.coords), axis=-1)
        plotter.add_points(
            coords_np,
            color="red",
            point_size=point_size,
            render_points_as_spheres=True,
            label="Pressure Points",
        )

    # Velocity points
    if velocity_batch is not None:
        coords_np = np.stack(flatten_coords(velocity_batch.coords), axis=-1)
        plotter.add_points(
            coords_np,
            color="blue",
            point_size=point_size,
            render_points_as_spheres=True,
            label="Velocity Points",
        )

    # Domain points
    if domain_batch is not None:
        coords_np = np.stack(flatten_coords(domain_batch), axis=-1)
        plotter.add_points(
            coords_np,
            color="green",
            point_size=max(1, point_size // 2),
            render_points_as_spheres=True,
            label="Domain Points",
        )

    # Boundary points + normals
    if boundary_batch is not None:
        coords_tuple, normals_tuple = boundary_batch
        coords_np = np.stack(flatten_coords(coords_tuple), axis=-1)
        normals_np = np.stack(flatten_coords(normals_tuple), axis=-1)

        plotter.add_points(
            coords_np,
            color="orange",
            point_size=point_size,
            render_points_as_spheres=True,
            label="Boundary Points",
        )
        plotter.add_arrows(
            coords_np,
            normals_np,
            mag=normal_scale,
            color="yellow",
            label="Boundary Normals",
        )

    # Ground-truth pressure volumetric reconstruction
    if gt_pressure is not None:
        # Flatten coordinates and values
        x, y, z = flatten_coords(gt_pressure.coords)
        p = np.abs(gt_pressure.vals.reshape(-1))  # pressure magnitude

        plotter.add_points(
            np.stack([x, y, z], axis=-1),
            scalars=p,
            style="points_gaussian",
            opacity=0.01,
            point_size=10,
        )

    plotter.add_legend()
    plotter.show_axes()
    plotter.show()
