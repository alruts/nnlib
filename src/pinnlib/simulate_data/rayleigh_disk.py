import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Complex

from pinnlib import default_medium_density, default_wave_speed
from pinnlib.data_utils import GridDiscretisationND

jax.config.update("jax_enable_x64", True)


def plot_discretization(
    data: GridDiscretisationND, cbar_label="Value", clim=None, **kwds
):
    """
    Plots GridDiscretisationND object in 1D, 2D, or 3D using scatter with colorbar.

    Parameters:
        x: GridDiscretisationND
            Must have attributes `coordinate_arrays` (tuple of ndarrays) and `vals` (color values)
        cmap: str
            Matplotlib colormap
    """

    ndim = data.ndim  # number of dimensions
    if ndim == 1:
        fig, ax = plt.subplots()
        sc = ax.scatter(*data.coordinate_arrays, data.vals, c=data.vals, **kwds)
        ax.set_xlabel("X")
        ax.set_ylabel("Value")
    elif ndim == 2:
        fig, ax = plt.subplots()
        sc = ax.scatter(
            data.coordinate_arrays[0], data.coordinate_arrays[1], c=data.vals, **kwds
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    elif ndim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(*data.coordinate_arrays, c=data.vals, **kwds)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    else:
        raise ValueError("Only 1D, 2D, or 3D data supported.")

    if clim:
        sc.set_clim(clim)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_aspect("equal", "box")

    return fig, ax


class RayleighDiskInBaffle(eqx.Module):
    """Rayleigh integral model for a circular piston with finite surface impedance."""

    # Core physical parameters
    medium_density: float
    wave_speed: float
    frequency: float
    disk_radius: float
    surface_impedance: float

    # Discretized geometry and associated fields
    disk_surface_points: Array
    disk_velocity: Array | float
    surface_element_areas: Array

    def __init__(
        self,
        medium_density: float,
        wave_speed: float,
        frequency: float,
        disk_radius: float,
        surface_impedance: float,
        piston_velocity: float,
        points_per_wavelength: int = 6,
    ):
        """Initialize the discretized circular disk and its velocity field."""

        # Store core parameters
        self.medium_density = medium_density
        self.wave_speed = wave_speed
        self.frequency = frequency
        self.disk_radius = disk_radius
        self.surface_impedance = surface_impedance

        # Derived acoustic parameters
        angular_frequency = 2 * jnp.pi * frequency
        wavenumber = angular_frequency / wave_speed
        wavelength = 2 * jnp.pi / wavenumber

        # Disk discretization
        dx = wavelength / points_per_wavelength
        num_radial = max(2, int(jnp.ceil(disk_radius / dx)))
        num_angular = max(8, int(jnp.ceil(2 * jnp.pi * disk_radius / dx)))

        # Radial and angular sample positions
        radial_positions = (
            jnp.sqrt((jnp.arange(num_radial) + 0.5) / num_radial) * disk_radius
        )
        angular_positions = jnp.linspace(0, 2 * jnp.pi, num_angular, endpoint=False)

        radial_grid, angular_grid = jnp.meshgrid(
            radial_positions, angular_positions, indexing="ij"
        )

        x_positions = radial_grid * jnp.cos(angular_grid)
        y_positions = radial_grid * jnp.sin(angular_grid)
        z_positions = jnp.zeros_like(x_positions)

        disk_points = jnp.stack([x_positions, y_positions, z_positions], axis=-1)
        self.disk_surface_points = disk_points.reshape(-1, 3)

        # Local surface element areas (radius-weighted)
        dr = disk_radius / num_radial
        dtheta = 2 * jnp.pi / num_angular
        local_areas = radial_grid * dr * dtheta
        self.surface_element_areas = local_areas.reshape(-1)

        # Edge tapering: velocity smoothly goes to zero at the edge
        radial_flat = radial_grid.reshape(-1)
        taper_profile = (1 - (radial_flat / disk_radius) ** 2) ** 0.5
        velocity_flat = piston_velocity * taper_profile

        # Include surface impedance: v_eff = v / (1 + Z_medium / Z_surface)
        effective_velocity = velocity_flat / (
            1 + (medium_density * wave_speed) / surface_impedance
        )
        self.disk_velocity = effective_velocity

    @eqx.filter_jit
    def __call__(self, observation_point: jnp.ndarray) -> Complex:
        """Compute complex acoustic pressure at a single observation point."""
        angular_frequency = 2 * jnp.pi * self.frequency
        wavenumber = angular_frequency / self.wave_speed

        displacement_vectors = observation_point - self.disk_surface_points
        distances = jnp.linalg.norm(displacement_vectors, axis=-1)

        effective_velocity = self.disk_velocity / (
            1 + (self.medium_density * self.wave_speed) / self.surface_impedance
        )

        contribution = (
            effective_velocity * jnp.exp(-1j * wavenumber * distances) / distances
        )

        pressure = (
            jnp.sum(contribution * self.surface_element_areas)
            * 1j
            * self.medium_density
            * angular_frequency
            / (2 * jnp.pi)
        )
        return pressure


if __name__ == "__main__":
    #
    # # Define Rayleigh integral problem
    #

    disk_model = RayleighDiskInBaffle(
        medium_density=default_medium_density(),
        wave_speed=default_wave_speed(),
        frequency=4000.0,
        disk_radius=0.0875,
        surface_impedance=2.0,
        piston_velocity=1.0,
        points_per_wavelength=7,
    )

    # Derived acoustic quantities
    angular_frequency = 2 * jnp.pi * disk_model.frequency
    wavenumber = angular_frequency / disk_model.wave_speed
    wavelength = 2 * jnp.pi / wavenumber

    # Observation grid
    grid_extent = 1.5 * wavelength
    points_per_wavelength_obs = 6
    dx_obs = wavelength / points_per_wavelength_obs

    n_points_x = int(2 * grid_extent / dx_obs)
    n_points_y = int(2 * grid_extent / dx_obs)
    n_points_z = int((2 * wavelength) / dx_obs)

    domain = GridDiscretisationND.discretise_fn(
        bounds=[
            (-grid_extent, grid_extent),
            (-grid_extent, grid_extent),
            (0.01, wavelength),
        ],
        fn=disk_model,
        n_points=[n_points_x, n_points_y, n_points_z],
    )

    #
    # # Plot magnitude and phase
    #

    for args in (["Magnitude", jnp.abs], ["Phase", jnp.angle]):
        label, tx = args
        fig, ax = plot_discretization(
            domain.transform(tx), cbar_label=label, alpha=0.1, cmap="seismic"
        )

        # Plot the disk border
        theta = jnp.linspace(0, 2 * jnp.pi, 200)  # parametric angle for circle
        x_circle = disk_model.disk_radius * jnp.cos(theta)
        y_circle = disk_model.disk_radius * jnp.sin(theta)
        ax.plot(
            x_circle, y_circle, "k--", linewidth=2, label="Piston Border", alpha=0.4
        )

        ax.legend()

        plt.tight_layout()
        plt.show()

    #
    # # Compute Helmholtz residual in whole domain to ensure clean signal for PINN
    #

    def residual_fn(primal, model):
        def second_derivative_jvp(p, argnum=0):
            n = jnp.zeros_like(p)
            n = n.at[argnum].set(1.0)

            dx_fn = lambda p: jax.jvp(model, (p,), (n,))[1]
            dxx_fn = lambda p: jax.jvp(dx_fn, (p,), (n,))[1]
            return dxx_fn(p)

        # make sure the function is close to hh residual
        dxx = second_derivative_jvp(primal, 0)
        dyy = second_derivative_jvp(primal, 1)
        dzz = second_derivative_jvp(primal, 2)
        p = disk_model(primal)
        return dxx + dyy + dzz + wavenumber**2 * p

    residual = GridDiscretisationND.discretise_fn(
        bounds=[
            (-grid_extent, grid_extent),
            (-grid_extent, grid_extent),
            (0.01, wavelength),
        ],
        fn=lambda p: residual_fn(p, disk_model),
        n_points=[n_points_x, n_points_y, n_points_z],
    )

    print(jnp.mean(jnp.abs(residual.vals)))
    print(jnp.max(jnp.abs(residual.vals)))
