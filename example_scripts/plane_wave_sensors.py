import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 1. Simulation parameters
# -----------------------------
Nx, Ny = 64, 64  # grid size
dx, dy = 0.01, 0.01  # spatial resolution [m]
c = 343  # speed of sound [m/s]
f = 1000  # frequency [Hz]
omega = 2 * np.pi * f
k = omega / c

# 2D grid
x = np.arange(Nx) * dx
y = np.arange(Ny) * dy
X, Y = np.meshgrid(x, y, indexing="ij")
N = Nx * Ny

# -----------------------------
# 2. Simulate a 2D sound field
# -----------------------------
thetas = np.random.random(5) * 2 * np.pi
breakpoint()
amplitudes = np.random.random(5) * 1j + np.random.random(5)
p = (
    sum(
        [
            amp * np.exp(1j * k * (X * np.cos(th) + Y * np.sin(th)))
            for th, amp in zip(thetas, amplitudes)
        ]
    )
    / 5
)
plt.figure()
plt.imshow(np.real(p), origin="lower", extent=[0, Nx * dx, 0, Ny * dy])
plt.title("Simulated 2D Sound Field (Real part)")
plt.colorbar()
plt.show()

# -----------------------------
# 3. Plane wave dictionary
# -----------------------------
n_angles = 64
angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
Phi = np.zeros((N, n_angles), dtype=complex)

for j, theta in enumerate(angles):
    Phi[:, j] = np.exp(1j * k * (X.ravel() * np.cos(theta) + Y.ravel() * np.sin(theta)))


# -----------------------------
# 4. Greedy sensor selection (D-optimality)
# -----------------------------
def greedy_sensor_selection(Phi, n_sensors):
    N, M = Phi.shape
    selected = []
    remaining = list(range(N))

    for _ in range(n_sensors):
        best_idx = None
        best_det = -np.inf

        for idx in remaining:
            trial = selected + [idx]
            Phi_sub = Phi[trial, :]
            det_val = np.linalg.det(Phi_sub.conj().T @ Phi_sub + 1e-5 * np.eye(M))
            if det_val > best_det:
                best_det = det_val
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


n_sensors = 30
sensor_indices = greedy_sensor_selection(Phi, n_sensors)

# -----------------------------
# 5. Visualize optimal sensor positions
# -----------------------------
sensor_x = X.ravel()[sensor_indices]
sensor_y = Y.ravel()[sensor_indices]

plt.figure()
plt.imshow(np.real(p), origin="lower", extent=[0, Nx * dx, 0, Ny * dy], alpha=0.5)
plt.scatter(sensor_x, sensor_y, color="red", s=50, label="Sensors")
plt.title(f"Optimal Sensor Positions ({n_sensors} sensors)")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()
plt.show()

# -----------------------------
# 6. Reconstruction from selected sensors
# -----------------------------
Phi_s = Phi[sensor_indices, :]
p_s = p.ravel()[sensor_indices]

# Least-squares reconstruction of plane wave coefficients
alpha = np.linalg.lstsq(Phi_s, p_s, rcond=None)[0]

# Reconstruct full sound field
p_rec = Phi @ alpha
p_rec = p_rec.reshape(Nx, Ny)

plt.figure()
plt.imshow(np.real(p_rec), origin="lower", extent=[0, Nx * dx, 0, Ny * dy])
plt.title("Reconstructed Sound Field (Real part)")
plt.colorbar()
plt.show()

# -----------------------------
# 7. Reconstruction error
# -----------------------------
error = np.abs(p - p_rec)

plt.figure()
plt.imshow(error, origin="lower", extent=[0, Nx * dx, 0, Ny * dy], cmap="hot")
plt.title("Reconstruction Error (Magnitude)")
plt.colorbar()
plt.show()

# Print mean and max error
print("Mean reconstruction error:", np.mean(error))
print("Max reconstruction error:", np.max(error))
