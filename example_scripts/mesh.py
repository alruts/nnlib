import numpy as np
import pyvista as pv
import trimesh

# -----------------------------
# 1. Load mesh using trimesh
# -----------------------------
mesh_tm = trimesh.load_mesh("data/disk_01.stl")  # your input mesh
scale = mesh_tm.scale


# -----------------------------
# 2. Sample points on mesh
# -----------------------------
def sample_point_on_triangle(tri):
    u, v = np.random.rand(2)
    if u + v > 1:
        u, v = 1 - u, 1 - v
    w = 1 - u - v
    return u * tri[0] + v * tri[1] + w * tri[2]


n_points = 512
pts = []
nrms = []

for _ in range(n_points):
    idx = np.random.randint(len(mesh_tm.faces))
    tri = mesh_tm.triangles[idx]
    nrm = mesh_tm.face_normals[idx]

    p = sample_point_on_triangle(tri)
    pts.append(p)
    nrms.append(nrm)

pts = np.array(pts)
nrms = np.array(nrms)


# -----------------------------
# 3. Convert to PyVista
# -----------------------------
mesh_pv = pv.wrap(mesh_tm)


# -----------------------------
# 4. Build visualization
# -----------------------------
plotter = pv.Plotter(shape=(1, 2), border="white")

# --------------------------------------
# LEFT PANE — raw mesh only
# --------------------------------------
plotter.subplot(0, 0)
plotter.add_text("Raw mesh", font_size=12)
plotter.add_mesh(mesh_pv, color="lightgray")
plotter.add_axes()
plotter.show_bounds(grid="front")


# --------------------------------------
# RIGHT PANE — mesh + points + normals + scale helpers
# --------------------------------------
plotter.subplot(0, 1)
plotter.add_text("Mesh + points + normals + scale", font_size=12)

# mesh
plotter.add_mesh(mesh_pv, color="lightgray", opacity=0.6)

# sampled points
plotter.add_points(pts, color="red", point_size=8)

# normal arrow glyphs
arrow_length = scale * 0.05
plotter.add_arrows(pts, nrms * arrow_length, color="blue")

# axes helper
plotter.add_axes()

# bounding box
plotter.show_bounds(grid="back", location="outer", ticks="outside")

# optional ground grid
grid_size = scale
grid = pv.Plane(i_size=grid_size, j_size=grid_size)
plotter.add_mesh(grid, color="lightgray", opacity=0.2)


# -----------------------------
# 5. Show full scene
# -----------------------------
plotter.show()
