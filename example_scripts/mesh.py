import numpy as np
import trimesh

# Load mesh
mesh: trimesh.Trimesh = trimesh.load_mesh("~/Documents/disk.stl")


# Function to sample a point uniformly on a triangle
def sample_point_on_triangle(tri):
    u, v = np.random.rand(2)
    if u + v > 1:
        u, v = 1 - u, 1 - v
    w = 1 - u - v
    return u * tri[0] + v * tri[1] + w * tri[2]


# Parameters
n_points = 512  # number of points to sample
point_scale = mesh.scale / 100
normal_length = mesh.scale * 0.05

# Lists to hold spheres and normal lines
spheres = []
lines = []

for _ in range(n_points):
    # Pick a random face index
    face_index = np.random.randint(len(mesh.faces))
    triangle = mesh.triangles[face_index]
    normal = mesh.face_normals[face_index]

    # Sample point on the triangle
    pt = sample_point_on_triangle(triangle)

    # Sphere for sampled point
    s = trimesh.creation.icosphere(radius=point_scale)
    s.apply_translation(pt)
    s.visual.vertex_colors = [255, 0, 0, 255]  # red
    spheres.append(s)

    # Line for normal
    vec = np.array([pt, pt + normal * normal_length])
    path = trimesh.load_path(vec.reshape((-1, 2, 3)))
    lines.append(path)

# Combine mesh, spheres, and lines into a scene
scene = trimesh.Scene([mesh, *spheres, *lines])
scene.show(smooth=False)
