import jax.numpy as jnp
import jax.random as jrandom
import trimesh

from nnlib.data_utils import MeshSampler

# --- Create a simple triangular mesh ---
vertices = jnp.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)
faces = jnp.array([[0, 1, 2]])
mesh = trimesh.load_mesh("data/test-mesh.stl")
# --- Initialize the sampler ---
sampler = MeshSampler(mesh, batch_size=1024, key=jrandom.PRNGKey(0))
infinite_loader = iter(sampler)


for step, ((x, y, z), (nx, ny, nz)) in zip(range(5), infinite_loader):
    # Flatten the per-device batches into a single list
    pts = jnp.stack([x, y, z], axis=-1).reshape(-1, 3)
    normals = jnp.stack([nx, ny, nz], axis=-1).reshape(-1, 3)

    # Convert to NumPy for trimesh
    pts = jnp.array(pts)
    normals = jnp.array(normals)

    # --- Visualization parameters ---
    point_scale = mesh.scale / 100 if mesh.scale > 0 else 0.01
    normal_length = mesh.scale * 0.05 if mesh.scale > 0 else 0.05

    spheres = []
    lines = []

    # --- Create spheres and normal lines ---
    for pt, n in zip(pts, normals):
        # Sphere for sampled point
        s = trimesh.creation.icosphere(radius=point_scale)
        s.apply_translation(jnp.array(pt))
        s.visual.vertex_colors = [255, 0, 0, 255]  # red
        spheres.append(s)

        # Line for normal
        vec = jnp.stack([pt, pt + n * normal_length], axis=0)
        path = trimesh.load_path(vec.reshape((-1, 2, 3)))
        lines.append(path)

    # --- Combine mesh, spheres, and lines into a scene ---
    scene = trimesh.Scene([mesh, *spheres, *lines])
    scene.show(smooth=False)
