import meshio
import numpy as np
import trimesh


def load_mesh_layers(mesh_file):
    """
    Load a mesh and split it into layers based on GMSH physical groups.

    Parameters:
        mesh_file (str): Path to the .msh file.

    Returns:
        dict: A dictionary mapping physical group names to trimesh.Trimesh objects.
    """
    mesh = meshio.read(mesh_file)

    # Extract triangles and tags
    triangles = mesh.cells_dict.get("triangle")
    if triangles is None:
        raise ValueError("Mesh contains no triangle cells.")

    tags = mesh.cell_data_dict["gmsh:physical"]["triangle"]

    # Map tag ID to layer name
    tag_to_name = {v[0]: k for k, v in mesh.field_data.items()}

    # Build layers dictionary
    layers = {
        tag_to_name[tag_id]: trimesh.Trimesh(
            vertices=mesh.points, faces=triangles[tags == tag_id], process=False
        )
        for tag_id in np.unique(tags)
    }

    return layers
