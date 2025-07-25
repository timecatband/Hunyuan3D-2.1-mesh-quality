# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import trimesh
import xatlas


def mesh_uv_wrap(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    options = xatlas.PackOptions()
    options.padding = 10

    if len(mesh.faces) > 500000000:
        raise ValueError("The mesh has more than 500,000,000 faces, which is not supported.")

    atlas = xatlas.Atlas()
    atlas.add_mesh(mesh.vertices, mesh.faces, normals=mesh.vertex_normals)
    atlas.generate(pack_options=options)
    vmapping, indices, uvs = atlas[0]

    mesh.vertices = mesh.vertices[vmapping]
    mesh.faces = indices
    mesh.visual.uv = uvs

    return mesh
