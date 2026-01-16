from pathlib import Path
import numpy as np
import os
from pygltflib import (
    GLTF2,
    Scene,
    Node,
    Mesh,
    Primitive,
    Attributes,
    Buffer,
    BufferView,
    Accessor,
    Material,
    PbrMetallicRoughness,
)

# Constants
BUFFER_TARGET_ARRAY_BUFFER = 34962
BUFFER_TARGET_ELEMENT_ARRAY_BUFFER = 34963
COMPONENT_TYPE_FLOAT = 5126
COMPONENT_TYPE_UNSIGNED_SHORT = 5123
ACCESSOR_TYPE_VEC3 = "VEC3"
ACCESSOR_TYPE_SCALAR = "SCALAR"


def create_cornell_box(
    output_dir: Path
    | str = "tests/data/glTF-Sample-Assets/Models/CornellBox/glTF-Binary",
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define Materials
    mat_white = Material(
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorFactor=[0.73, 0.73, 0.73, 1.0],
            roughnessFactor=1.0,
            metallicFactor=0.0,
        ),
        name="White",
    )
    mat_red = Material(
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorFactor=[0.65, 0.05, 0.05, 1.0],
            roughnessFactor=1.0,
            metallicFactor=0.0,
        ),
        name="Red",
    )
    mat_green = Material(
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorFactor=[0.12, 0.45, 0.15, 1.0],
            roughnessFactor=1.0,
            metallicFactor=0.0,
        ),
        name="Green",
    )
    mat_light = Material(
        pbrMetallicRoughness=PbrMetallicRoughness(baseColorFactor=[0.0, 0.0, 0.0, 1.0]),
        emissiveFactor=[1.0, 1.0, 1.0],
        extensions={"KHR_materials_emissive_strength": {"emissiveStrength": 15.0}},
        name="Light",
    )

    materials = [mat_white, mat_red, mat_green, mat_light]

    # Geometry Preparation
    geo_by_mat = {
        0: [],
        1: [],
        2: [],
        3: [],
    }  # List of quads [v0,v1,v2,v3] per material index

    # Transform Helper
    def transform_quad(quad, translate, rotate_y_deg, scale):
        angle = np.radians(rotate_y_deg)
        c, s = np.cos(angle), np.sin(angle)
        rot_mat = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        new_quad = []
        for v in quad:
            v_scaled = np.array(v) * scale
            v_rot = np.dot(rot_mat, v_scaled)
            v_trans = v_rot + np.array(translate)
            new_quad.append(v_trans.tolist())
        return new_quad

    # Basic Shapes
    # Floor (White)
    geo_by_mat[0].append([(-1, 0, 1), (1, 0, 1), (1, 0, -1), (-1, 0, -1)])
    # Ceiling (White)
    geo_by_mat[0].append([(-1, 2, -1), (1, 2, -1), (1, 2, 1), (-1, 2, 1)])
    # Back Wall (White)
    geo_by_mat[0].append([(-1, 0, -1), (1, 0, -1), (1, 2, -1), (-1, 2, -1)])

    # Left Wall (Red)
    geo_by_mat[1].append([(-1, 0, 1), (-1, 0, -1), (-1, 2, -1), (-1, 2, 1)])

    # Right Wall (Green)
    geo_by_mat[2].append([(1, 0, -1), (1, 0, 1), (1, 2, 1), (1, 2, -1)])

    # Light (Light material) - small quad on ceiling
    l_size = 0.5
    geo_by_mat[3].append(
        [
            (-l_size / 2, 1.99, -l_size / 2),
            (l_size / 2, 1.99, -l_size / 2),
            (l_size / 2, 1.99, l_size / 2),
            (-l_size / 2, 1.99, l_size / 2),
        ]
    )

    # Box Base (Unit Cube centered at origin)
    box_base = [
        [(-0.5, 0, 0.5), (0.5, 0, 0.5), (0.5, 0, -0.5), (-0.5, 0, -0.5)],  # Bottom
        [(-0.5, 1, 0.5), (0.5, 1, 0.5), (0.5, 1, -0.5), (-0.5, 1, -0.5)],  # Top
        [(-0.5, 0, 0.5), (0.5, 0, 0.5), (0.5, 1, 0.5), (-0.5, 1, 0.5)],  # Front
        [(0.5, 0, 0.5), (0.5, 0, -0.5), (0.5, 1, -0.5), (0.5, 1, 0.5)],  # Right
        [(0.5, 0, -0.5), (-0.5, 0, -0.5), (-0.5, 1, -0.5), (0.5, 1, -0.5)],  # Back
        [(-0.5, 0, -0.5), (-0.5, 0, 0.5), (-0.5, 1, 0.5), (-0.5, 1, -0.5)],  # Left
    ]

    # Tall Box (White)
    # Center (0.35, 0, -0.35), Scale (0.6, 1.2, 0.6), Rot 20
    for face in box_base:
        geo_by_mat[0].append(
            transform_quad(face, (0.35, 0, -0.35), 20, np.array([0.6, 1.2, 0.6]))
        )

    # Short Box (White)
    # Center (-0.35, 0, 0.35), Scale (0.6, 0.6, 0.6), Rot -20
    for face in box_base:
        geo_by_mat[0].append(
            transform_quad(face, (-0.35, 0, 0.35), -20, np.array([0.6, 0.6, 0.6]))
        )

    # Build Buffers
    all_positions = []
    all_normals = []
    all_indices = []

    primitives_list = []

    for mat_idx, quads in geo_by_mat.items():
        if not quads:
            continue

        min_pos = [float("inf")] * 3
        max_pos = [float("-inf")] * 3

        mat_indices = []
        mat_positions = []
        mat_normals = []

        # Local vertex count for this primitive set
        local_p_count = 0

        for q in quads:
            # q is [v0, v1, v2, v3] CCW
            v0, v1, v2, v3 = (
                np.array(q[0], dtype=float),
                np.array(q[1], dtype=float),
                np.array(q[2], dtype=float),
                np.array(q[3], dtype=float),
            )

            # Compute Normal
            tr1_n = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(tr1_n)
            if norm > 0:
                tr1_n /= norm
            else:
                tr1_n = np.array([0.0, 1.0, 0.0])

            # Add vertices
            for v in [v0, v1, v2, v3]:
                mat_positions.append(v.tolist())
                mat_normals.append(tr1_n.tolist())
                # Update bounds
                for i in range(3):
                    min_pos[i] = min(min_pos[i], v[i])
                    max_pos[i] = max(max_pos[i], v[i])

            # Indices (0,1,2) and (0,2,3)
            # Relative to the start of this Quad
            mat_indices.extend(
                [
                    local_p_count + 0,
                    local_p_count + 1,
                    local_p_count + 2,
                    local_p_count + 0,
                    local_p_count + 2,
                    local_p_count + 3,
                ]
            )
            local_p_count += 4

        # Global Offset
        prim_vertex_start = len(all_positions)
        prim_index_start = len(all_indices)

        all_positions.extend(mat_positions)
        all_normals.extend(mat_normals)
        all_indices.extend([i + prim_vertex_start for i in mat_indices])

        primitives_list.append(
            {
                "mat_idx": mat_idx,
                "index_start": prim_index_start,
                "index_count": len(mat_indices),
                "vertex_start": prim_vertex_start,
                "vertex_count": len(mat_positions),
                "min": min_pos,
                "max": max_pos,
            }
        )

    # Validate
    if not all_positions:
        print("No geometry generated!")
        return

    # Convert to bytes
    points_array = np.array(all_positions, dtype=np.float32)
    normals_array = np.array(all_normals, dtype=np.float32)
    indices_array = np.array(all_indices, dtype=np.uint16)

    points_bytes = points_array.tobytes()
    normals_bytes = normals_array.tobytes()
    indices_bytes = indices_array.tobytes()

    # Calculate Offsets
    points_offset = 0
    points_len = len(points_bytes)

    # Paddle for alignment if necessary (4 bytes usually)
    # len(points_bytes) should be multiple of 12 (3 floats * 4 bytes)
    # so it remains 4-byte aligned.

    normals_offset = points_len
    normals_len = len(normals_bytes)

    indices_offset = normals_offset + normals_len
    indices_len = len(indices_bytes)

    full_buffer_bytes = points_bytes + normals_bytes + indices_bytes

    # buffer_filename = "CornellBox.bin"
    # buffer_uri = buffer_filename

    buffer = Buffer(byteLength=len(full_buffer_bytes))

    scene_primitives = []
    accessors = []
    bufferViews = []

    # BufferViews
    # 0: Positions
    bufferViews.append(
        BufferView(
            buffer=0,
            byteOffset=points_offset,
            byteLength=points_len,
            target=BUFFER_TARGET_ARRAY_BUFFER,
        )
    )
    # 1: Normals
    bufferViews.append(
        BufferView(
            buffer=0,
            byteOffset=normals_offset,
            byteLength=normals_len,
            target=BUFFER_TARGET_ARRAY_BUFFER,
        )
    )
    # 2: Indices
    bufferViews.append(
        BufferView(
            buffer=0,
            byteOffset=indices_offset,
            byteLength=indices_len,
            target=BUFFER_TARGET_ELEMENT_ARRAY_BUFFER,
        )
    )

    # Global Accessors for attributes
    # Accessor 0: Positions
    accessors.append(
        Accessor(
            bufferView=0,
            componentType=COMPONENT_TYPE_FLOAT,
            count=len(all_positions),
            type=ACCESSOR_TYPE_VEC3,
            min=points_array.min(axis=0).tolist(),
            max=points_array.max(axis=0).tolist(),
        )
    )

    # Accessor 1: Normals
    accessors.append(
        Accessor(
            bufferView=1,
            componentType=COMPONENT_TYPE_FLOAT,
            count=len(all_normals),
            type=ACCESSOR_TYPE_VEC3,
        )
    )

    indices_accessor_base = 2

    for i, p in enumerate(primitives_list):
        # Index Accessor per primitive
        # Byte offset into the BufferView(2)
        # p["index_start"] is index count offset.
        # offset in bytes = index_start * 2
        acc_byte_offset = p["index_start"] * 2

        acc = Accessor(
            bufferView=2,
            byteOffset=acc_byte_offset,
            componentType=COMPONENT_TYPE_UNSIGNED_SHORT,
            count=p["index_count"],
            type=ACCESSOR_TYPE_SCALAR,
        )
        accessors.append(acc)

        # Determine attributes
        # We share one big vertex buffer, so attributes are 0 and 1
        # BUT wait, the indices in `indices_array` refer to vertices in the Accessor 0.
        # So we can just point to Accessor 0 and 1.

        prim = Primitive(
            attributes=Attributes(POSITION=0, NORMAL=1),
            indices=indices_accessor_base + i,
            material=p["mat_idx"],
        )
        scene_primitives.append(prim)

    gltf = GLTF2(
        scene=0,
        scenes=[Scene(nodes=[0])],
        nodes=[Node(mesh=0)],
        meshes=[Mesh(primitives=scene_primitives)],
        materials=materials,
        accessors=accessors,
        bufferViews=bufferViews,
        buffers=[buffer],
    )

    gltf.set_binary_blob(full_buffer_bytes)
    output_glb = os.path.join(output_dir, "CornellBox.glb")
    gltf.save(output_glb)
    print(f"Created Cornell Box at {output_glb}")


if __name__ == "__main__":
    create_cornell_box()
