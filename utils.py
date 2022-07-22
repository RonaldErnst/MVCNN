import io
import torch

import numpy as np

from pytorch3d.io.mtl_io import make_mesh_texture_atlas
from pytorch3d.io.obj_io import (
    _parse_obj,
    _format_faces_indices,
    load_mtl,
    _Faces,
    _Aux
)
from pytorch3d.io.utils import _make_tensor

from PIL import Image

from os.path import normpath, join

from zipp import Path


def _parse_mtl(file, device="cpu"):
    material_properties = {}
    texture_files = {}
    material_name = ""

    for line in file:
        tokens = [tok.decode().lower() for tok in line.strip().split()]
        if not tokens:
            continue
        if tokens[0] == "newmtl":
            material_name = tokens[1]
            material_properties[material_name] = {}
        elif tokens[0] == "map_kd":
            # diffuse texture map
            # account for the case where filenames might have spaces
            filename = line.strip()[7:].decode()
            texture_files[material_name] = filename
        elif tokens[0] == "kd":
            # rgb diffuse reflectivity
            kd = np.array(tokens[1:4]).astype(np.float32)
            kd = torch.from_numpy(kd).to(device)
            material_properties[material_name]["diffuse_color"] = kd
        elif tokens[0] == "ka":
            # rgb ambient reflectivity
            ka = np.array(tokens[1:4]).astype(np.float32)
            ka = torch.from_numpy(ka).to(device)
            material_properties[material_name]["ambient_color"] = ka
        elif tokens[0] == "ks":
            # rgb specular reflectivity
            ks = np.array(tokens[1:4]).astype(np.float32)
            ks = torch.from_numpy(ks).to(device)
            material_properties[material_name]["specular_color"] = ks
        elif tokens[0] == "ns":
            # specular exponent
            ns = np.array(tokens[1:4]).astype(np.float32)
            ns = torch.from_numpy(ns).to(device)
            material_properties[material_name]["shininess"] = ns

    return material_properties, texture_files

def _load_texture_images(
    material_names,
    archive,
    data_dir: str,
    material_properties,
    texture_files,
):
    final_material_properties = {}
    texture_images = {}

    # Only keep the materials referenced in the obj.
    for material_name in material_names:
        if material_name in texture_files:
            # Load the texture image.
            path = normpath(join(data_dir, texture_files[material_name]))
            image = Path(archive, at=path).read_bytes()
            image = Image.open(io.BytesIO(image))
            image = image.convert("RGB")
            image = np.asarray(image).astype(np.float32)
            image = torch.from_numpy(image / 255.0)
            texture_images[material_name] = image

        if material_name in material_properties:
            final_material_properties[material_name] = material_properties[
                material_name
            ]

    return final_material_properties, texture_images

def load_obj(
    filename,
    archive,
    data_dir,
    device
):
    """
    Load a mesh from a file-like object. See load_obj function more details.
    Any material files associated with the obj are expected to be in the
    directory given by data_dir.
    """

    with archive.open(filename, 'r') as f_obj:
        (
            verts,
            normals,
            verts_uvs,
            faces_verts_idx,
            faces_normals_idx,
            faces_textures_idx,
            faces_materials_idx,
            material_names,
            mtl_path,
        ) = _parse_obj(f_obj, data_dir)

    verts = _make_tensor(verts, cols=3, dtype=torch.float32, device=device)  # (V, 3)
    normals = _make_tensor(
        normals, cols=3, dtype=torch.float32, device=device
    )  # (N, 3)
    verts_uvs = _make_tensor(
        verts_uvs, cols=2, dtype=torch.float32, device=device
    )  # (T, 2)

    faces_verts_idx = _format_faces_indices(
        faces_verts_idx, verts.shape[0], device=device
    )

    # Repeat for normals and textures if present.
    if len(faces_normals_idx):
        faces_normals_idx = _format_faces_indices(
            faces_normals_idx, normals.shape[0], device=device, pad_value=-1
        )
    if len(faces_textures_idx):
        faces_textures_idx = _format_faces_indices(
            faces_textures_idx, verts_uvs.shape[0], device=device, pad_value=-1
        )
    if len(faces_materials_idx):
        faces_materials_idx = torch.tensor(
            faces_materials_idx, dtype=torch.int64, device=device
        )

    with archive.open(mtl_path, 'r') as mtl_file:
        material_properties, texture_files = _parse_mtl(mtl_file, device)
    material_colors, texture_images = _load_texture_images(material_names, archive, data_dir,
        material_properties, texture_files)

    # Using the images and properties from the
    # material file make a per face texture map.

    # Create an array of strings of material names for each face.
    # If faces_materials_idx == -1 then that face doesn't have a material.
    idx = faces_materials_idx.cpu().numpy()
    face_material_names = np.array(material_names)[idx]  # (F,)
    face_material_names[idx == -1] = ""

    # Construct the atlas.
    texture_atlas = make_mesh_texture_atlas(
        material_colors,
        texture_images,
        face_material_names,
        faces_textures_idx,
        verts_uvs,
        texture_size=4,
        texture_wrap='repeat',
    )

    faces = _Faces(
        verts_idx=faces_verts_idx,
        normals_idx=faces_normals_idx,
        textures_idx=faces_textures_idx,
        materials_idx=faces_materials_idx,
    )
    return verts, faces, texture_atlas