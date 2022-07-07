from turtle import back
import torch

import matplotlib.pyplot as plt

from os import getcwd

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    TexturesAtlas,
)

from zipfile import ZipFile

from utils import load_obj


class ObjMultiViewRenderer:
    def __init__(self, archive, device, image_size=256):
        self.archive = archive
        self.device = device

        self.batch_size = 12
        elevation = 30
        azimuth = torch.linspace(0, 360 // self.batch_size *
            (self.batch_size - 1), self.batch_size)
        
        R, T = look_at_view_transform(dist=1.5, elev=elevation, azim=azimuth)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=image_size
        )
        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )

        shader = HardPhongShader(device=device, cameras=cameras)
        self.renderer = MeshRenderer(rasterizer, shader)
    
    def _get_centroid(verts, faces, areas):
        triangles = torch.cat(
            tuple(verts[faces[:, i]] for i in range(3)), dim=1
        ).reshape((len(faces), 3, 3))
        centers = triangles.mean(dim=1)
        areas = torch.cat((
                areas.reshape((len(faces), 1)),
            ) * 3, dim=1
        )
        return torch.sum(centers * areas, dim=0) / torch.sum(areas, dim=0)

    def _create_centered_mesh(verts, faces, texture_atlas):
        mesh = Meshes(
            verts=[verts],
            faces=[faces.verts_idx],
            textures=TexturesAtlas(atlas=[texture_atlas])
        )
        verts -= ObjMultiViewRenderer._get_centroid(verts, faces.verts_idx,
            mesh.faces_areas_packed())
        return mesh

    def render(self, filename, data_dir):
        verts, faces, texture_atlas = load_obj(
            filename,
            self.archive,
            data_dir,
            device=self.device,
        )
        
        mesh = ObjMultiViewRenderer._create_centered_mesh(verts, faces, texture_atlas)
        
        return self.renderer(mesh.extend(self.batch_size))
    
    def saveimg(self, objname):
        with open(objname + '.obj') as file:
            images = self.render(file)
        plt.figure(figsize=(10, 10))
        for i in range(len(images)):
            plt.imsave(objname + f'_{i}.png', images[i, ..., :3].cpu().numpy())


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    omvr = ObjMultiViewRenderer(device)
    omvr.saveimg('bank')
