import os
from pytorch3d.io import load_obj

import cv2
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    BlendParams,
)
from pytorch3d.renderer.materials import Materials
from PIL import Image
import numpy as np
import torch
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.structures import Meshes

import shutil
def convert_image(x,im_size):
    x_out = np.clip(x.detach().cpu().numpy(), -1.0, 1.0)
    x_out -= x_out.min()
    x_out /= x_out.max()
    x_out = (x_out*255).astype(np.uint8)
    for i in range(0,im_size):
        for j in range(0,im_size):
            if x_out[i,j,0] == 255:
                x_out[i,j]=0
    return x_out

rootdir = 'example/models/'

device = torch.device("cuda:0")
list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
for i in range(0,len(list)):
    path = os.path.join(rootdir,list[i])
    print(list[i])
    if os.path.isfile(path):
        str = list[i][0:-4]
        obj_filename = path
        folder = 'example/content/'+str+'/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        shutil.copy(path, folder + list[i])
        verts, faces, aux = load_obj(obj_filename)
        vert_colors = torch.ones(len(verts),3)
        flag = torch.ones(len(verts),1)
        im_size = 512
        data_temp = []
        with open(obj_filename, "r") as f:  # 打开文件
            ind = 0
            while(True):
                line = f.readline()
                str_list = line.split(" ")
                if(str_list[0]!='v'):
                    break
                flag_k = 0
                for i in range(4,7):
                    vert_colors[ind ,i-4] = float(str_list[i])/255
                ind = ind +1
        texture_with_colors = TexturesVertex(verts_features=[vert_colors])
        mesh = Meshes(verts=[verts],faces=[faces.verts_idx],textures=texture_with_colors, )
        materials = Materials(specular_color=((0,0,0),),device=device)
        R, T = look_at_view_transform(33.0, 0, 0, at=((0, 0.4, 0),))
        R1, T1 = look_at_view_transform(33.0, 0, 10, at=((0, 0.4, 0),))
        cameras = FoVPerspectiveCameras(device=device,fov=6.5,R=R, T=T)
        cameras1 = FoVPerspectiveCameras(device=device,fov=6.5,R=R1, T=T1)
        R, T = look_at_view_transform(33.0, 0, -10, at=((-0.0, 0.1, 0),))
        R1, T1 = look_at_view_transform(33.0, 0, 10, at=((-0.0, 0.1, 0),))
        cameras = FoVPerspectiveCameras(device=device,fov=5.0,R=R, T=T)
        cameras1 = FoVPerspectiveCameras(device=device,fov=5.0,R=R1, T=T1)
        raster_settings = RasterizationSettings(
            image_size=im_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        lights = PointLights(device=device, location=[[0, 0, 20.0]])
        blend_params=BlendParams(
            1e-4,
            1e-4,
            background_color=torch.ones(3, dtype=torch.float32, device=device)*0.76,
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights,
                materials=materials,
                blend_params=blend_params,
            )
        )


        renderer1 = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras1,
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=cameras1,
             lights=lights,
                materials=materials,
                blend_params=blend_params,
            )
        )

        ans = cameras.transform_points_screen(device=device,points=verts.reshape(1,-1,3).to(device),image_size=[[im_size,im_size]])
        ans1 = cameras1.transform_points_screen(device=device,points=verts.reshape(1,-1,3).to(device),image_size=[[im_size,im_size]])

        point2d = ans.data[0,:,0:2].cpu().numpy()
        point2d1 = ans1.data[0,:,0:2].cpu().numpy()

        world_to_ndc_transform = cameras.get_full_projection_transform()
        matrix1 = world_to_ndc_transform.get_matrix()
        t1 = (im_size-1.0)/2.0
        matrix2 = torch.tensor([[[-t1,0,0],[0,-t1,0],[0,0,0],[t1,t1,1]]]).to(device)
        matrix_H = torch.bmm(matrix1,matrix2)
        world_to_ndc_transform1 = cameras1.get_full_projection_transform()
        matrix1_1 = world_to_ndc_transform1.get_matrix()
        t1 = (im_size-1.0)/2.0
        matrix2_1 = torch.tensor([[[-t1,0,0],[0,-t1,0],[0,0,0],[t1,t1,1]]]).to(device)
        matrix_H_1 = torch.bmm(matrix1_1,matrix2_1)
        matrix_H_copy = np.array(matrix_H[0,:,:].t().cpu())
        matrix_H_copy_1 = np.array(matrix_H_1[0,:,:].t().cpu())
        np.savetxt(folder+'H1.txt',matrix_H_copy)
        np.savetxt(folder+'H2.txt',matrix_H_copy_1)
        np.savetxt(folder+'point2d1.txt',point2d)
        np.savetxt(folder+'point2d2.txt',point2d1)
        np.savetxt(folder+'flag.txt',flag)
        images = renderer(mesh.to(device))
        images1 = renderer1(mesh.to(device))
        save_im = convert_image(images[0,:,:,0:3],im_size)
        save_im = Image.fromarray(save_im)
        save_im.save(folder+'content1.jpg')
        save_im1 = convert_image(images1[0,:,:,0:3],im_size)
        save_im1 = Image.fromarray(save_im1)
        save_im1.save(folder+'content2.jpg')
        #reconstruction test
        point3d_re = cv2.triangulatePoints(matrix_H_copy,matrix_H_copy_1,point2d.transpose() ,point2d1.transpose())
        point3d_re = point3d_re.transpose()
        denom_3d = point3d_re[..., 3:]
        point3d_re = point3d_re[..., :3] / denom_3d

        d = 1
