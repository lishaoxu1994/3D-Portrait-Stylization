
# https://github.com/sunniesuhyoung/DST

import os
import sys
import time
import torch
import numpy as np
from PIL import Image
import cv2
from styletransfer_pair_rrps import transfer_pair

from vggfeatures import VGG16_Extractor
from utils_plot import convert_image
from utils_misc import pil_loader, pil_resize_long_edge_to, pil_to_tensor

def run_style_pair(warp_lr,pairloss_flag,transform_flag,content_path1, content_path2, style_path, content_folder, style, content_pts_path1, content_pts_path2,
                 style_pts_path1, style_pts_path2, point_pair1_path, point_pair2_path, point_pair_flag_path,matrixH1_path, matrixH2_path,
                 output_dir, output_prefix,im_size,max_iter,checkpoint_iter,content_weight,style_weight,warp_weight,
                 reg_weight,optim,lr,verbose,save_intermediate,save_extra,device,):

    save_path = 'output/'+content_folder
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Print settings
    print('\n\n---------------------------------')
    print('Started Deformable Style Transfer')
    print('---------------------------------')

    print('\nSettings')
    print('   content_path:', content_path1)
    print('   content_path:', content_path2)
    print('   style_path:', style_path)
    print('   content_pts_path:', content_pts_path1)
    print('   style_pts_path:', style_pts_path1)
    print('   content_pts_path:', content_pts_path2)
    print('   style_pts_path:', style_pts_path2)
    print('   output_dir:', output_dir)
    print('   output_prefix:', output_prefix)
    print('   im_size:', im_size)
    print('   max_iter:', max_iter)
    print('   checkpoint_iter:', checkpoint_iter)
    print('   content_weight:', content_weight)
    print('   warp_weight:', warp_weight)
    print('   reg_weight:', reg_weight)
    print('   optim:', optim)
    print('   lr:', lr)
    print('   verbose:', verbose)
    print('   save_intermediate:', save_intermediate)
    print('   save_extra:', save_extra)

    # Create output directory
    if not os.path.exists(output_dir):
       os.makedirs(output_dir)

    # Define feature extractor
    extractor = VGG16_Extractor().to(device)

    # Load content/style images and keypoints
    content_pil1 = pil_loader(content_path1)
    style_pil1 = pil_loader(style_path)
    content_pts1 = np.loadtxt(content_pts_path1, delimiter=',')
    style_pts1 = np.loadtxt(style_pts_path1, delimiter=',')
    content_pil2 = pil_loader(content_path2)
    style_pil2 = pil_loader(style_path)
    content_pts2 = np.loadtxt(content_pts_path2, delimiter=',')
    style_pts2 = np.loadtxt(style_pts_path2, delimiter=',')
    # Rescale images
    content_resized1 = pil_resize_long_edge_to(content_pil1, im_size)
    style_resized1 = pil_resize_long_edge_to(style_pil1, im_size)
    content_im_orig1 = pil_to_tensor(content_resized1).to(device)

    content_resized2 = pil_resize_long_edge_to(content_pil2, im_size)
    style_resized2 = pil_resize_long_edge_to(style_pil2, im_size)
    content_im_orig2 = pil_to_tensor(content_resized2).to(device)
    style_im_orig = pil_to_tensor(style_resized2).to(device)

    # Rescale points (assuming that points are in the original image's scale)
    c_width1, c_height1 = content_pil1.size
    c_fac1 = im_size/max(c_width1, c_height1)
    for i in range(content_pts1.shape[0]):
        content_pts1[i][0] *= c_fac1
        content_pts1[i][1] *= c_fac1
    c_width2, c_height2 = content_pil2.size
    c_fac2 = im_size/max(c_width2, c_height2)
    for i in range(content_pts2.shape[0]):
        content_pts2[i][0] *= c_fac2
        content_pts2[i][1] *= c_fac2

    s_width1, s_height1 = style_pil1.size
    s_fac1 = im_size/max(s_width1, s_height1)
    for i in range(style_pts1.shape[0]):
        style_pts1[i][0] *= s_fac1
        style_pts1[i][1] *= s_fac1
    s_width2, s_height2 = style_pil2.size
    s_fac2 = im_size/max(s_width2, s_height2)
    for i in range(style_pts2.shape[0]):
        style_pts2[i][0] *= s_fac2
        style_pts2[i][1] *= s_fac2

    
    temp = content_pts1[:, 1].copy()
    content_pts1[:, 1] = content_pts1[:, 0]
    content_pts1[:, 0] = temp
    temp = content_pts2[:, 1].copy()
    content_pts2[:, 1] = content_pts2[:, 0]
    content_pts2[:, 0] = temp
    '''
    temp = style_pts1[:, 1].copy()
    style_pts1[:, 1] = style_pts1[:, 0]
    style_pts1[:, 0] = temp
    temp = style_pts2[:, 1].copy()
    style_pts2[:, 1] = style_pts2[:, 0]
    style_pts2[:, 0] = temp
    '''


    content_pts1 = torch.from_numpy(content_pts1).float()
    style_pts1 = torch.from_numpy(style_pts1).float()
    content_pts2 = torch.from_numpy(content_pts2).float()
    style_pts2 = torch.from_numpy(style_pts2).float()

    #pair_point
    point_pair1_load = np.loadtxt(point_pair1_path, delimiter=' ')
    point_pair2_load = np.loadtxt(point_pair2_path, delimiter=' ')
    point_pair_flag_load = np.loadtxt(point_pair_flag_path, delimiter=' ')
    matrixH1 = np.loadtxt(matrixH1_path, delimiter=' ')
    matrixH2 = np.loadtxt(matrixH2_path, delimiter=' ')
    c_width2, c_height2 = content_pil2.size
    s_pair = im_size/max(c_width2, c_height2)
    for i in range(point_pair1_load.shape[0]):
        point_pair1_load[i][0] *= s_pair
        point_pair1_load[i][1] *= s_pair
        point_pair2_load[i][0] *= s_pair
        point_pair2_load[i][1] *= s_pair
    #from keypoint_update import point_update
    #from scipy.spatial import Delaunay
    #point_pair1_load, point_pair2_load = point_update(point_pair1_load, point_pair2_load)
    #tri = Delaunay(point_pair1_load)
    temp = point_pair1_load[:, 1].copy()
    point_pair1_load[:, 1] = point_pair1_load[:, 0]
    point_pair1_load[:, 0] = temp
    temp = point_pair2_load[:, 1].copy()
    point_pair2_load[:, 1] = point_pair2_load[:, 0]
    point_pair2_load[:, 0] = temp

    point_pair1 = np.clip(point_pair1_load, 0, im_size - 1)
    point_pair2 = np.clip(point_pair2_load, 0, im_size - 1)

    point_pair1 = torch.from_numpy(point_pair1).float()
    point_pair2 = torch.from_numpy(point_pair2).float()
    pair_flag = torch.from_numpy(point_pair_flag_load).float()

    # Initialize the output image as the content image (This is a simpler initialization
    # than what's described in the STROTSS paper, but we found that results are similar)
    initial_im1 = content_im_orig1.clone()
    initial_im2 = content_im_orig2.clone()

    # Run deformable style transfer
    start_time = time.time()
    output1, output2, pair_warp1, pair_warp2 = transfer_pair(pairloss_flag,transform_flag,initial_im1, initial_im2, content_im_orig1, content_im_orig2, style_im_orig,point_pair1,point_pair2,pair_flag, extractor,
                content_path1, content_path2, style_path, content_pts1, content_pts2, style_pts1, style_pts2, style_pts_path1, style_pts_path2,
                output_dir, output_prefix,
                im_size=im_size,
                max_iter=max_iter,
                checkpoint_iter=checkpoint_iter,
                content_weight=content_weight,
                style_weight=style_weight,
                warp_weight=warp_weight,
                reg_weight=reg_weight,
                optim=optim,
                lr=lr,
                lr1=lr,
                lr2=lr,
                verbose=verbose,
                warp_lr_fac1 = warp_lr,
                warp_lr_fac2 = warp_lr,
                save_intermediate=save_intermediate,
                save_extra=save_extra,
                device=device)

    #rebuild3d
    img_pos1 = pair_warp1.detach().cpu().numpy()
    img_pos2 = pair_warp2.detach().cpu().numpy()
    temp = img_pos1[:, 1].copy()
    img_pos1[:, 1] = img_pos1[:, 0]
    img_pos1[:, 0] = temp
    temp = img_pos2[:, 1].copy()
    img_pos2[:, 1] = img_pos2[:, 0]
    img_pos2[:, 0] = temp

    point3d = cv2.triangulatePoints(matrixH1,matrixH2,img_pos1.transpose(),img_pos2.transpose())
    point3d = point3d.transpose()
    deonm_3d = point3d[...,3:]
    point3d = point3d[...,:3] / deonm_3d


    point3d = point3d[..., :3]
    min_floor = 0
    max_floor = im_size - 1
    floor1 = torch.clamp(torch.clamp(pair_warp1, min=min_floor), max=max_floor)
    img_pos1_c = floor1.detach().cpu().int().numpy()
    color_img1 = convert_image(output1[0])
    colors_3d1 = color_img1[img_pos1_c[:, 0], img_pos1_c[:, 1], :]
    point3d_color1 = np.append(point3d, colors_3d1, axis=1)
    #np.savetxt(save_path + '/' + style + '_'  + optim + '_'  +'_point1.txt', point3d_color1)

    # Write the stylized output image
    save_im1 = convert_image(output1[0])
    save_im1 = Image.fromarray(save_im1)
    save_im1.save(save_path + '/' + style + '_' +'ans1.png')
    save_im2 = convert_image(output2[0])
    save_im2 = Image.fromarray(save_im2)
    save_im2.save(save_path + '/' + style + '_' +'ans2.png')
    print('\nSaved the stylized image at', save_path)
    #get_mesh(tri.simplices, point3d_color1, save_path + '/' + style + '_' + optim + '_' + str(lr) + '_' + str(reg_weight) + '_' + str(pairloss_flag) + '_' + str(transform_flag) + '_' + str(warp_weight) + '_' + str(warp_lr)  +'ans.obj')
    get_mesh('example/content/' + content_folder + '/'+ content_folder + '.obj', point3d_color1, save_path + '/' + style + '_ans.obj')
    # Report total time
    end_time = time.time()
    total_time = (end_time - start_time) / 60
    print('\nFinished after {:04.3f} minutes\n'.format(total_time))


def get_mesh_simple(surfaces, point_color, to_be_saved_path):

    with open(to_be_saved_path, 'w') as f3:
        for ind in range(0,len(point_color)):
            point = ' '.join(str(i) for i in point_color[ind,:])
            line = 'v ' + point
            f3.write(line + '\n')
        for ind in range(0,len(surfaces)):
            surface = ' '.join(str(i) for i in surfaces[ind, :])
            line = 'f ' + str(surface)
            f3.write(line + '\n')
        f3.close()
    return
def get_mesh(original_3D_path, generated_3D, to_be_saved_path):
    with open(original_3D_path, 'r') as f1:
        row_list1 = f1.read().splitlines()
    with open(to_be_saved_path, 'w') as f3:
        for ind in range(0,len(generated_3D)):
            point = ' '.join(str(i) for i in generated_3D[ind,:])
            line = 'v ' + point
            f3.write(line + '\n')
        for row in row_list1:
            if len(row)>0 and row[0] == 'f':
                f3.write(row + '\n')
        f3.close()
    return