

import time
from cv2 import transform
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
from loss import content_loss, remd_loss, moment_loss, TV, pairwise_distances_sq_l2,pair_loss_cal
from warp import apply_warp
from utils_pyr import syn_lap_pyr, dec_lap_pyr
from utils_keypoints import init_keypoint_params, gen_dst_pts_keypoints
from utils_misc import sample_indices, spatial_feature_extract
from utils_save import save_loss, save_points
from utils_plot import save_plots, plot_intermediate
from utils_plot import convert_image
from image_transfer import rotate_3
from keypoint_update import point_update
def transfer_pair(pairloss_flag,transform_flag,input_im1, input_im2, content_im1, content_im2, style_im, point_pair1, point_pair2, pair_flag, extractor, content_path1,
             content_path2, style_path, content_pts1,content_pts2, style_pts1,style_pts2, style_pts_path1,style_pts_path2, output_dir, output_prefix,
        im_size = 256,
        max_iter = 250,
        checkpoint_iter = 50,
        content_weight = 8.,
        style_weight=1.0,
        warp_weight = 0.3,
        reg_weight = 10,
        scales = 3,
        pyr_levs1 = 5,
        pyr_levs2 = 5,
        sharp_warp = False,
        optim = 'adam',
        lr = 1e-3,
        lr1 = 1e-3,
        lr2 = 1e-3,
        warp_lr_fac1 = 70.,
        warp_lr_fac2 = 70.,
        verbose = False,
        save_intermediate = False,
        save_extra = False,
        device = 'cuda:0'):

    # If warp weight is 0, run the base method STROTSS
    use_DST = True
    if warp_weight == 0.:
        use_DST = False

    if transform_flag == 1:
        style_im1, style_pts1 = rotate_3(style_im, style_pts1, angle_vari= -10)
        style_im2, style_pts2 = rotate_3(style_im, style_pts2, angle_vari= 10)

        style_im1 = torch.from_numpy(style_im1).to(device)
        style_im2 = torch.from_numpy(style_im2).to(device)
        style_pts1 = torch.from_numpy(style_pts1).float()
        style_pts2 = torch.from_numpy(style_pts2).float()
    else:
        style_im1 = style_im
        style_im2 = style_im
    '''
    temp = style_pts1[:, 1].clone()
    style_pts1[:, 1] = style_pts1[:, 0]
    style_pts1[:, 0] = temp
    temp = style_pts2[:, 1].clone()
    style_pts2[:, 1] = style_pts2[:, 0]
    style_pts2[:, 0] = temp
    '''
    point_pair1_copy = point_pair1.clone()
    point_pair2_copy = point_pair2.clone()
    point_pair1, point_pair2 = point_update(point_pair1, point_pair2)

    # Initialize warp parameters
    src_Kpts1, target_Kpts1, border_Kpts1, no_flow_Kpts1 = init_keypoint_params(input_im1, content_path1, content_pts1, style_pts1, device)
    src_Kpts2, target_Kpts2, border_Kpts2, no_flow_Kpts2 = init_keypoint_params(input_im2, content_path2, content_pts2, style_pts2, device)
    thetas_Kpts1 = Variable(torch.rand_like(src_Kpts1).data * 1e-4, requires_grad=True)
    thetas_Kpts2 = Variable(torch.rand_like(src_Kpts2).data * 1e-4, requires_grad=True)
    # Clamp the target points so that they don't go outside the boundary
    target_Kpts1[:,0] = torch.clamp(target_Kpts1[:,0], min=5, max=content_im1.size(2)-5)
    target_Kpts1[:,1] = torch.clamp(target_Kpts1[:,1], min=5, max=content_im1.size(3)-5)
    target_Kpts_o1 = target_Kpts1.clone().detach()
    target_Kpts2[:,0] = torch.clamp(target_Kpts2[:,0], min=5, max=content_im2.size(2)-5)
    target_Kpts2[:,1] = torch.clamp(target_Kpts2[:,1], min=5, max=content_im2.size(3)-5)
    target_Kpts_o2 = target_Kpts2.clone().detach()
    # Assign colors to each set of points (used for visualization only)
    np.random.seed(1)
    colors1 = []
    colors2 = []
    for j in range(src_Kpts1.shape[0]):
        colors1.append(np.random.random(size=3))
    for j in range(src_Kpts2.shape[0]):
        colors2.append(np.random.random(size=3))
    # Initialize pixel parameters
    s_pyr1 = dec_lap_pyr(input_im1, pyr_levs1)
    s_pyr1 = [Variable(li.data, requires_grad=True) for li in s_pyr1]
    s_pyr2 = dec_lap_pyr(input_im2, pyr_levs2)
    s_pyr2 = [Variable(li.data, requires_grad=True) for li in s_pyr2]
    # Define parameters to be optimized
    s_pyr_list1 = [{'params': si1} for si1 in s_pyr1]
    s_pyr_list2 = [{'params': si2} for si2 in s_pyr2]
    if use_DST:
        thetas_opt_list1 = [{'params': thetas_Kpts1, 'lr': lr * warp_lr_fac1}]
        thetas_opt_list2 = [{'params': thetas_Kpts2, 'lr': lr * warp_lr_fac2}]
    else:
        thetas_opt_list1 = []
        thetas_opt_list2 = []

    # Construct optimizer
    opt_ans = s_pyr_list1 + s_pyr_list2 + thetas_opt_list1 + thetas_opt_list2
    if optim == 'sgd':
        optimizer = torch.optim.SGD(s_pyr_list1 + s_pyr_list2 + thetas_opt_list1 + thetas_opt_list2, lr=lr, momentum=0.9)
        #optimizer1 = torch.optim.SGD(s_pyr_list1 + thetas_opt_list1, lr=lr1, momentum=0.9)
        #optimizer2 = torch.optim.SGD(s_pyr_list2 + thetas_opt_list2, lr=lr2, momentum=0.9)
    elif optim == 'Adadelta':  # 1
        if pairloss_flag ==0:
            optimizer1 = torch.optim.Adadelta(s_pyr_list1 + thetas_opt_list1, lr=lr1,rho=0.9, eps=1e-06, weight_decay=0)
            optimizer2 = torch.optim.Adadelta(s_pyr_list2 + thetas_opt_list2, lr=lr2,rho=0.9, eps=1e-06, weight_decay=0)
        else:
            optimizer = torch.optim.Adadelta(s_pyr_list1 + s_pyr_list2 + thetas_opt_list1 + thetas_opt_list2, lr=lr)    
    elif optim == 'Adagrad':  # no
        optimizer = torch.optim.Adagrad(s_pyr_list1 + s_pyr_list2 + thetas_opt_list1 + thetas_opt_list2, lr=lr)
    elif optim == 'AdamW':  # no
        optimizer = torch.optim.AdamW(s_pyr_list1 + s_pyr_list2 + thetas_opt_list1 + thetas_opt_list2, lr=lr)
    elif optim == 'SparseAdam':  # SparseAdam does not support dense gradients, please consider Adam instead
        optimizer = torch.optim.SparseAdam(s_pyr_list1 + s_pyr_list2 + thetas_opt_list1 + thetas_opt_list2, lr=lr)
    elif optim == 'Adamax':
        optimizer = torch.optim.Adamax(s_pyr_list1 + s_pyr_list2 + thetas_opt_list1 + thetas_opt_list2, lr=lr)
    elif optim == 'ASGD':  # U
        optimizer = torch.optim.ASGD(s_pyr_list1 + s_pyr_list2 + thetas_opt_list1 + thetas_opt_list2, lr=lr)
    elif optim == 'RMSprop':  # no
        optimizer = torch.optim.RMSprop(s_pyr_list1 + s_pyr_list2 + thetas_opt_list1 + thetas_opt_list2, lr=lr)
    elif optim == 'Rprop':  # need stepsize
        optimizer = torch.optim.Rprop(s_pyr_list1 + s_pyr_list2 + thetas_opt_list1 + thetas_opt_list2, lr=lr)
    else:  # 1
        optimizer = torch.optim.Adam(s_pyr_list1 + s_pyr_list2 + thetas_opt_list1 + thetas_opt_list2,lr=lr)

    # Set scales
    scale_list = list(range(scales))
    if scales == 1:
        scale_list = [0]

    # Create lists to store various loss values
    ell_list = []
    ell_style_list1 = []
    ell_content_list1 = []
    ell_warp_list1 = []
    ell_warp_TV_list1 = []
    ell_style_list2 = []
    ell_content_list2 = []
    ell_warp_list2 = []
    ell_warp_TV_list2 = []
    # Iteratively stylize over more levels of image pyramid
    point_pair1_save =[]
    point_pair2_save =[]
    for scale in scale_list:

        down_fac = 2**(scales-1-scale)
        begin_ind = (scales-1-scale)
        content_weight_scaled = content_weight*down_fac

        print('\nOptimizing at scale {}, image size ({}, {})'.format(scale + 1, content_im1.size(2) // down_fac, content_im1.size(3) // down_fac))
        if down_fac > 1.:
            content_im_scaled1 = F.interpolate(content_im1, (content_im1.size(2)//down_fac, content_im1.size(3)//down_fac), mode='bilinear')
            content_im_scaled2 = F.interpolate(content_im2, (content_im2.size(2)//down_fac, content_im2.size(3)//down_fac), mode='bilinear')
            style_im_scaled1 = F.interpolate(style_im1, (style_im1.size(2)//down_fac, style_im1.size(3)//down_fac), mode='bilinear')
            style_im_scaled2 = F.interpolate(style_im2, (style_im2.size(2)//down_fac, style_im2.size(3)//down_fac), mode='bilinear')
        else:
            content_im_scaled1 = content_im1.clone()
            content_im_scaled2 = content_im2.clone()
            style_im_scaled1 = style_im1.clone()
            style_im_scaled2 = style_im2.clone()
        # Compute feature maps that won't change for this scale
        with torch.no_grad():
            feat_content1 = extractor(content_im_scaled1)
            feat_content2 = extractor(content_im_scaled2)
            feat_style = None
            for i in range(5):
                with torch.no_grad():
                    feat_e1 = extractor.forward_samples_hypercolumn(style_im_scaled1, samps=1000)
                    feat_e2 = extractor.forward_samples_hypercolumn(style_im_scaled2, samps=1000)
                    feat_style1 = feat_e1 if feat_style is None else torch.cat((feat_style, feat_e1), dim=2)
                    feat_style2 = feat_e2 if feat_style is None else torch.cat((feat_style, feat_e2), dim=2)

            feat_max = 3 + 2*64 + 2*128 + 3*256 + 2*512 # 2179 = sum of all extracted channels
            spatial_style1 = feat_style1.view(1, feat_max, -1, 1)
            spatial_style2 = feat_style2.view(1, feat_max, -1, 1)
            xx1, xy1 = sample_indices(feat_content1[0], feat_style1)
            xx2, xy2 = sample_indices(feat_content2[0], feat_style2)

        # Begin optimization for this scale
        for i in range(max_iter):
            if pairloss_flag == 0:
                optimizer1.zero_grad()
                optimizer2.zero_grad()
            else:
                optimizer.zero_grad()

            # Get current stylized image from the laplacian pyramid
            curr_im1 = syn_lap_pyr(s_pyr1[begin_ind:])
            new_im1 = curr_im1.clone()
            content_im_warp1 = content_im_scaled1.clone()
            curr_im2 = syn_lap_pyr(s_pyr2[begin_ind:])
            new_im2 = curr_im2.clone()
            content_im_warp2 = content_im_scaled2.clone()

            # Generate destination points with the current thetas
            src_Kpts_aug1, dst_Kpts_aug1, flow_Kpts_aug1 = gen_dst_pts_keypoints(src_Kpts1, thetas_Kpts1, no_flow_Kpts1, border_Kpts1)
            src_Kpts_aug2, dst_Kpts_aug2, flow_Kpts_aug2 = gen_dst_pts_keypoints(src_Kpts2, thetas_Kpts2, no_flow_Kpts2, border_Kpts2)
            # Calculate warp loss
            ell_warp1 = torch.norm(target_Kpts_o1 - dst_Kpts_aug1[:target_Kpts1.size(0)], dim=1).mean()
            ell_warp2 = torch.norm(target_Kpts_o2 - dst_Kpts_aug2[:target_Kpts2.size(0)], dim=1).mean()
            # Scale points to [0-1]
            src_Kpts_aug1 = src_Kpts_aug1/torch.max(src_Kpts_aug1, 0, keepdim=True)[0]
            dst_Kpts_aug1 = dst_Kpts_aug1/torch.max(dst_Kpts_aug1, 0, keepdim=True)[0]
            dst_Kpts_aug1 = torch.clamp(dst_Kpts_aug1, min=0., max=1.)
            src_Kpts_aug2 = src_Kpts_aug2/torch.max(src_Kpts_aug2, 0, keepdim=True)[0]
            dst_Kpts_aug2 = dst_Kpts_aug2/torch.max(dst_Kpts_aug2, 0, keepdim=True)[0]
            dst_Kpts_aug2 = torch.clamp(dst_Kpts_aug2, min=0., max=1.)
            #point_pair adjust
            point_pair1_temp = point_pair1*curr_im1.size(2)/content_im1.size(2)
            point_pair2_temp = point_pair2*curr_im1.size(2)/content_im1.size(2)
            # Warp
            new_im1, content_im_warp1, warp_field1, point_pair1_temp = apply_warp(new_im1, point_pair1_temp.to(device), [src_Kpts_aug1], [dst_Kpts_aug1], device, sharp=sharp_warp, im2=content_im_warp1)
            new_im1 = new_im1.to(device)
            new_im2, content_im_warp2, warp_field2, point_pair2_temp = apply_warp(new_im2, point_pair2_temp.to(device), [src_Kpts_aug2], [dst_Kpts_aug2], device, sharp=sharp_warp, im2=content_im_warp2)
            new_im2 = new_im2.to(device)
            #point_pair1_temp = point_pair1_temp/curr_im1.size(2)*content_im1.size(2)
            #point_pair2_temp = point_pair2_temp/curr_im1.size(2)*content_im1.size(2)   
            
            #point_pair adjust
            #point_pair1 = point_pair1_temp*content_im1.size(2)/curr_im1.size(2)
            #point_pair2 = point_pair2_temp*content_im1.size(2)/curr_im1.size(2)
            # Calculate total variation
            ell_warp_TV1 = TV(warp_field1)
            ell_warp_TV2 = TV(warp_field2)
            # Extract VGG features of warped and unwarped stylized images
            feat_result_warped1 = extractor(new_im1)
            feat_result_unwarped1 = extractor(curr_im1)
            feat_result_warped2 = extractor(new_im2)
            feat_result_unwarped2 = extractor(curr_im2)
            # Sample features to calculate losses with
            n = 2048
            if i % 1 == 0 and i != 0:
                np.random.shuffle(xx1)
                np.random.shuffle(xy1)
                np.random.shuffle(xx2)
                np.random.shuffle(xy2)
            spatial_result_warped1, spatial_content1 = spatial_feature_extract(feat_result_warped1, feat_content1, xx1[:n], xy1[:n])
            spatial_result_unwarped1, _ = spatial_feature_extract(feat_result_unwarped1, feat_content1, xx1[:n], xy1[:n])
            spatial_result_warped2, spatial_content2 = spatial_feature_extract(feat_result_warped2, feat_content2, xx2[:n], xy2[:n])
            spatial_result_unwarped2, _ = spatial_feature_extract(feat_result_unwarped2, feat_content2, xx2[:n], xy2[:n])
            
            style_para = 10
            # Content loss
            ell_content1 = content_loss(spatial_result_unwarped1, spatial_content1)
            ell_content2 = content_loss(spatial_result_unwarped2, spatial_content2)
            # Style loss

            # Lstyle(Unwarped X, S)
            loss_remd11 = remd_loss(spatial_result_unwarped1, spatial_style1, cos_d=True)
            loss_moment11 = moment_loss(spatial_result_unwarped1, spatial_style1, moments=[1,2])
            loss_color11 = remd_loss(spatial_result_unwarped1[:,:3,:,:], spatial_style1[:,:3,:,:], cos_d=False)
            loss_style11 = loss_remd11 + loss_moment11 + (1./max(content_weight_scaled, 1.))*loss_color11
            loss_remd12 = remd_loss(spatial_result_unwarped2, spatial_style2, cos_d=True)
            loss_moment12 = moment_loss(spatial_result_unwarped2, spatial_style2, moments=[1,2])
            loss_color12 = remd_loss(spatial_result_unwarped2[:,:3,:,:], spatial_style2[:,:3,:,:], cos_d=False)
            loss_style12 = loss_remd12 + loss_moment12 + (1./max(content_weight_scaled, 1.))*loss_color12
            # Lstyle(Warped X, S)
            loss_remd21 = remd_loss(spatial_result_warped1, spatial_style1, cos_d=True)
            loss_moment21 = moment_loss(spatial_result_warped1, spatial_style1, moments=[1,2])
            loss_color21 = remd_loss(spatial_result_warped1[:,:3,:,:], spatial_style1[:,:3,:,:], cos_d=False)
            loss_style21 = loss_remd21 + loss_moment21 + (1./max(content_weight_scaled, 1.))*loss_color21
            loss_remd22 = remd_loss(spatial_result_warped2, spatial_style2, cos_d=True)
            loss_moment22 = moment_loss(spatial_result_warped2, spatial_style2, moments=[1,2])
            loss_color22 = remd_loss(spatial_result_warped2[:,:3,:,:], spatial_style2[:,:3,:,:], cos_d=False)
            loss_style22 = loss_remd22 + loss_moment22 + (1./max(content_weight_scaled, 1.))*loss_color22
            # pairloss
            pair_loss = pair_loss_cal(new_im1, new_im2, point_pair1_temp.to(device), point_pair2_temp.to(device), pair_flag.to(device), im_size)
            pair_loss_weight = 20
            if pairloss_flag == 0:
                pair_loss = 0
            #pair_loss = 0
            # Total loss
            if use_DST:
                ell_style1 = style_weight * (loss_style11 + loss_style21)
                ell_style2 = style_weight * (loss_style12 + loss_style22)
                ell = content_weight_scaled * ell_content1 + ell_style1 + warp_weight * ell_warp1 + reg_weight * ell_warp_TV1 + \
                      content_weight_scaled * ell_content2 + ell_style2 + warp_weight * ell_warp2 + reg_weight * ell_warp_TV2 + pair_loss*pair_loss_weight
                ell1 = content_weight_scaled * ell_content1 + ell_style1 + warp_weight * ell_warp1 + reg_weight * ell_warp_TV1
                ell2 = content_weight_scaled * ell_content2 + ell_style2 + warp_weight * ell_warp2 + reg_weight * ell_warp_TV2
            else:
                ell_style1 = style_weight * (loss_style11)
                ell_style2 = style_weight * (loss_style12)
                ell = content_weight_scaled * ell_content1 + ell_style1 + content_weight_scaled * ell_content2 + ell_style2 + pair_loss*pair_loss_weight
                ell1 = content_weight_scaled * ell_content1 + ell_style1 + pair_loss
                ell2 = content_weight_scaled * ell_content2 + ell_style2 + pair_loss

            # Record loss values
            ell_list.append(ell.item())
            ell_content_list1.append(ell_content1.item())
            ell_style_list1.append(ell_style1.item())
            ell_warp_list1.append(ell_warp1.item())
            ell_warp_TV_list1.append(ell_warp_TV1.item())

            ell_content_list2.append(ell_content2.item())
            ell_style_list2.append(ell_style2.item())
            ell_warp_list2.append(ell_warp2.item())
            ell_warp_TV_list2.append(ell_warp_TV2.item())
            # Output intermediate loss
            if i==0 or i%checkpoint_iter == 0:
                save_im1 = convert_image(new_im1[0])
                save_im1 = Image.fromarray(save_im1)
                save_im1.save(output_dir + '/' + output_prefix + optim +str(scale+1)+'_'+str(i)+'_'+ '1.png')
                save_im2 = convert_image(new_im2[0])
                save_im2 = Image.fromarray(save_im2)
                save_im2.save(output_dir + '/' + output_prefix + optim +str(scale+1)+'_'+str(i)+'_'+  '2.png')
                print('   STEP {:03d}: Loss {:04.3f}'.format(i, ell))
                if verbose:
                    print('1')
                    print('             = alpha*Lcontent {:04.3f}'.format(content_weight_scaled*ell_content1))
                    print('               + Lstyle {:04.3f}'.format(ell_style1))
                    print('               + beta*Lwarp {:04.3f}'.format(warp_weight*ell_warp1))
                    print('               + gamma*TV {:04.3f}'.format(reg_weight*ell_warp_TV1))
                    print('2')
                    print('             = alpha*Lcontent {:04.3f}'.format(content_weight_scaled*ell_content2))
                    print('               + Lstyle {:04.3f}'.format(ell_style2))
                    print('               + beta*Lwarp {:04.3f}'.format(warp_weight*ell_warp2))
                    print('               + gamma*TV {:04.3f}'.format(reg_weight*ell_warp_TV2))
                    print('3')
                    print('               + pair_loss {:04.3f}'.format(pair_loss*pair_loss_weight))
                if save_intermediate:
                    plot_intermediate(new_im1, content_im_warp1, output_dir, output_prefix, colors1,
                                        down_fac, src_Kpts1, thetas_Kpts1, target_Kpts1, scale, i)
                    plot_intermediate(new_im2, content_im_warp2, output_dir, output_prefix, colors2,
                                        down_fac, src_Kpts2, thetas_Kpts2, target_Kpts2, scale, i)
            # Take a gradient step

            if pairloss_flag == 0:
                ell1.backward()
                optimizer1.step()
                ell2.backward()
                optimizer2.step()
            else:  
                ell.backward()
                optimizer.step()    


    # Optimization finished
    src_Kpts_aug1, dst_Kpts_aug1, flow_Kpts_aug1 = gen_dst_pts_keypoints(src_Kpts1, thetas_Kpts1, no_flow_Kpts1, border_Kpts1)
    sizes1 = torch.FloatTensor([new_im1.size(2), new_im1.size(3)]).to(device)
    src_Kpts_aug1 = src_Kpts_aug1/sizes1
    dst_Kpts_aug1 = dst_Kpts_aug1/sizes1
    dst_Kpts_aug1 = torch.clamp(dst_Kpts_aug1, min=0., max=1.)
    dst_Kpts1 = dst_Kpts_aug1[:src_Kpts1.size(0)]

    src_Kpts_aug2, dst_Kpts_aug2, flow_Kpts_aug2 = gen_dst_pts_keypoints(src_Kpts2, thetas_Kpts2, no_flow_Kpts2, border_Kpts2)
    sizes2 = torch.FloatTensor([new_im2.size(2), new_im2.size(3)]).to(device)
    src_Kpts_aug2 = src_Kpts_aug2/sizes2
    dst_Kpts_aug2 = dst_Kpts_aug2/sizes2
    dst_Kpts_aug2 = torch.clamp(dst_Kpts_aug2, min=0., max=1.)
    dst_Kpts2 = dst_Kpts_aug2[:src_Kpts2.size(0)]
    # Apply final warp
    sharp_final = True
    new_im1 = curr_im1.clone()
    content_im_warp1 = content_im1.clone()
    new_im1, pair_warp1 = apply_warp(new_im1, point_pair1_copy.to(device), [src_Kpts_aug1], [dst_Kpts_aug1], device, sharp=sharp_final)
    new_im2 = curr_im2.clone()
    content_im_warp2 = content_im2.clone()
    new_im2, pair_warp2 = apply_warp(new_im2, point_pair2_copy.to(device), [src_Kpts_aug2], [dst_Kpts_aug2], device, sharp=sharp_final)
    # Optionally save loss, keypoints, and optimized warp parameter thetas
    if save_extra:
        save_plots(im_size, curr_im1, new_im1, content_im1, style_im, output_dir, output_prefix, style_path, style_pts_path1, colors1,
                    src_Kpts1, src_Kpts_aug1, dst_Kpts1*sizes1, dst_Kpts_aug1, target_Kpts1, target_Kpts_o1, border_Kpts1, device)
        save_loss(output_dir, output_prefix, content_weight, warp_weight, reg_weight, max_iter, scale_list,
                    ell_list, ell_style_list1, ell_content_list1, ell_warp_list1, ell_warp_TV_list1)
        save_points(output_dir, output_prefix, src_Kpts1, dst_Kpts1*sizes1, src_Kpts_aug1*sizes1,
                    dst_Kpts_aug1*sizes1, target_Kpts1, thetas_Kpts1)
        save_plots(im_size, curr_im2, new_im2, content_im2, style_im, output_dir, output_prefix, style_path, style_pts_path2, colors2,
                    src_Kpts2, src_Kpts_aug2, dst_Kpts2*sizes2, dst_Kpts_aug2, target_Kpts2, target_Kpts_o2, border_Kpts2, device)
        save_loss(output_dir, output_prefix, content_weight, warp_weight, reg_weight, max_iter, scale_list,
                    ell_list, ell_style_list2, ell_content_list2, ell_warp_list2, ell_warp_TV_list2)
        save_points(output_dir, output_prefix, src_Kpts2, dst_Kpts2*sizes2, src_Kpts_aug2*sizes2,
                    dst_Kpts_aug2*sizes2, target_Kpts2, thetas_Kpts2)
    # Return the stylized output image
    return new_im1, new_im2, pair_warp1, pair_warp2
