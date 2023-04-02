import shutil
import os
import random

# parameters


im_size = 512
device = 'cuda:1'
content_path = 'example/content.jpg'
style_path = 'example/style.jpg'
pts_path = 'example/NBBresults'

from run_style_pair import run_style_pair

def get_result(warp_lr, content_folder, style, lr, warp, reg, content_wei, style_wei, pairloss_flag, transform_flag):
    output_prefix = 'example'
    content_load_folder = 'example/content/' + content_folder
    style_folder = 'example/style/'

    content_path1 =  content_load_folder+ '/content1.jpg'
    content_path2 = content_load_folder + '/content2.jpg'
    content_pts_path1 = content_load_folder + '/content_pts1.txt'
    content_pts_path2 = content_load_folder + '/content_pts2.txt'
    matrixH1_path = content_load_folder + '/H1.txt'
    matrixH2_path = content_load_folder + '/H2.txt'
    point_pair1_path = content_load_folder + '/point2d1.txt'
    point_pair2_path = content_load_folder + '/point2d2.txt'
    point_pair_flag_path = content_load_folder + '/flag.txt'
    style_pts_path1 = style_folder + str(style) + '.txt'
    style_pts_path2 = style_folder + str(style) + '.txt'
    style_path = style_folder + str(style) + '.png'

    max_iter = 150
    checkpoint_iter = 50
    content_weight = content_wei
    style_weight = style_wei
    warp_weight = warp
    reg_weight = reg
    verbose = 1
    save_intermediate = 0
    save_extra = 0
    output_dir = 'output'
    im_size = 512
    optim_type = 'Adadelta'
    run_style_pair(warp_lr, pairloss_flag, transform_flag, content_path1, content_path2, style_path, content_folder,
                     style, content_pts_path1, content_pts_path2,
                     style_pts_path1, style_pts_path2, point_pair1_path, point_pair2_path, point_pair_flag_path,
                     matrixH1_path, matrixH2_path,
                     output_dir, output_prefix, im_size, max_iter, checkpoint_iter, content_weight, style_weight,
                     warp_weight,
                     reg_weight, optim_type, lr, verbose, save_intermediate, save_extra, device, )

if __name__ == '__main__':
    content_folder_list = []
    #torch.cuda.empty_cache()
    content_folder = 'example/content/'
    listx = os.listdir(content_folder)
    for i in range(0, len(listx)):
        content_folder_list.append(listx[i])
    style_list_all = list(range(1, 160))
    lr = 0.5
    reg = 30
    warp = 5
    content_wei = 0
    style_wei = 10
    warp_lr = 30
    for ind in range(0, len(content_folder_list)):
        content_folder = content_folder_list[ind]
        style_list = random.sample(style_list_all, 1)
        for style in style_list:
            get_result(warp_lr, content_folder, str(style), lr, warp, reg, content_wei, style_wei, pairloss_flag=1, transform_flag=1)