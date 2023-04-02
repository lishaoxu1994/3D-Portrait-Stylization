import imp
import numpy as np
import torch
def point_update(keypoints1, keypoints2):
    matrix_pixel_x = np.zeros((512,512), dtype=np.int)
    matrix_pixel_y = np.zeros((512,512), dtype=np.int)
    matrix_flag = np.zeros((512,512), dtype=np.int)
    keypoints1 = keypoints1.numpy().astype(int)
    keypoints2 = keypoints2.numpy().astype(int)

    for ind in range(0,len(keypoints1)):
        if(keypoints1[ind,0]>511 or keypoints1[ind,1]>511 or keypoints2[ind,0]>511 or keypoints2[ind,0]>511):
            continue
        matrix_flag[keypoints1[ind,0],keypoints1[ind,1]] = matrix_flag[keypoints1[ind,0],keypoints1[ind,1]] + 1
        matrix_pixel_x[keypoints1[ind,0],keypoints1[ind,1]] = matrix_pixel_x[keypoints1[ind,0],keypoints1[ind,1]] + keypoints2[ind,0]
        matrix_pixel_y[keypoints1[ind,0],keypoints1[ind,1]] = matrix_pixel_y[keypoints1[ind,0],keypoints1[ind,1]] + keypoints2[ind,1]
    keypoints1_out = []
    keypoints2_out = []
    for ind_x in range(0,512):
        for ind_y in range(0,512):
            if matrix_flag[ind_x, ind_y] != 0:
                times = matrix_flag[ind_x, ind_y]
                keypoints1_out.append([ind_x,ind_y])
                keypoints2_out.append([matrix_pixel_x[ind_x,ind_y]/times, matrix_pixel_y[ind_x,ind_y]/times])
    #keypoints1_out = np.array(keypoints1_out)
    #keypoints2_out = np.array(keypoints2_out)
    keypoints1_out = torch.tensor(keypoints1_out).float()
    keypoints2_out = torch.tensor(keypoints2_out).float()
    error = keypoints1_out - keypoints2_out

    return keypoints1_out,keypoints2_out