# -*- coding:utf-8 -*-
import cv2
import numpy as np


def rad(x):
    return x * np.pi / 180


def rotate_3(img, keypoint, angle_vari=30):
    img = img[0,:,:]
    img = np.array(img.cpu())
    img = np.transpose(img, (1, 2, 0))
    w, h = img.shape[0:2]
    fov = 42
    #anglex = np.random.uniform(0, 0)
    #angley = np.random.uniform(-angle_vari, -angle_vari)
    #anglez = np.random.uniform(0, 0)
    anglex = 0
    angley = -angle_vari
    anglez = 0
    # 镜头与图像间的距离，21为半可视角，算z的距离是为了保证在此可视角度下恰好显示整幅图像
    z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))
    # 齐次变换矩阵
    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0],
                   [0, -np.sin(rad(anglex)), np.cos(rad(anglex)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
                   [0, 1, 0, 0],
                   [-np.sin(rad(angley)), 0, np.cos(rad(angley)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], np.float32)

    r = rx.dot(ry).dot(rz)

    # 四对点的生成
    pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)

    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter

    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)

    list_dst = [dst1, dst2, dst3, dst4]

    org = np.array([[0, 0],
                    [w, 0],
                    [0, h],
                    [w, h]], np.float32)

    dst = np.zeros((4, 2), np.float32)

    # 投影至成像平面
    for i in range(4):
        dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
        dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]

    warpR = cv2.getPerspectiveTransform(org, dst)

    result = cv2.warpPerspective(img, warpR, (h, w))

    #style_pts1 = np.loadtxt("example/style/0.txt", delimiter=',')
    #style_pts1 = org
    style_pts1 = np.expand_dims(keypoint, axis=0)
    warp_pts1 = cv2.perspectiveTransform(style_pts1, warpR)
    #plot_image_keypoints(img, style_pts1)

    #plot_image_keypoints(result, warp_pts1[0,:,:])
    result = np.transpose(result, (2, 0, 1))
    result = np.expand_dims(result, axis= 0)


    return result, warp_pts1[0,:,:]


def index_transfer(pts_in):
    temp = pts_in[:,0].copy()
    pts_in[:,0] = pts_in[:,1]
    pts_in[:,1] = temp
    return pts_in

def plot_image_keypoints(img, keypoints):

    for i in range(0,68):
        cv2.circle(img, (int(keypoints[i][1]), int(keypoints[i][0])), 4, (0, 255, 0), -1, 8)
        #cv2.putText(img, str('1'), (int(keypoints[i][1]), int(keypoints[i][0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))


    cv2.imshow('Frame', img)
    cv2.waitKey(0)


def rotate(image, angle_vari=30):
    angle = np.random.uniform(-angle_vari, angle_vari)
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    dst = cv2.warpAffine(image, M, (cols, rows))
    return dst


if __name__ == '__main__':
    img = cv2.imread("example/style/0.png")
    angle_vari = 30
    keypoint = None
    while True:
        result = rotate_3(img, keypoint, angle_vari=angle_vari)
        cv2.imshow("result", result)
        c = cv2.waitKey()