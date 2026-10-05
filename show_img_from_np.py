import numpy as np
import csv
import cv2
import matplotlib.pyplot as plt
import pickle
import os
import logging

def show_cv_heatmap(matrix, window_name="Heatmap", resize_scale=1.0,index_1 = 0, path = "./" ,save = False ,show =True):
    """
    用OpenCV显示二维numpy数组的热图（红高蓝低）

    参数:
        matrix: 输入的二维numpy数组
        window_name: 窗口名称 (默认"Heatmap")
        resize_scale: 图像缩放比例 (默认1.0不缩放)
    """
    # 归一化到0-255范围
    normalized = ((matrix - matrix.min()) / (matrix.max() - matrix.min()) * 255).astype(np.uint8)

    # 应用红蓝渐变色（COLORMAP_JET）
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    # 始终获取原始尺寸（无论是否缩放）
    h, w = heatmap.shape[:2]

    # 仅当需要缩放时执行resize
    if resize_scale != 1.0:
        heatmap = cv2.resize(heatmap, (int(w * resize_scale), int(h * resize_scale)))

    # 显示图像
    if(show):
        cv2.imshow(window_name, heatmap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if(save):
        cv2.imwrite(path + "/{}.png".format(index_1),heatmap)



if __name__ == '__main__':

    from dataloader_my import Dataset_SNN
    # with open(r"C:\Users\LH\Desktop\SNN_MY\data_snn_my\data_1\data_origin_plk\origin_data.pkl","rb") as f:
    #     data = pickle.load(f)

    # print(len(data))
    dataset = Dataset_SNN(normal = True)
    img, label= dataset[400]
    print(label)
    show_cv_heatmap(np.array(img))
