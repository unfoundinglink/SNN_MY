from dataloader_my import Dataset_SNN
import torch
import csv
from snntorch import surrogate, spikegen
import numpy as np
def tensor_to_csv(tensor, filename):
    """
    将二维PyTorch tensor写入CSV文件

    参数:
        tensor (torch.Tensor): 二维tensor矩阵
        filename (str): 要保存的CSV文件名

    返回:
        None
    """
    # 检查tensor是否为二维
    if tensor.dim() != 2:
        raise ValueError("输入tensor必须是二维矩阵")

    # 转换为numpy数组
    array = tensor.numpy()

    # 写入CSV文件
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(array)

    print(f"成功将tensor写入 {filename}")

def exponential_decay(x, a = 3.247, b = 0.065, c = 0.019):
    return a * np.exp(-b * x) + c


def reshape_3d_to_2d(matrix_3d):
    """
    将一个形状为 (200, 40, 40) 的三维矩阵转换为形状为 (1600, 200) 的二维矩阵

    参数:
    matrix_3d: numpy.ndarray, 形状为 (200, 40, 40) 的三维矩阵

    返回:
    numpy.ndarray: 形状为 (1600, 200) 的二维矩阵
    """
    # 首先将后两个维度展平 (40x40 -> 1600)
    # 然后转置以使原来的第一个维度(200)变成第二个维度
    return matrix_3d.reshape(200, 40 * 40).T


# 示例用法
if __name__ == "__main__":
    # 创建一个示例tensor
    dataset = Dataset_SNN(normal=False)
    img, label = dataset[401]
    print(label)
    print(img)
    # tensor_to_csv(img, "./output.csv")
    data_encoder =spikegen.rate(exponential_decay(img), num_steps=200)
    out_2d_encoder = reshape_3d_to_2d(data_encoder)
    # 调用函数写入CSV
    print(out_2d_encoder.shape)
    tensor_to_csv(out_2d_encoder,"./encoder.csv")
