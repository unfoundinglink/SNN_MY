import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate, spikegen
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from model import  SNNModel
from dataloader_my import Dataset_SNN
# 参数设置

import torch
import csv


def tensor_to_csv(tensor, filename, headers=None):
    """
    将形状为(200, 1, 7)的PyTorch张量写入CSV文件

    参数:
        tensor: 输入张量，形状应为(200, 1, 7)
        filename: 要保存的CSV文件名
        headers: 可选的列标题列表

    返回:
        None
    """
    # 检查输入张量的形状
    if tensor.dim() != 3 or tensor.shape != (200, 1, 7):
        raise ValueError("输入张量的形状必须是(200, 1, 7)")

    # 处理梯度问题并转换为numpy数组
    if tensor.requires_grad:
        array = tensor.detach().squeeze(1).numpy()  # 先detach再转换
    else:
        array = tensor.squeeze(1).numpy()

    # 写入CSV文件
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 写入列标题(如果有)
        if headers is not None:
            if len(headers) != 7:
                raise ValueError("标题数量必须与列数(7)匹配")
            writer.writerow(headers)

        # 写入数据
        writer.writerows(array)

    print(f"数据已成功写入 {filename}")




if __name__ == '__main__':

    num_steps = 200  # 时间步长
    beta = 0.95      # LIF神经元的衰减因子

    # 定义SNN模型

    my_snn_model = SNNModel()
    my_snn_model.load_state_dict(torch.load(r'C:\Users\LH\Desktop\SNN_MY\model_save_1\model_weights_119_99.75%.pth'))
    my_snn_model.eval()  # 设置为评估模式

    test_dataset = Dataset_SNN(normal= True)
    img, label = test_dataset[200]  # 取第38张测试图片

    # 生成脉冲编码
    spike_data = spikegen.rate(img.unsqueeze(0), num_steps=num_steps)  # 形状 (25, 1, 28, 28)
    print(spike_data.shape)
    # 前向传播
    spk_rec = my_snn_model(spike_data)  # 形状 (25, 10)
    _, predicted = spk_rec.sum(dim=0).max(1)
    print(spk_rec.shape)

    print(predicted)
    column_headers = ['angle_45', 'angle_90', 'angle_135', 'angle_180',
                     'angle_225', 'angle_270', 'angle_315']
    tensor_to_csv(spk_rec, "output_data_2.csv",headers=column_headers)
