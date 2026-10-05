import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools
import os

# 网络结构
num_inputs = 28 * 28
num_hidden = 1000
num_outputs = 10

# 时间参数
num_steps = 25
beta = 0.95


# 定义网络结构
class SNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 编码层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        # 隐藏层
        self.fc2 = nn.Linear(512, 256)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        # 输出层
        self.fc3 = nn.Linear(256, 10)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        # 初始化膜电位
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # 存储输出脉冲
        spk_out_rec = []

        # 模拟时间步
        for step in range(num_steps):
            # 输入编码
            cur_input = x[step] if step < x.shape[0] else 0
            flat_input = self.flatten(cur_input)
            fc1_out = self.fc1(flat_input)
            spk1, mem1 = self.lif1(fc1_out, mem1)

            # 隐藏层
            fc2_out = self.fc2(spk1)
            spk2, mem2 = self.lif2(fc2_out, mem2)

            # 输出层
            fc3_out = self.fc3(spk2)
            spk3, mem3 = self.lif3(fc3_out, mem3)
            spk_out_rec.append(spk3)

        return torch.stack(spk_out_rec, dim=0)

if __name__ == '__main__':

    # Load the network onto CUDA if available
    num_steps = 100
    spike_data = spikegen.rate(torch.tensor([0.1,0.6]), num_steps=num_steps)
    print(spike_data.shape)

    # spk,mem =net(img.view(batch_size, -1))
    # print("---------")
    # print(spk)
    # print("---------")
    # print(mem)
    # print("---------")
    # print(label)
