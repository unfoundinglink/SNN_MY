import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate, spikegen
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
batch_size = 128
num_epochs = 10
learning_rate = 1e-3
num_steps = 25  # 时间步长
beta = 0.9  # LIF神经元的衰减因子


class SNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 编码层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2, 20)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        # 隐藏层
        self.fc2 = nn.Linear(20, 10)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        # 输出层
        self.fc3 = nn.Linear(10, 3)
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
    # a = torch.rand(1, 1, 2)
    a = torch.tensor([[[0.1,0.9]]])
    print(a)
    spike_data = spikegen.rate(a, num_steps=num_steps)  # shape: [num_steps, batch_size, 784]

    print(spike_data)
    print(spike_data.shape)
    # # data = spike_data.unsqueeze(1)
    # # print(data.shape)
    snn = SNNModel()
    print(snn(spike_data).shape)