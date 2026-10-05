import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate, spikegen
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

num_steps = 200  # 时间步长
beta = 0.95  # LIF神经元的衰减因子


class SNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 编码层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(40 * 40, 512)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        # 隐藏层
        self.fc2 = nn.Linear(512, 256)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        # 输出层
        self.fc3 = nn.Linear(256, 7)
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


class ConvSNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 卷积层 + 脉冲神经元
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)  # 输入通道1, 输出通道3
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        # 池化层
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(3, 6, kernel_size=3)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        # 池化层
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三个卷积层
        self.conv3 = nn.Conv2d(6, 1, kernel_size=3)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())


        # 全连接层
        self.fc1 = nn.Linear(36, 72)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), output=True)

        self.fc2 = nn.Linear(72, 3)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), output=True)


    def forward(self, x):
        # 初始化膜电位
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif3.init_leaky()
        mem5 = self.lif3.init_leaky()

        spk_rec = []

        for step in range(num_steps):
            # 卷积层1 + LIF神经元
            # x_step = x[:, step, ...] if x.dim() == 5 else x  # 处理时间维度
            x_step = x[step] if step < x.shape[0] else 0
            cur1 = self.conv1(x_step)
            spk1, mem1 = self.lif1(cur1, mem1)

            # 池化
            pooled1 = self.pool1(spk1)

            # 卷积层2 + LIF神经元
            cur2 = self.conv2(pooled1)
            spk2, mem2 = self.lif2(cur2, mem2)

            pooled2 = self.pool2(spk2)

            cur3 = self.conv3(pooled2)
            spk3, mem3 = self.lif3(cur3, mem3)


            # 展平 + 全连接
            flattened = torch.flatten(spk3, 1)
            cur4 = self.fc1(flattened)
            spk4, mem4 = self.lif4(cur4, mem4)

            cur5 = self.fc2(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)

            spk_rec.append(spk5)

        return torch.stack(spk_rec, dim=0)  # [time_steps, batch, 10]


if __name__ == '__main__':

    from dataloader_my import Dataset_SNN

    dataset = Dataset_SNN()
    img ,label = dataset[0]


    spike_data = spikegen.rate(torch.tensor(img,dtype=torch.float32).unsqueeze(0), num_steps=num_steps)  # shape: [num_steps, batch_size, 784]
    # spike_data = spike_data.unsqueeze(0)
    # print(spike_data)
    print(spike_data.shape)

    # # print(data.shape)
    snn = SNNModel()
    print(snn(spike_data))
    print(snn(spike_data).shape)
    # csnn = ConvSNN()
    #
    # print(csnn(spike_data))
    # print(csnn(spike_data).shape)