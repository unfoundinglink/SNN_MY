import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate, spikegen
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 超参数设置
batch_size = 20
num_epochs = 25
learning_rate = 1e-3
num_steps = 25  # 时间步长
beta = 0.9  # LIF神经元的衰减因子

# 数据加载和预处理
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])
# train_dataset = MNIST_RGB_Dataset_test()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义SNN模型
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


# 初始化模型、损失函数和优化器
model = SNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 训练函数
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for data, targets in loader:
        optimizer.zero_grad()

        # 生成脉冲输入（泊松编码）
        # print("------------------------------------")
        # print(data)
        # print(data.shape)
        # print("------------------------------------")
        spike_data = spikegen.rate(data, num_steps=num_steps)

        # 前向传播
        # print("------------------------------------")
        # print(spike_data)
        # print(spike_data.shape)
        # print("------------------------------------")
        spk_rec = model(spike_data)

        # 计算损失（取最后一个时间步的脉冲）
        loss = criterion(spk_rec[-1], targets)
        total_loss += loss.item()

        # 反向传播
        loss.backward()
        optimizer.step()

    return total_loss / len(loader)


# 测试函数
def test(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in loader:
            spike_data = spikegen.rate(data, num_steps=num_steps)
            spk_rec = model(spike_data)

            # 预测类别（脉冲计数最多）
            _, predicted = torch.max(spk_rec.sum(dim=0), 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return correct / total


# 训练和测试
train_losses = []
test_accs = []

for epoch in range(num_epochs):
    loss = train(model, train_loader, optimizer, criterion)
    acc = test(model, test_loader)

    train_losses.append(loss)
    test_accs.append(acc)
    with open("./acc_rate.txt","a") as f:
        f.write(str(acc) + "\n")
    with open("./acc_rate_loss.txt","a") as f:
        f.write(str(loss) + "\n")

    torch.save(model.state_dict(), f'./model_save/model_weights_{epoch}_{acc * 100:.2f}%.pth')

    # # 加载时需先实例化模型
    # new_model = SimpleModel()  # 必须与保存时的结构相同
    # new_model.load_state_dict(torch.load('model_weights.pth'))
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Test Acc: {acc * 100:.2f}%")

# 可视化训练结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(test_accs, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()