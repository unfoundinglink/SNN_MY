import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate, spikegen
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import SNNModel ,ConvSNN
from dataloader_my import Dataset_SNN_train,Dataset_SNN_test
from extend_dataset import expand_dataset_with_noise
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit


from torch.utils.data import DataLoader, random_split
batch_size = 96
num_epochs = 120
learning_rate = 1e-5
num_steps = 200  # 仿真时间步
beta = 0.95



# 训练函数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SNNModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# torch.manual_seed(42)
# 3. 训练函数
def train(model, loader, optimizer, epoch):
    model.train()
    losses = []

    for batch_idx, (data, targets) in enumerate(loader):
        data, targets = data.to(device), targets.to(device)
        # data = data.unsqueeze(1)
        data = spikegen.rate(data,num_steps=200)
        # print(data.shape)
        # 前向传播
        spk_rec = model(data)

        # 计算损失（取最后时间步的输出）
        loss = criterion(spk_rec[-1], targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    return torch.tensor(losses).mean().item()


# 4. 测试函数
def test(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            # data = data.unsqueeze(1)
            data = spikegen.rate(data, num_steps=200)
            spk_rec = model(data)

            # 统计正确率（取脉冲计数最多的类别）
            _, predicted = spk_rec.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()


    return 100 * correct / total


dataset_test = Dataset_SNN_train(normal=True)
dataset_train = expand_dataset_with_noise(
        original_dataset=dataset_test,
        multiplier=9,
        std_range=(0.00, 0.02),  # 噪声强度范围
        apply_prob=1  # 80%的样本添加噪声
    )


train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)



# 5. 训练循环
train_losses = []
test_accs = []

for epoch in range(num_epochs):
    avg_loss = train(model, train_loader, optimizer, epoch)
    acc = test(model, test_loader)


    train_losses.append(avg_loss)
    test_accs.append(acc)
    with open("./acc_rate_3.txt","a") as f:
        f.write(str(acc) + "\n")
    with open("./acc_rate_loss_3.txt","a") as f:
        f.write(str(avg_loss) + "\n")

    torch.save(model.state_dict(), f'./model_save_3/model_weights_{epoch}_{acc :.2f}%.pth')

    print(f"Epoch {epoch}, Test Acc: {acc:.2f}%")

# 6. 可视化结果
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")

plt.subplot(1, 2, 2)
plt.plot(test_accs)
plt.title("Test Accuracy")
plt.xlabel("Epoch")
plt.show()