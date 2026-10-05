import torch
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import Lambda


class GaussianNoiseDataset(Dataset):
    """
    通过添加高斯噪声扩展原始数据集的Dataset类

    参数:
        original_dataset: 原始数据集对象
        multiplier: 要扩展到的倍数(包括原始数据)
        mean: 高斯噪声均值 (默认0)
        std_range: 噪声标准差的范围 (min, max)
        apply_prob: 对每个样本应用噪声的概率 (默认1.0)
    """

    def __init__(self, original_dataset, multiplier, mean=0., std_range=(0.01, 0.1), apply_prob=1.0):
        self.original_dataset = original_dataset
        self.multiplier = multiplier
        self.mean = mean
        self.std_range = std_range
        self.apply_prob = apply_prob

        # 计算需要生成的额外样本数
        self.extra_samples = len(original_dataset) * (multiplier - 1)

    def __len__(self):
        return self.extra_samples

    def __getitem__(self, idx):
        # 计算对应的原始样本索引
        original_idx = idx % len(self.original_dataset)
        image, label = self.original_dataset[original_idx]

        # 决定是否添加噪声
        if torch.rand(1).item() < self.apply_prob:
            # 随机选择噪声强度
            std = torch.empty(1).uniform_(*self.std_range).item()

            # 添加高斯噪声
            noise = torch.randn_like(image) * std + self.mean
            noisy_image = image + noise

            # 确保像素值在合法范围内(假设图像已经归一化到[0,1]或[-1,1])
            noisy_image = torch.clamp(noisy_image, image.min(), image.max())
        else:
            noisy_image = image.clone()

        return noisy_image, label


def expand_dataset_with_noise(original_dataset, multiplier, mean=0.01, std_range=(0.01, 0.1), apply_prob=1.0):
    """
    通过添加高斯噪声扩展数据集

    参数:
        original_dataset: 原始数据集
        multiplier: 要扩展到的总倍数(例如3表示原始+2倍新样本)
        mean: 噪声均值
        std_range: 噪声标准差范围元组(min, max)
        apply_prob: 应用噪声的概率

    返回:
        合并后的数据集(原始+噪声增强样本)
    """
    # 创建噪声数据集
    noise_dataset = GaussianNoiseDataset(
        original_dataset=original_dataset,
        multiplier=multiplier,
        mean=mean,
        std_range=std_range,
        apply_prob=apply_prob
    )

    # 合并原始数据集和噪声数据集
    expanded_dataset = ConcatDataset([original_dataset, noise_dataset])

    return expanded_dataset


# 使用示例 --------------------------------------------------

if __name__ == "__main__":
    # 示例: 使用MNIST数据集
    from torchvision.datasets import MNIST
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from show_img_from_np import show_cv_heatmap
    from dataloader_my import Dataset_SNN_train, Dataset_SNN_test,Dataset_SNN
    data_set = Dataset_SNN(normal= True)
    print(f"原始数据集大小: {len(data_set)}")

    # 2. 扩展数据集(扩展到3倍: 原始 + 2倍噪声样本)
    expanded_dataset = expand_dataset_with_noise(
        original_dataset=data_set,
        multiplier=9,
        std_range=(0.00, 0.02),  # 噪声强度范围
        apply_prob=1  # 80%的样本添加噪声
    )

    print(f"扩展后数据集大小: {len(expanded_dataset)}")


    # 3. 可视化示例



    # 查看原始图像和噪声图像
    original_img, _ = data_set[0]
    noisy_img1, _ = expanded_dataset[len(data_set)]  # 第一个噪声样本
    noisy_img2, _ = expanded_dataset[len(data_set) + 1]  # 第二个噪声样本

    print("原始图像:",len(data_set))
    show_cv_heatmap(np.array(original_img))
    print("噪声增强图像1:",len(expanded_dataset))
    show_cv_heatmap(np.array(noisy_img1))
    print("噪声增强图像2:",len(expanded_dataset))
    show_cv_heatmap(np.array(noisy_img2))