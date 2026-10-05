import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader
from snntorch import surrogate, spikegen
def save_confusion_matrix_as_csv(
    model,
    dataloader,
    class_names,
    output_csv_path="confusion_matrix.csv",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    计算模型的混淆矩阵并保存为CSV文件（PyTorch版本）

    参数:
        model: 训练好的PyTorch模型
        dataloader: 数据加载器 (torch.utils.data.DataLoader)
        class_names: 类别名称列表 (如 ["cat", "dog"])
        output_csv_path: 输出CSV文件路径 (默认: "confusion_matrix.csv")
        device: 计算设备 (默认: "cuda" 或 "cpu")
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = spikegen.rate(inputs, num_steps=200)
            outputs = model(inputs)
            # _, preds = torch.max(outputs, 1)
            _, preds = outputs .sum(dim=0).max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 转换为DataFrame并添加行列标签
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    # 保存为CSV
    cm_df.to_csv(output_csv_path)
    print(f"混淆矩阵已保存至 {output_csv_path}")

    return cm_df


def save_normalized_confusion_matrix(
        model,
        dataloader,
        class_names,
        output_csv_path="confusion_matrix_normalized.csv",
        device="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    计算模型的归一化混淆矩阵并保存为CSV文件（PyTorch版本）

    参数:
        model: 训练好的PyTorch模型
        dataloader: 数据加载器 (torch.utils.data.DataLoader)
        class_names: 类别名称列表 (如 ["cat", "dog"])
        output_csv_path: 输出CSV文件路径 (默认: "confusion_matrix_normalized.csv")
        device: 计算设备 (默认: "cuda" 或 "cpu")
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = spikegen.rate(inputs, num_steps=200)
            outputs = model(inputs)
            # _, preds = torch.max(outputs, 1)
            _, preds = outputs .sum(dim=0).max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 生成原始混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 归一化混淆矩阵（按行）
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.round(cm_normalized, 4)  # 保留4位小数

    # 将原始值和归一化值合并为一个DataFrame
    cm_combined = pd.DataFrame(
        data=np.concatenate([cm, cm_normalized], axis=1),
        index=class_names,
        columns=[f"{cls}_raw" for cls in class_names] + [f"{cls}_norm" for cls in class_names]
    )

    # 保存为CSV
    cm_combined.to_csv(output_csv_path)
    print(f"归一化混淆矩阵已保存至 {output_csv_path}")

    return cm_combined


if __name__ == '__main__':
    from model import SNNModel
    from dataloader_my import Dataset_SNN
    class_names = ["45°", "90°", "135°","180°","225°","270°","315°"]  # 你的类别名称
    model = SNNModel()
    model.load_state_dict(torch.load(r'C:\Users\LH\Desktop\SNN_MY\model_save_1\model_weights_119_99.75%.pth'))
    test_dataset = Dataset_SNN(normal= True)
    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 调用函数
    # save_confusion_matrix_as_csv(
    #     model=model,
    #     dataloader=dataloader,
    #     class_names=class_names,
    #     output_csv_path="torch_confusion_matrix.csv"
    # )
    save_normalized_confusion_matrix(
        model=model,
        dataloader=dataloader,
        class_names=class_names,
        output_csv_path="normalized_cm_1.csv"
    )