import numpy as np
import torch

from snntorch import surrogate, spikegen
def exponential_decay(x, a=3.247, b=0.065, c=0.019):
    return a * np.exp(-b * x) + c


import torch
import csv
from pathlib import Path


def write_tensors_to_csv(tensor_list, csv_path, headers=None):
    """
    将多个Tensor序列写入CSV文件

    参数:
        tensor_list: 包含五个Tensor的列表，每个Tensor形状为[序列长度, 特征维度]
        csv_path: 输出CSV文件路径
        headers: CSV文件的列名列表
    """
    # 确保输入是五个Tensor
    assert len(tensor_list) == 5, "需要 exactly 五个Tensor"

    # 获取最大序列长度
    max_len = max(tensor.shape[0] for tensor in tensor_list)

    # 准备数据
    data_to_write = []
    for i in range(max_len):
        row = []
        for tensor in tensor_list:
            if i < tensor.shape[0]:
                # 将Tensor数据转换为列表
                if tensor.ndim == 1:
                    row.extend([tensor[i].item()])
                else:
                    row.extend(tensor[i].tolist())
            else:
                # 用空值填充较短的序列
                if tensor.ndim == 1:
                    row.extend([''])
                else:
                    row.extend([''] * tensor.shape[1])
        data_to_write.append(row)

    # 写入CSV文件
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # 写入表头（如果提供）
        if headers:
            writer.writerow(headers)

        # 写入数据
        writer.writerows(data_to_write)

if __name__ == '__main__':
    encode_data_25 = exponential_decay(25)
    encode_data_25 =torch.tensor(encode_data_25)
    data_25 = spikegen.rate(encode_data_25, num_steps=200)



    encode_data_30 = exponential_decay(30)
    encode_data_30 =torch.tensor(encode_data_30)
    data_30 = spikegen.rate(encode_data_30, num_steps=200)


    encode_data_35 = exponential_decay(35)
    encode_data_35 =torch.tensor(encode_data_35)
    data_35 = spikegen.rate(encode_data_35, num_steps=200)


    encode_data_40 = exponential_decay(40)
    encode_data_40 =torch.tensor(encode_data_40)
    data_40 = spikegen.rate(encode_data_40, num_steps=200)


    encode_data_45 = exponential_decay(45)
    encode_data_45 =torch.tensor(encode_data_45)
    data_45 = spikegen.rate(encode_data_45, num_steps=200)

    print(data_25,data_30,data_35,data_40,data_45)
    tensor_list = [data_25,data_30,data_35,data_40,data_45]
    headers = []
    for i, tensor in enumerate(tensor_list, 1):
        if tensor.ndim == 1:
            headers.append(f'Tensor{i}_Feature1')
        else:
            for j in range(tensor.shape[1]):
                headers.append(f'Tensor{i}_Feature{j + 1}')

    # 写入CSV文件
    write_tensors_to_csv(tensor_list, 'tensor_sequences.csv', headers)
    print("Tensor数据已成功写入 tensor_sequences.csv")