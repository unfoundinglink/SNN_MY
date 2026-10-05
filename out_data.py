import pickle

import numpy as np
import csv

def numpy_to_csv(array, filename, delimiter=',', header=None, fmt='%.18e'):
    """
    将二维NumPy数组写入CSV文件

    参数:
        array (np.ndarray): 输入的二维NumPy数组
        filename (str): 输出的CSV文件路径（如 'data.csv'）
        delimiter (str): 列分隔符（默认逗号）
        header (list/None): 列标题列表（如 ['col1', 'col2']）
        fmt (str): 数字格式（默认科学计数法，可选 '%.3f' 保留3位小数）
    """
    # 检查输入是否为二维数组
    if array.ndim != 2:
        raise ValueError("输入数组必须是二维的")

    # 使用NumPy的savetxt写入CSV
    np.savetxt(
        filename,
        array,
        delimiter=delimiter,
        header=delimiter.join(header) if header else '',
        comments='',  # 避免header前自动添加#
        fmt=fmt
    )
if __name__ == '__main__':

    with open(r"C:\Users\LH\Desktop\SNN_MY\data_snn_my\data_2\data_fin_plk\fin_data.pkl", "rb") as f:
        data = pickle.load(f)

    print(type(data[2]))
    numpy_to_csv(data[3], './output901.csv')