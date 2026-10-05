import csv
import numpy as np


def extract_csv_to_numpy(file_path, start_row=0, end_row=None, start_col=0, end_col=None, dtype=float):
    """
    从CSV文件中提取特定区域的数据并转换为NumPy数组

    参数:
        file_path: CSV文件路径
        start_row: 起始行索引(包含，从0开始)
        end_row: 结束行索引(不包含，None表示到最后)
        start_col: 起始列索引(包含，从0开始)
        end_col: 结束列索引(不包含，None表示到最后)
        dtype: 目标数据类型(如float, int, str等)，默认为float

    返回:
        NumPy数组形式的数据
    """
    data = []

    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)

        # 跳过起始行之前的所有行
        for _ in range(start_row):
            next(reader, None)

        # 读取指定行范围
        for i, row in enumerate(reader):
            if end_row is not None and start_row + i >= end_row:
                break

            # 提取指定列范围
            selected_row = row[start_col:end_col]

            # 尝试转换为指定类型
            try:
                if dtype != str:
                    converted_row = [dtype(x) if x.replace('.', '', 1).isdigit() else x for x in selected_row]
                else:
                    converted_row = selected_row
            except:
                converted_row = selected_row

            data.append(converted_row)

    # 转换为NumPy数组
    try:
        np_array = np.array(data, dtype=dtype)
    except:
        # 如果转换失败(例如混合数据类型)，使用对象类型
        np_array = np.array(data, dtype=object)

    return np_array

if __name__ == '__main__':

    float_data = extract_csv_to_numpy(r'C:\Users\LH\Desktop\SNN_MY\data_raw\IR_008.csv', start_row=16, end_row=208, start_col=0, end_col=256)
    print(float_data)