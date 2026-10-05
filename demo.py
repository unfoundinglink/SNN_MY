import csv
import numpy as np


def process_and_save_temperature_matrix(input_csv, output_csv):
    """
    读取CSV中的矩阵，进行左右镜像并将温度数据缩放到25-45度范围，然后保存为新CSV

    参数:
        input_csv (str): 输入CSV文件路径
        output_csv (str): 输出CSV文件路径

    返回:
        numpy.ndarray: 处理后的矩阵
    """
    try:
        # 1. 读取CSV文件（尝试多种编码方式）
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5']
        matrix = None

        for encoding in encodings:
            try:
                with open(input_csv, 'r', encoding=encoding) as f:
                    reader = csv.reader(f)
                    # 跳过可能的空行和注释行
                    matrix = [list(map(float, row)) for row in reader if row and not row[0].startswith('#')]
                break
            except UnicodeDecodeError:
                continue
            except ValueError as e:
                print(f"数据格式错误（编码尝试: {encoding}）: {e}")
                return None

        if matrix is None:
            print("错误: 无法解码文件，尝试了所有编码方式")
            return None

        matrix = np.array(matrix)
        print("原始矩阵:")
        print(matrix)

        # 2. 左右镜像
        mirrored_matrix = np.fliplr(matrix)
        print("\n左右镜像后的矩阵:")
        print(mirrored_matrix)

        # 3. 温度数据缩放到25-45度
        min_temp = matrix.min()
        max_temp = matrix.max()

        # 避免除以零(当所有温度值相同时)
        if max_temp == min_temp:
            scaled_matrix = np.full_like(mirrored_matrix, 35)  # 取中间值
        else:
            scaled_matrix = ((mirrored_matrix - min_temp) / (max_temp - min_temp)) * (45 - 25) + 25

        scaled_matrix = np.round(scaled_matrix, 2)  # 保留两位小数

        print("\n缩放后的矩阵(25-45度):")
        print(scaled_matrix)

        # 4. 保存为新CSV文件
        try:
            with open(output_csv, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(scaled_matrix)

            print(f"\n处理后的矩阵已保存到: {output_csv}")
            return scaled_matrix
        except IOError as e:
            print(f"无法写入输出文件: {e}")
            return None

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return None


# 使用示例
processed = process_and_save_temperature_matrix(
    r'C:\Users\LH\Desktop\SNN_MY\demo123456.csv',
    r'C:\Users\LH\Desktop\SNN_MY\output_demo.csv'
)

if processed is not None:
    print("处理成功！")
else:
    print("处理失败！")