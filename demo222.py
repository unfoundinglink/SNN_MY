import numpy as np


def scale_temperature_data(input_csv, output_csv):
    """
    读取CSV中的温度数据，缩放到22.1-48.3度范围并保存为新CSV

    参数:
        input_csv (str): 输入CSV文件路径
        output_csv (str): 输出CSV文件路径

    返回:
        numpy.ndarray: 缩放后的温度矩阵（保留3位小数）
    """
    try:
        # 1. 增强型文件读取（支持不规则分隔符）
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb18030']
        matrix = None

        for encoding in encodings:
            try:
                # 使用更健壮的读取方式处理不规则数据
                data = []
                with open(input_csv, 'r', encoding=encoding) as f:
                    for line in f:
                        # 移除BOM头和空白字符
                        line = line.strip().replace('\ufeff', '')
                        if line and not line.startswith(('#', '//')):
                            # 处理多种分隔符（逗号/分号/制表符）
                            row = [x.strip() for x in line.replace(';', ',').replace('\t', ',').split(',') if x]
                            try:
                                data.append([float(x) for x in row])
                            except ValueError:
                                print(f"忽略非数字行: {line}")
                                continue

                if data:
                    matrix = np.array(data)
                    break

            except UnicodeDecodeError:
                continue

        if matrix is None:
            print("错误: 无法解码文件或没有有效数据")
            return None

        print(f"原始矩阵形状: {matrix.shape} 数据范围: {np.min(matrix):.2f}~{np.max(matrix):.2f}℃")

        # 2. 温度缩放计算（22.1~48.3℃）
        original_min = np.min(matrix)
        original_max = np.max(matrix)
        TARGET_MIN = 22.1  # 精确下限
        TARGET_MAX = 48.3  # 精确上限

        if original_max == original_min:
            scaled_matrix = np.full_like(matrix, (TARGET_MIN + TARGET_MAX) / 2)
            print(f"警告: 所有温度值相同，设置为{(TARGET_MIN + TARGET_MAX) / 2:.3f}℃")
        else:
            # 向量化计算（优化性能）
            scaled_matrix = (matrix - original_min) * (TARGET_MAX - TARGET_MIN) / (
                        original_max - original_min) + TARGET_MIN
            scaled_matrix = np.round(scaled_matrix, 3)  # 保留3位小数

        # 边界精确保证
        scaled_matrix = np.clip(scaled_matrix, TARGET_MIN, TARGET_MAX)

        print(f"\n缩放后范围验证: {np.min(scaled_matrix):.3f}~{np.max(scaled_matrix):.3f}℃")

        # 3. 增强型CSV输出（保留小数位数）
        np.savetxt(
            output_csv,
            scaled_matrix,
            delimiter=',',
            fmt='%.3f',  # 保证3位小数
            header=f"Scaled Temperature ({TARGET_MIN}~{TARGET_MAX}℃)",
            comments=''
        )

        print(f"结果已保存到: {output_csv}")
        return scaled_matrix

    except Exception as e:
        print(f"处理失败: {str(e)}")
        return None


# 使用示例
if __name__ == "__main__":
    result = scale_temperature_data(
        r'C:\Users\LH\Desktop\SNN_MY\demo123456.csv',
        './scaled_temps_11111.csv'
    )

    if result is not None:
        print("处理成功！结果矩阵示例:")
        print(result[:3, :3])  # 显示前3行前3列