import pandas as pd


def extract_csv_section(input_file, output_file, column_name, start_value, end_value):
    """
    截取CSV文件中指定列的从start_value到end_value的数据

    参数:
    input_file: 输入CSV文件路径
    output_file: 输出CSV文件路径
    column_name: 作为标准的列名
    start_value: 起始值
    end_value: 结束值
    """

    # 读取CSV文件，指定分隔符为制表符
    df = pd.read_csv(input_file, sep='\t')

    # 检查列名
    print("CSV文件列名:", df.columns.tolist())

    # 确保指定的列存在
    if column_name not in df.columns:
        raise ValueError(f"列 '{column_name}' 不存在于CSV文件中。可用的列有: {df.columns.tolist()}")

    # 截取指定范围的数据
    mask = (df[column_name] >= start_value) & (df[column_name] <= end_value)
    filtered_df = df.loc[mask]

    # 保存到新的CSV文件
    filtered_df.to_csv(output_file, index=False)
    print(f"数据已成功截取并保存到 {output_file}")
    print(f"原始数据行数: {len(df)}")
    print(f"截取后数据行数: {len(filtered_df)}")


# 使用示例
if __name__ == "__main__":
    # 设置参数
    input_csv = r"C:\Users\LH\Desktop\SNN_MY\test\find.csv"  # 输入文件路径
    output_csv = "output1.csv"  # 输出文件路径
    target_column = "Time[s]"  # 作为标准的列名
    start_val = 2.38  # 起始值
    end_val = 2.47  # 结束值

    # 执行截取操作
    extract_csv_section(input_csv, output_csv, target_column, start_val, end_val)