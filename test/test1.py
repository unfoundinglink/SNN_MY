import pandas as pd
import numpy as np


def find_subsequence_position(y_sequence, x_sequence):
    """
    在y序列中查找x子序列的位置

    Parameters:
    y_sequence (list/array): Y列的序列
    x_sequence (list/array): X列的序列

    Returns:
    dict: 包含开始位置和结束位置的信息，如果没找到返回None
    """
    y_len = len(y_sequence)
    x_len = len(x_sequence)

    if x_len == 0 or y_len == 0 or x_len > y_len:
        return None

    # 在Y序列中查找X序列
    for i in range(y_len - x_len + 1):
        # 检查从位置i开始的子序列是否匹配X序列
        if np.array_equal(y_sequence[i:i + x_len], x_sequence):
            return {
                'start_position': i + 2,  # Excel行号（从1开始，+1表头）
                'end_position': i + x_len + 1,
                'match_length': x_len
            }

    return None


def find_x_in_y_excel(file_path, x_sheet=0, x_col=0, y_sheet=0, y_col=1):
    """
    在Excel文件的Y列中查找X列作为子序列的位置

    Parameters:
    file_path (str): Excel文件路径
    x_sheet: X列所在的工作表索引或名称
    x_col: X列的索引或列名
    y_sheet: Y列所在的工作表索引或名称
    y_col: Y列的索引或列名

    Returns:
    dict: 包含匹配位置的信息
    """
    try:
        # 读取X列数据
        x_df = pd.read_excel(file_path, sheet_name=x_sheet)
        if isinstance(x_col, str) and x_col in x_df.columns:
            x_values = x_df[x_col].dropna().values
        else:
            x_values = x_df.iloc[:, x_col].dropna().values

        # 读取Y列数据
        y_df = pd.read_excel(file_path, sheet_name=y_sheet)
        if isinstance(y_col, str) and y_col in y_df.columns:
            y_values = y_df[y_col].dropna().values
        else:
            y_values = y_df.iloc[:, y_col].dropna().values

        print(f"X序列: {x_values}")
        print(f"Y序列: {y_values}")
        print(f"X序列长度: {len(x_values)}")
        print(f"Y序列长度: {len(y_values)}")

        # 在Y中查找X子序列
        result = find_subsequence_position(y_values, x_values)

        if result:
            return {
                'x_sequence': x_values.tolist(),
                'y_sequence': y_values.tolist(),
                'start_position': result['start_position'],
                'end_position': result['end_position'],
                'match_length': result['match_length'],
                'found': True
            }
        else:
            return {
                'x_sequence': x_values.tolist(),
                'y_sequence': y_values.tolist(),
                'found': False,
                'message': "在Y列中未找到完整的X序列"
            }

    except Exception as e:
        return {
            'found': False,
            'error': str(e)
        }


def find_all_occurrences(file_path, x_sheet=0, x_col=0, y_sheet=0, y_col=1):
    """
    查找X序列在Y序列中的所有出现位置
    """
    try:
        # 读取数据
        x_df = pd.read_excel(file_path, sheet_name=x_sheet)
        y_df = pd.read_excel(file_path, sheet_name=y_sheet)

        if isinstance(x_col, str):
            x_values = x_df[x_col].dropna().values
        else:
            x_values = x_df.iloc[:, x_col].dropna().values

        if isinstance(y_col, str):
            y_values = y_df[y_col].dropna().values
        else:
            y_values = y_df.iloc[:, y_col].dropna().values

        x_len = len(x_values)
        y_len = len(y_values)

        occurrences = []

        # 查找所有匹配位置
        for i in range(y_len - x_len + 1):
            if np.array_equal(y_values[i:i + x_len], x_values):
                occurrences.append({
                    'start_position': i + 2,  # Excel行号
                    'end_position': i + x_len + 1,
                    'match_index': len(occurrences) + 1
                })

        return {
            'x_sequence': x_values.tolist(),
            'y_sequence': y_values.tolist(),
            'occurrences': occurrences,
            'total_matches': len(occurrences),
            'found': len(occurrences) > 0
        }

    except Exception as e:
        return {
            'found': False,
            'error': str(e)
        }


# 使用示例
if __name__ == "__main__":
    file_path = r"C:\Users\LH\Desktop\SNN_MY\test\test.xlsx"  # 替换为您的文件路径

    print("=== 查找X序列在Y序列中的位置 ===")

    # 方法1: 查找第一个匹配位置
    print("\n1. 查找第一个匹配位置:")
    result = find_x_in_y_excel(
        file_path=file_path,
        x_sheet=0,  # 第一个工作表
        x_col=0,  # 第一列 (0-based索引)
        y_sheet=0,  # 第一个工作表
        y_col=1  # 第二列 (0-based索引)
    )

    if result['found']:
        print(f"找到匹配！")
        print(f"X序列: {result['x_sequence']}")
        print(f"在Y序列中的位置: 第{result['start_position']}行 到 第{result['end_position']}行")
    else:
        print("未找到匹配")
        if 'error' in result:
            print(f"错误: {result['error']}")
        else:
            print(result['message'])

    print("\n" + "=" * 50 + "\n")

    # # 方法2: 查找所有匹配位置
    # print("2. 查找所有匹配位置:")
    # all_results = find_all_occurrences(
    #     file_path=file_path,
    #     x_sheet=0,
    #     x_col=0,
    #     y_sheet=0,
    #     y_col=1
    # )
    #
    # if all_results['found']:
    #     print(f"找到 {all_results['total_matches']} 个匹配:")
    #     for occ in all_results['occurrences']:
    #         print(f"  匹配 {occ['match_index']}: 第{occ['start_position']}行 到 第{occ['end_position']}行")
    # else:
    #     print("未找到任何匹配")
    #     if 'error' in all_results:
    #         print(f"错误: {all_results['error']}")