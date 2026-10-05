import os
import pickle
from typing import List, Set, Dict
import shutil
import logging

import os
import pickle
from typing import Set, List, Dict
import shutil

def delete_files_by_extension(root_dir, extensions_to_delete):
    """
    保留目录结构，删除指定扩展名的文件

    参数:
        root_dir (str): 要清理的根目录
        extensions_to_delete (list): 要删除的文件扩展名列表（如 ['.tmp', '.log']）
    """
    # 配置日志记录
    logging.basicConfig(
        filename='deleted_files.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    logging.info(f"开始清理目录: {root_dir}")

    # 遍历目录
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()  # 获取文件扩展名（小写）

            # 检查是否是需要删除的扩展名
            if file_ext in extensions_to_delete:
                try:
                    os.remove(file_path)
                    logging.info(f"已删除: {file_path}")
                    print(f"已删除: {file_path}")
                except Exception as e:
                    logging.error(f"删除失败: {file_path} - 错误: {e}")
                    print(f"删除失败: {file_path} - 错误: {e}")

    print("清理完成！已删除的文件记录在 'deleted_files.log'")


from typing import List, Set


def clean_pkl_by_index(source_dir: str, target_dir: str, dry_run: bool = True):
    """
    根据源目录数字名清理.pkl列表的索引位置数据

    参数:
        source_dir: 源目录路径（包含数字文件名）
        target_dir: 目标目录路径
        dry_run: 模拟运行模式（默认True）
    """
    # 1. 获取源目录所有数字文件名（转换为整数集合）
    valid_indices: Set[int] = set()
    for file in os.listdir(source_dir):
        if os.path.isfile(os.path.join(source_dir, file)):
            name, _ = os.path.splitext(file)
            if name.isdigit():
                valid_indices.add(int(name))

    print(f"源目录中找到 {len(valid_indices)} 个有效数字索引")

    # 2. 遍历处理目录
    for root, dirs, files in os.walk(target_dir):
        print(f"\n处理目录: {root}")

        # 处理.pkl文件
        pkl_files = [f for f in files if f.endswith('.pkl')]

        for pkl_file in pkl_files:
            pkl_path = os.path.join(root, pkl_file)

            try:
                with open(pkl_path, 'rb') as f:
                    original_list = pickle.load(f)

                if not isinstance(original_list, list):
                    print(f"  跳过非列表文件: {pkl_file}")
                    continue

                # 保留有效索引位置的数据（注意列表索引从0开始）
                cleaned_list = [
                    item for idx, item in enumerate(original_list)
                    if idx in valid_indices
                ]

                if dry_run:
                    original_len = len(original_list)
                    print(f"  {pkl_file} 将保留 {len(cleaned_list)}/{original_len} 条数据")
                    print(f"  保留的索引位置: {[i for i in range(len(original_list)) if i in valid_indices]}")
                else:
                    # 保存清理后的列表
                    with open(pkl_path, 'wb') as f:
                        pickle.dump(cleaned_list, f)
                    print(f"  已更新 {pkl_file} (保留 {len(cleaned_list)} 条数据)")

            except Exception as e:
                print(f"  处理 {pkl_file} 失败: {e}")

        # 处理其他文件（非.pkl文件）
        other_files = [f for f in files if not f.endswith('.pkl')]

        for file in other_files:
            file_path = os.path.join(root, file)
            name, _ = os.path.splitext(file)

            # 保留条件：文件名数字在源目录中
            if name.isdigit() and int(name) in valid_indices:
                if dry_run:
                    print(f"  [保留] {file}")
                else:
                    pass  # 不需要操作
            else:
                if dry_run:
                    print(f"  [将要删除] {file}")
                else:
                    try:
                        os.remove(file_path)
                        print(f"  [已删除] {file}")
                    except Exception as e:
                        print(f"  删除失败 {file}: {e}")

def clean_main(path_source,path_target):
    config = {
        "source_dir": path_source,  # 替换为实际路径
        "target_dir": path_target,  # 替换为目标路径
        "dry_run": True  # 首次运行使用模拟模式
    }

    print("===== 模拟运行开始 =====")
    clean_pkl_by_index(**config)

    # 确认后实际执行
    if input("\n确认要实际执行操作吗？(y/n): ").lower() == 'y':
        print("\n===== 实际执行开始 =====")
        config["dry_run"] = False
        clean_pkl_by_index(**config)
    else:
        print("操作已取消")



if __name__ == "__main__":
    # 配置参数
    path_source = r"C:\Users\LH\Desktop\SNN_MY\data_snn_my\data_7\data_fin"
    path_target = r"C:\Users\LH\Desktop\SNN_MY\data_snn_my\data_7"

    clean_main(path_source,path_target)

    # path_save = r"C:\Users\LH\Desktop\SNN_MY\data_snn_my"
    # delete_files_by_extension(path_save + "\\" + "data_3", [".png", ".pkl"])



