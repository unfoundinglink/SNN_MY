import os
import pickle
from typing import List, Any


def find_and_load_pickle_lists(root_dir: str, target_filename: str = "*.pkl") -> List[List[Any]]:
    """
    在指定目录及其子目录中查找pickle文件，并解析其中的列表

    参数:
        root_dir: 要搜索的根目录路径
        target_filename: 要查找的文件名模式（支持通配符，如"*.pkl"）

    返回:
        包含所有pickle文件中列表的嵌套列表

    异常:
        如果pickle文件不包含列表，会跳过并打印警告
    """
    all_lists = []

    # 遍历目录树
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # 检查文件名匹配（支持通配符）
            if not fnmatch.fnmatch(filename, target_filename):
                continue

            file_path = os.path.join(dirpath, filename)

            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)

                    if isinstance(data, list):
                        all_lists.append(data)
                    else:
                        print(f"警告: 文件 {file_path} 不包含列表，已跳过")

            except (pickle.PickleError, EOFError) as e:
                print(f"错误: 无法解析文件 {file_path} - {str(e)}")
            except Exception as e:
                print(f"错误: 处理文件 {file_path} 时发生意外错误 - {str(e)}")

    return all_lists

def write_data_all(found_lists,search_dir = r"C:\Users\LH\Desktop\SNN_MY\data_snn_my" ):
    data_all = []
    for each in range(len(found_lists)):
        # label = [0 for j in range(len(found_lists))]
        # label[each] = 1
        label = each
        for item in found_lists[each]:
            data_all.append([label,item])
    with open(search_dir +  "\\" + "data_all.pkl","wb") as f:
        pickle.dump(data_all,f)

def read_data_all(search_dir = r"C:\Users\LH\Desktop\SNN_MY\data_snn_my" ):

    with open(search_dir + "\\" + "data_all.pkl","rb") as f:
        data_all = pickle.load(f)

    # print(data_all)
    print(len(data_all))
    # print(data_all[401])
    print(data_all[1600][0])
    print(data_all[1600][1].shape)




# 使用示例
if __name__ == "__main__":
    import fnmatch  # 用于通配符匹配
    import pickle
    search_dir = r"C:\Users\LH\Desktop\SNN_MY\data_snn_my"  # 替换为你的目录
    found_lists = find_and_load_pickle_lists(search_dir, "fin_data.pkl")
    write_data_all(found_lists)

    # print(found_lists)
    # print(found_lists[0][0].shape)

    # print(f"找到 {len(found_lists)} 个pickle列表:")
    # for i, lst in enumerate(found_lists, 1):
    #     print(f"列表 {i}: 长度={len(lst)}, 示例元素={lst[:3] if lst else '空列表'}...")

    read_data_all()
