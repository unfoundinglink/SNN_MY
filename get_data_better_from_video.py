import numpy as np
import csv
import cv2
import matplotlib.pyplot as plt
import pickle
import os
import logging
import shutil
def extract_matrices_from_csv(csv_file, matrix_rows, matrix_cols, skip_rows=0):
    """
    从CSV文件中提取指定大小的矩阵并组合成3维numpy数组

    参数:
        csv_file (str): CSV文件路径
        matrix_rows (int): 每个矩阵的行数
        matrix_cols (int): 每个矩阵的列数
        skip_rows (int): 跳过文件开头的行数(默认为0)

    返回:
        numpy.ndarray: 3维数组，形状为(num_matrices, matrix_rows, matrix_cols)
    """
    matrices = []
    current_matrix = []

    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)

            # 跳过指定的行数
            for _ in range(skip_rows):
                next(reader)

            for row in reader:
                # 尝试将行转换为数值列表
                numeric_row = []
                has_numeric = False

                for item in row:
                    try:
                        num = float(item)
                        numeric_row.append(num)
                        has_numeric = True
                    except ValueError:
                        pass  # 忽略非数值项

                # 如果行包含数值数据
                if has_numeric:
                    # 检查行的长度是否匹配矩阵列数
                    if len(numeric_row) == matrix_cols:
                        current_matrix.append(numeric_row)
                    elif len(numeric_row) > matrix_cols:
                        # 如果行比需要的长，只取前matrix_cols个元素
                        current_matrix.append(numeric_row[:matrix_cols])
                    # 如果行比需要的短，忽略该行

                    # 检查是否收集完一个完整的矩阵
                    if len(current_matrix) == matrix_rows:
                        matrices.append(current_matrix)
                        current_matrix = []
                # 如果遇到空行或非数值行，且current_matrix不为空，则重置(视为矩阵分隔)
                elif current_matrix:
                    current_matrix = []

    except UnicodeDecodeError:
        # 如果utf-8失败，尝试gbk编码
        with open(csv_file, 'r', encoding='gbk') as file:
            reader = csv.reader(file)

            # 跳过指定的行数
            for _ in range(skip_rows):
                next(reader)

            for row in reader:
                # 尝试将行转换为数值列表
                numeric_row = []
                has_numeric = False

                for item in row:
                    try:
                        num = float(item)
                        numeric_row.append(num)
                        has_numeric = True
                    except ValueError:
                        pass  # 忽略非数值项

                # 如果行包含数值数据
                if has_numeric:
                    # 检查行的长度是否匹配矩阵列数
                    if len(numeric_row) == matrix_cols:
                        current_matrix.append(numeric_row)
                    elif len(numeric_row) > matrix_cols:
                        current_matrix.append(numeric_row[:matrix_cols])

                    if len(current_matrix) == matrix_rows:
                        matrices.append(current_matrix)
                        current_matrix = []
                elif current_matrix:
                    current_matrix = []

    # 转换为3维numpy数组
    if matrices:
        return np.array(matrices)
    else:
        return np.array([])  # 如果没有找到矩阵，返回空数组

def show_heatmap(matrix):
    """
    用红蓝渐变显示二维numpy数组的热图
    参数:
        matrix: 二维numpy数组
    """
    plt.imshow(matrix, cmap='coolwarm')  # 'coolwarm'是红蓝渐变的配色
    plt.colorbar()  # 显示颜色条
    plt.show()


def show_cv_heatmap(matrix, window_name="Heatmap", resize_scale=1.0,index_1 = 0, path = "./" ,save = True ,show =False):
    """
    用OpenCV显示二维numpy数组的热图（红高蓝低）

    参数:
        matrix: 输入的二维numpy数组
        window_name: 窗口名称 (默认"Heatmap")
        resize_scale: 图像缩放比例 (默认1.0不缩放)
    """
    # 归一化到0-255范围
    normalized = ((matrix - matrix.min()) / (matrix.max() - matrix.min()) * 255).astype(np.uint8)

    # 应用红蓝渐变色（COLORMAP_JET）
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    # 始终获取原始尺寸（无论是否缩放）
    h, w = heatmap.shape[:2]

    # 仅当需要缩放时执行resize
    if resize_scale != 1.0:
        heatmap = cv2.resize(heatmap, (int(w * resize_scale), int(h * resize_scale)))

    # 显示图像
    if(show):
        cv2.imshow(window_name, heatmap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if(save):
        cv2.imwrite(path + "/{}.png".format(index_1),heatmap)



def center_crop_2d(matrix, crop_size):
    """
    从二维矩阵中心裁剪指定大小的区域

    参数:
        matrix: 输入的二维numpy数组
        crop_size: 目标尺寸 (int或tuple)
                  - 如果是int，裁剪正方形区域 (crop_size x crop_size)
                  - 如果是tuple (h, w)，裁剪矩形区域

    返回:
        裁剪后的矩阵

    示例:
    """
    # 统一转换为(h, w)格式
    if isinstance(crop_size, int):
        crop_h = crop_w = crop_size
    else:
        crop_h, crop_w = crop_size

    h, w = matrix.shape

    # 计算起始和结束索引
    start_h = max(0, (h - crop_h) // 2)
    start_w = max(0, (w - crop_w) // 2)
    end_h = min(h, start_h + crop_h)
    end_w = min(w, start_w + crop_w)

    # 执行裁剪
    cropped = matrix[start_h:end_h, start_w:end_w]

    # 如果所需尺寸大于原矩阵，用零填充
    if crop_h > h or crop_w > w:
        pad_h = max(0, (crop_h - h) // 2)
        pad_w = max(0, (crop_w - w) // 2)
        cropped = np.pad(cropped,((pad_h, crop_h - h - pad_h),(pad_w, crop_w - w - pad_w)),mode='constant')

    return cropped


def average_pooling(matrix, kernel_size):
    h, w = matrix.shape
    pool_h = h // kernel_size
    pool_w = w // kernel_size

    # 调整输入尺寸，确保能被 kernel_size 整除
    matrix = matrix[:pool_h * kernel_size, :pool_w * kernel_size]

    # 重塑矩阵以方便计算
    reshaped = matrix.reshape(pool_h, kernel_size, pool_w, kernel_size)

    # 计算每个池化块的平均值
    pooled = reshaped.mean(axis=(1, 3))
    return pooled


def main(index):
    array = extract_matrices_from_csv(r'C:\Users\LH\Desktop\SNN_MY\my_csv\IR_00002.csv', matrix_rows=192,
                              matrix_cols=256)
    final_sim = average_pooling(center_crop_2d(array[index], 120), 3)
    return  final_sim


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


def main_all(csv_path,path_save,name):

    origin_data = []
    crop_data = []
    fin_data = []

    path_origin_img = path_save + "\\" + name + "\\" +  "data_origin"
    path_origin_plk = path_save + "\\" + name + "\\" +  "data_origin_plk"

    path_crop_img  = path_save + "\\" + name + "\\" +  "data_crop"
    path_crop_plk = path_save + "\\" + name + "\\" +  "data_crop_plk"

    path_fin_img = path_save + "\\" + name + "\\" +  "data_fin"
    path_fin_plk = path_save + "\\" + name + "\\" +  "data_fin_plk"

    array = extract_matrices_from_csv(csv_path, matrix_rows=192,matrix_cols=256)
    print(array.shape)  # 输出类似 (5, 3, 4) 表示找到了5个3x4的矩阵
    i = 0
    for each in range(array.shape[0]):
        i += 1
        show_cv_heatmap(array[each], index_1=each, path=path_origin_img)
        origin_data.append(array[each])

        show_cv_heatmap(center_crop_2d(array[each], 120), index_1=each, path=path_crop_img)
        crop_data.append(center_crop_2d(array[each], 120))

        final_sim = average_pooling(center_crop_2d(array[each], 120), 3)
        show_cv_heatmap(final_sim, index_1=each, path=path_fin_img)
        fin_data.append(final_sim)
        print(final_sim.shape)
        print(i)

    with open(path_origin_plk + "/origin_data.pkl", 'wb') as f:  # 注意使用二进制模式'wb'
        pickle.dump(origin_data, f)

    with open(path_crop_plk + "/crop_data.pkl", 'wb') as f1:  # 注意使用二进制模式'wb'
        pickle.dump(crop_data, f1)

    with open(path_fin_plk + "/fin_data.pkl", 'wb') as f2:  # 注意使用二进制模式'wb'
        pickle.dump(fin_data, f2)





if __name__ == '__main__':
    csv_path_1 = r"C:\Users\LH\Desktop\SNN_MY\my_csv\IR_00001.csv"
    csv_path_2 = r"C:\Users\LH\Desktop\SNN_MY\my_csv\IR_00002.csv"
    csv_path_3 = r"C:\Users\LH\Desktop\SNN_MY\my_csv\IR_00003.csv"
    csv_path_4 = r"C:\Users\LH\Desktop\SNN_MY\my_csv\IR_00004.csv"
    csv_path_5 = r"C:\Users\LH\Desktop\SNN_MY\my_csv\IR_00005.csv"
    csv_path_6 = r"C:\Users\LH\Desktop\SNN_MY\my_csv\IR_00006.csv"
    csv_path_7 = r"C:\Users\LH\Desktop\SNN_MY\my_csv\IR_00007.csv"
    path_save = r"C:\Users\LH\Desktop\SNN_MY\data_snn_my"

    main_all(csv_path_4 ,path_save ,"data_4")
    main_all(csv_path_5, path_save, "data_5")
    main_all(csv_path_6, path_save, "data_6")
    main_all(csv_path_7, path_save, "data_7")
    # main_all(csv_path_2, path_save, "data_2")
    # main_all(csv_path_3, path_save, "data_3")
    #
    # delete_files_by_extension(path_save + "\\" + "data_3",[".png",".plk"] )
