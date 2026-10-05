import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image


def temperature_to_image(temperature_data, vmin=None, vmax=None, cmap='jet'):
    """
    将温度矩阵转换为可视化图片（保留原始数据）

    参数:
        temperature_data: 二维温度矩阵 (np.ndarray)
        vmin/vmax: 颜色映射范围 (None则自动计算)
        cmap: 颜色映射 (如'jet', 'viridis', 'coolwarm')

    返回:
        image: PIL.Image对象 (可直接保存为PNG)
    """
    # 归一化到0-255
    vmin = np.min(temperature_data) if vmin is None else vmin
    vmax = np.max(temperature_data) if vmax is None else vmax
    normalized = ((temperature_data - vmin) / (vmax - vmin + 1e-10) * 255).astype(np.uint8)

    # 应用颜色映射
    cmap = plt.get_cmap(cmap)
    colored = cmap(normalized / 255.0)[:, :, :3]  # 去除alpha通道
    colored = (colored * 255).astype(np.uint8)

    # 将vmin/vmax编码到图片的Alpha通道
    alpha = np.zeros_like(normalized) + 255
    encoded = np.dstack((colored, alpha))

    # 在图片右下角存储vmin/vmax (4x4像素区域)
    h, w = encoded.shape[:2]
    encoded[h - 4:h, w - 4:w, 3] = 255  # Alpha通道标记区域
    encoded[h - 4:h - 2, w - 4:w, 0] = int(vmin % 256)
    encoded[h - 4:h - 2, w - 4:w, 1] = int(vmin // 256)
    encoded[h - 2:h, w - 4:w, 0] = int(vmax % 256)
    encoded[h - 2:h, w - 4:w, 1] = int(vmax // 256)

    return Image.fromarray(encoded)


def image_to_temperature(image):
    """
    从图片恢复原始温度数据 (修复OpenCV类型错误)
    """
    if isinstance(image, str):
        image = Image.open(image)
    encoded = np.array(image)

    # 从Alpha通道提取vmin/vmax
    h, w = encoded.shape[:2]
    marker = encoded[h - 4:, w - 4:, 3]
    if np.all(marker == 255):
        vmin = encoded[h - 4, w - 4, 0] + encoded[h - 4, w - 4, 1] * 256
        vmax = encoded[h - 2, w - 4, 0] + encoded[h - 2, w - 4, 1] * 256
    else:
        vmin, vmax = 0, 100  # 默认值

    # 修复点：确保输入为uint8类型
    colored = (encoded[:, :, :3]).astype(np.uint8)  # 强制转换为uint8
    grayscale = cv2.cvtColor(colored, cv2.COLOR_RGB2GRAY).astype(float)
    temperature_data = grayscale / 255.0 * (vmax - vmin) + vmin

    return temperature_data


# 示例使用
if __name__ == "__main__":
    # 生成模拟温度数据 (20x30矩阵，范围25.0~45.0°C)
    temp_data = np.random.rand(20, 30) * 20 + 25
    print(temp_data)
    # 转换为图片并保存
    img = temperature_to_image(temp_data, cmap='coolwarm')
    img.save("temperature.png")
    print("图片已保存，大小:", img.size)

    # 从图片恢复数据
    recovered_data = image_to_temperature("temperature.png")

    # 计算误差
    error = np.max(np.abs(temp_data - recovered_data))
    print(f"最大恢复误差: {error:.4f}°C")

    # 可视化对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(temp_data, cmap='coolwarm')
    axes[0].set_title("原始数据")
    axes[1].imshow(img)
    axes[1].set_title("编码图片")
    axes[2].imshow(recovered_data, cmap='coolwarm')
    axes[2].set_title("恢复数据")
    plt.show()