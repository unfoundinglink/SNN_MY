import cv2
import numpy as np

# 设置渐变条的宽度和高度（可自定义）
width = 512  # 宽度（像素）
height = 100  # 高度（像素）

# 生成水平渐变（0~255）
gradient = np.linspace(0, 255, width, dtype=np.uint8)
gradient = np.tile(gradient, (height, 1))  # 垂直方向复制，增加高度

# 应用 COLORMAP_JET 颜色映射
heatmap = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)

# 保存图片（当前目录下）
output_path = "jet_colormap_gradient.png"
cv2.imwrite(output_path, heatmap)

# 显示图片（可选）
cv2.imshow("COLORMAP_JET Gradient", heatmap)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"渐变条图片已保存至: {output_path}")