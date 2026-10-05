import cv2
import numpy as np


img = cv2.imread(r"C:\Users\LH\Desktop\SNN_MY\my_img\IR_008.jpeg")

cv2.imshow("1",img)

print(img[0][3])
# 等待按键关闭窗口（0 表示无限等待）
cv2.waitKey(0)

# 关闭所有 OpenCV 窗口
cv2.destroyAllWindows()