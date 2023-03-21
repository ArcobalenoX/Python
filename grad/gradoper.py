import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
imgpath = r"E:\Code\Python\liif-self\data\selfWHURS\WHURS-train\GT\airport_35.jpg"
img = cv2.imread(imgpath)

rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgname = os.path.basename(imgpath)[:-4]

# 灰度化处理图像
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 高斯滤波
#gray_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

gray_image = gray_image

# Roberts 算子
kernelx = np.array([[-1, 0], [0, 1]], dtype = int)
kernely = np.array([[0, -1], [1, 0]], dtype = int)
x = cv2.filter2D(gray_image, cv2.CV_16S, kernelx)
y = cv2.filter2D(gray_image, cv2.CV_16S, kernely)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
print("Roberts mgrad  ",np.mean(Roberts))


# Prewitt 算子
kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
x = cv2.filter2D(gray_image, cv2.CV_16S, kernelx)
y = cv2.filter2D(gray_image, cv2.CV_16S, kernely)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
print("Prewitt mgrad  ",np.mean(Prewitt))

# Sobel 算子
x = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0)
y = cv2.Sobel(gray_image, cv2.CV_16S, 0, 1)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
print("Sobel mgrad  ",np.mean(Sobel))


# Scharr 算子
x = cv2.Scharr(gray_image, cv2.CV_16S, 1, 0)
y = cv2.Scharr(gray_image, cv2.CV_16S, 0, 1)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Scharr = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
print("Scharr mgrad  ",np.mean(Scharr))


# 拉普拉斯算法
dst = cv2.Laplacian(gray_image, cv2.CV_16S, ksize = 3)
Laplacian = cv2.convertScaleAbs(dst)
print("Laplacian mgrad  ",np.mean(Laplacian))

# 展示图像
titles = ['Source Image', 'Roberts Image',
          'Prewitt Image','Sobel Image', 
          'Scharr Image','Laplacian Image']
images = [rgb_img, Roberts, Prewitt, Sobel,Scharr, Laplacian]


for i in np.arange(6):
   plt.subplot(2, 3, i+1)
   cv2.imwrite(f"{imgname} {titles[i]}.png",images[i])
   plt.imshow(images[i], 'gray')
   plt.title(titles[i])
   plt.xticks([]), plt.yticks([])
plt.show()
