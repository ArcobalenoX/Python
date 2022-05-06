from PIL import Image
import PIL
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


#os.mkdir("gaussian_blur")
tag = "test"
images_path = "E:\Code\Python\datas\RS\WHU-RS19-"+tag+"\GT"
for i in os.listdir(images_path):
    img_path = os.path.join(images_path,i)
    img = cv2.imread(img_path)
    print(img.shape)
    gaussi = cv2.GaussianBlur(img, (5,5), sigmaX=0.5)
    cv2.imwrite(os.path.join("gaussian_blur",i),gaussi)