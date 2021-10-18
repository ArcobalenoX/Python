# -*- coding: utf-8 -*-
from PIL import Image
import PIL
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage
from skimage.util import random_noise
def cv2PIL(im):
    '''cv2图像转PIL
    JPG RGB
    PNG RGBA

    :param im: cv2图像，numpy.ndarray
    :return: PIL图像，PIL.Image
    '''
    try:
        return Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA))
    except:
        return Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))


def PIL2cv(im):
    '''PIL图像转cv2
    JPG RGB
    PNG RGBA
    :param im: PIL图像，PIL.Image
    :return: cv2图像，numpy.ndarray
    '''
    try:
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGBA2BGRA)
    except:
        imnp = np.array(im)
        print(imnp.shape) 
        return cv2.cvtColor(imnp, cv2.COLOR_RGB2BGR)


img = np.array(Image.open('6.png'))
print(img.shape)
h, w, c =img.shape

gaussi = random_noise(img[:,:int(w/2),:], mode='gaussian')
gaussi = gaussi*255  #由于输出是[0，1]的浮点型，先转成灰度图（我的输入就是灰度图）
gaussi = gaussi.astype(np.int32)   #再变成整型数组
gaussi = np.hstack((gaussi,img[:,int(w/2):,:]))
# plt.imshow(gaussi,'gray')
# plt.show()

salypepper = random_noise(img[:int(img.size/2)], mode='s&p')
salypepper = salypepper*255  #由于输出是[0，1]的浮点型，先转成灰度图（我的输入就是灰度图）
salypepper = salypepper.astype(np.int32)   #再变成整型数组
# plt.imshow(salypepper,'gray')
# plt.show()

junyun = random_noise(img[:int(img.size/2)], mode='speckle',mean=0,var=0.01)
junyun = junyun*255  #由于输出是[0，1]的浮点型，先转成灰度图（我的输入就是灰度图）
junyun = junyun.astype(np.int32)   #再变成整型数组
# plt.imshow(junyun,'gray')
# plt.show()

imgs=[img,gaussi,salypepper,junyun]
titiles = ['source','gaussi','salt & peppet','mul']
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(imgs[i],'gray')
    plt.axis('off')
    plt.title(titiles[i])
plt.show()



A_path = '5.png'
B_path = '6.png'
A = cv2.imread(A_path)[:1024,:1024]
B = cv2.imread(B_path)[:1024,:1024]


A = cv2.cvtColor(PIL2cv(img), cv2.COLOR_BGRA2BGR)
B = cv2.cvtColor(PIL2cv(gaussi), cv2.COLOR_BGRA2BGR)

# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in np.arange(6):     #将苹果进行高斯金字塔处理，总共六级处理
    G = cv2.pyrDown(G)
    gpA.append(G)
# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in np.arange(6):  # #将橘子进行高斯金字塔处理，总共六级处理
    G = cv2.pyrDown(G)
    gpB.append(G)


# generate Laplacian Pyramid for A
lpA = [gpA[5]]               
for i in np.arange(5,0,-1):    #将苹果进行拉普拉斯金字塔处理，总共5级处理
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i-1],GE)
    lpA.append(L)

# generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in np.arange(5,0,-1):    #将橘子进行拉普拉斯金字塔处理，总共5级处理
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i-1],GE)
    lpB.append(L)
# Now add left and right halves of images in each level
#numpy.hstack(tup)
#Take a sequence of arrays and stack them horizontally
#to make a single array.
LS = []
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))    #将两个图像的矩阵的左半部分和右半部分拼接到一起
    LS.append(ls)
# now reconstruct
ls_ = LS[0]   #这里LS[0]为高斯金字塔的最小图片
for i in range(1,6):                        #第一次循环的图像为高斯金字塔的最小图片，依次通过拉普拉斯金字塔恢复到大图像
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])                #采用金字塔拼接方法的图像
# image with direct connecting each half
real = np.hstack((A[:,:int(cols/2)],B[:,int(cols/2):]))   #直接的拼接


PIL_A =img = cv2PIL(A)
PIL_B =img = cv2PIL(B)
PIL_ls =img = cv2PIL(ls_)
PIL_real =img = cv2PIL(real)


imgs=[PIL_A,PIL_B,PIL_ls,PIL_real]
titiles = ['A','B','LS','real']
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(imgs[i],'gray')
    plt.axis('off')
    plt.title(titiles[i])
plt.show()

cv2.imwrite('Pyramid_blending2.jpg',ls_)
cv2.imwrite('Direct_blending.jpg',real)


