import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt 
from torchvision.transforms import ToTensor, ToPILImage
import glob
import csv
import cv2 
import math
from shutil import copy
from skimage.metrics import structural_similarity as skssim
from skimage.metrics import peak_signal_noise_ratio as skpsnr
from skimage.measure import entropy, shannon_entropy
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


def image_self_psnr(img_path,scale=2):
    hr = Image.open(img_path)
    hrnp = np.array(hr)
    h,w,c = hrnp.shape
    lr = hr.resize((h//2,w//2))
    lrnp = np.array(lr)
    sr = lr.resize((h,w))
    srnp = np.array(sr)
    psnr = skpsnr(hrnp,srnp)

    return psnr




# Sobel算子
def sobel_demo(image):
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)  # x方向一阶导数
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)  # y方向一阶导数
    gradx = cv2.convertScaleAbs(grad_x)  # 转回原来的uint8形式
    grady = cv2.convertScaleAbs(grad_y)
    gradxy = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)  # 图像融合
    return gradxy 

# Scharr算子是Sobel算子的增强版本
def scharr_demo(image):
    grad_x = cv2.Scharr(image, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(image, cv2.CV_32F, 0, 1)
    gradx = cv2.convertScaleAbs(grad_x)
    grady = cv2.convertScaleAbs(grad_y)
    gradxy = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
    return gradxy

#拉普拉斯算子
def laplacian_demo(image): 
    dst = cv2.Laplacian(image, cv2.CV_32F)
    lpls = cv2.convertScaleAbs(dst)
    # 自己定义卷积核
    # kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    # dst = cv2.filter2D(image, cv2.CV_32F, kernel=kernel)
    #lpls = cv2.convertScaleAbs(lpls) #单通道
    return lpls


def get_image_grad(csv_path):
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        for i in os.listdir(in_path):
            path = os.path.join(in_path,i)
            if os.path.isdir(path):
                for img in os.listdir(path):
                    imgpath = os.path.join(path,img)
                    print(imgpath)
                    if not is_image_file(imgpath):
                        continue

                    imgcv = cv2.imread(imgpath)                
                    imgcv = cv2.cvtColor(imgcv,cv2.COLOR_BGR2GRAY)
                    h,w = imgcv.shape
                    #shan_entropy = shannon_entropy(imgcv)
                    #grad = sobel_demo(imgcv)
                    #grad = scharr_demo(imgcv)
                    grad = laplacian_demo(imgcv)                
                    cv2.imwrite(os.path.join(out_path,img),grad)
                    grad = np.sum(grad)/h/w
                    #print(img,grad)
                    writer.writerow([img,grad]) 

def get_sorted_grad_list(csv_path):
    grad = {}
    with open(csv_path, "r") as f:
        f_csv = csv.reader(f)
        #for i in f_csv:
        for k,v in f_csv:
            grad[k] = float(v)
    sorted_grad = sorted(grad.items(),key=lambda x:x[1])
    grad_list = list(sorted_grad)
    return grad_list




if __name__ == "__main__":

    #image_path = r"..\WHURS19-test\Meadow\meadow_41.jpg"
    #image_path = r"..\WHURS19-test\Beach\beach_41.jpg"
    #image_path = r"..\WHURS19-test\Mountain\Mountain_41.jpg"
    #image_self_psnr(image_path)

    #img_ = cv2.imread(image_path)
    #img_ = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
    #print(os.path.basename(image_path),shannon_entropy(img_))   

    oper = "sobel"
    #oper = "laplacian"
    #tag = "train"
    tag = "test"

    csv_path = oper + tag + ".csv"
    in_path = r"E:\Code\Python\iPython\WHURS19-"+tag
    out_path = oper+"-"+tag


    if not os.path.exists(out_path):
        os.mkdir(out_path)

    low_grad_dir = "low-"+oper+"-"+tag
    if not os.path.exists(low_grad_dir):
        os.mkdir(low_grad_dir)

    mid_grad_dir = "mid-"+oper+"-"+tag
    if not os.path.exists(mid_grad_dir):
        os.mkdir(mid_grad_dir)

    high_grad_dir = "high-"+oper+"-"+tag
    if not os.path.exists(high_grad_dir):
        os.mkdir(high_grad_dir)    

    grad_list = get_sorted_grad_list(csv_path)
    for n,i in enumerate(grad_list):
        src = os.path.join(r"E:\Code\Python\datas\RS\WHU-RS19-"+tag+"\GT",i[0])    
        if n < len(grad_list)//3:
            dst = os.path.join(low_grad_dir,i[0])
        elif n < len(grad_list)//3*2:
            dst = os.path.join(mid_grad_dir,i[0])
        else:
            dst = os.path.join(high_grad_dir,i[0])
        copy(src,dst)





