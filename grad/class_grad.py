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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


# Sobel算子
def sobel_demo(image):
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)  # x方向一阶导数
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)  # y方向一阶导数
    gradx = cv2.convertScaleAbs(grad_x)  # 转回原来的uint8形式
    grady = cv2.convertScaleAbs(grad_y)
    gradxy = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)  # 图像融合
    return gradxy 


def get_image_grad(dataset, csv_path):
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        for i in os.listdir(dataset): #whurs19
            path = os.path.join(dataset,i) #whurs19/airport
            if os.path.isdir(path):  # airport
                for img in os.listdir(path): #airport/airport_1.jpg
                    imgpath = os.path.join(path,img)
                    #print(imgpath)
                    if not is_image_file(imgpath):
                        continue
                    imgcv = cv2.imread(imgpath)                
                    imgcv = cv2.cvtColor(imgcv,cv2.COLOR_BGR2GRAY)
                    h,w = imgcv.shape
                    grad = sobel_demo(imgcv)
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

    oper = "sobel"
    #tag = "train"
    tag = "test"

    csv_path = oper + tag + ".csv"
    dataset_path = r"E:\Code\Python\datas\selfWHURS\WHURS19-"+tag
    out_path = oper+"-cross-"+tag

    #get_image_grad(dataset_path, csv_path)

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    low_grad_dir = "low-"+out_path
    if not os.path.exists(low_grad_dir):
        os.mkdir(low_grad_dir)

    mid_grad_dir = "mid-"+out_path
    if not os.path.exists(mid_grad_dir):
        os.mkdir(mid_grad_dir)

    high_grad_dir = "high-"+out_path
    if not os.path.exists(high_grad_dir):
        os.mkdir(high_grad_dir)    

    grad_list = get_sorted_grad_list(csv_path)
    #grad_list = grad_list[:200] +grad_list[280:480]  + grad_list[560:]
    #grad_list = grad_list[:50] +grad_list[70:120]  + grad_list[-50:]


    #grad_list = grad_list[:300] +grad_list[230:530]  + grad_list[-300:]
    grad_list = grad_list[:75] +grad_list[60:135]  + grad_list[-75:]

    print(len(grad_list))
    #print(grad_list)
    
    for n,i in enumerate(grad_list):
        src = os.path.join(r"E:\Code\Python\datas\RS\WHU-RS19-"+tag+"\GT",i[0])    
        if n < len(grad_list)//3:
            dst = os.path.join(low_grad_dir,i[0])
        elif n < len(grad_list)//3*2:
            dst = os.path.join(mid_grad_dir,i[0])
        else:
            dst = os.path.join(high_grad_dir,i[0])
        copy(src,dst)
    




