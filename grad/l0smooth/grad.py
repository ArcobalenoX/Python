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
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif','tif'])


def image_self_psnr(img_path,scale=2):
    hr = Image.open(img_path)
    hrnp = np.array(hr)
    h,w,c = hrnp.shape
    lr = hr.resize((h//scale,w//scale))
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




def opencv_mgrad(imgpath):
    imgcv = cv2.imread(imgpath)                
    imgcv = cv2.cvtColor(imgcv,cv2.COLOR_BGR2GRAY)
    grad = sobel_demo(imgcv)
    mgrad = np.mean(grad)              
    gradimg = cv2.cvtColor(grad,cv2.COLOR_GRAY2BGR)
    return gradimg,mgrad

def get_imgages_grad(dataset, csv_path , save_path):
    """
    获得多类别目录下的sobel梯度图和平均梯度
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        for i in os.listdir(dataset): #whurs19
            path = os.path.join(dataset,i) #whurs19/airport
            if os.path.isdir(path):  # airport
                for img in os.listdir(path): #airport/airport_1.jpg
                    if  is_image_file(img):
                        imgpath = os.path.join(path,img)
                        #print(imgpath)                        
                        gradimg, mgrad = opencv_mgrad(imgpath)
                        cv2.imwrite(os.path.join(save_path,img),gradimg)
                        print(img,mgrad)
                        writer.writerow([img,mgrad]) 

def get_imgdir_grad(dataset, csv_path , save_path):
    """
    获得一个目录下图片的梯度图和平均梯度
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        for img in os.listdir(dataset): 
            if  is_image_file(img):
                imgpath = os.path.join(dataset,img)                
                gradimg, mgrad = opencv_mgrad(imgpath)
                cv2.imwrite(os.path.join(save_path,img),gradimg)                                 
                print(img,mgrad)
                writer.writerow([img,mgrad]) 

def class_grad(img_dir):
    """
    获得一个目录下图片的某类的梯度图和平均梯度
    """
    all_grad = []
    for i in os.listdir(img_dir):
        if is_image_file(i):
            imgpath = os.path.join(img_dir, i)
            gradimg, mgrad = opencv_mgrad(imgpath)
            print(f"{i} {mgrad:.4f}")
            all_grad.append(mgrad)
    mean = np.mean(all_grad)
    std = np.std(all_grad)
    return mean,std

def whurs19_sobel():
    whurs = r"E:\Code\Python\datas\RS\WHURS19"
    class19 = os.listdir(whurs)
    for c in class19:
        path =  os.path.join(whurs,c)
        mean, std = class_grad(path)
        print(f"{c:20} {mean:.4f}  {std:.4f}")    

def get_sorted_grad_list(csv_path):
    """
    将梯度排序
    """
    grad = {}
    with open(csv_path, "r") as f:
        f_csv = csv.reader(f)
        for k,v in f_csv:
            grad[k] = float(v)
    sorted_grad = sorted(grad.items(),key=lambda x:x[1])
    grad_list = list(sorted_grad)
    return grad_list


def sobel_grading(src_dir, csv_path,out_path):
    """
    将图片按梯度分为三级
    """

    low_grad_dir = "low"+out_path
    if not os.path.exists(low_grad_dir):
        os.mkdir(low_grad_dir)

    mid_grad_dir = "mid"+out_path
    if not os.path.exists(mid_grad_dir):
        os.mkdir(mid_grad_dir)

    high_grad_dir = "high"+out_path
    if not os.path.exists(high_grad_dir):
        os.mkdir(high_grad_dir)    

    grad_list = get_sorted_grad_list(csv_path)

    #grad_list = grad_list[:200] +grad_list[280:480]  + grad_list[560:]
    #grad_list = grad_list[:300] +grad_list[230:530]  + grad_list[-300:]    

    #grad_list = grad_list[:50] +grad_list[70:120]  + grad_list[-50:]
    #grad_list = grad_list[:75] +grad_list[60:135]  + grad_list[-75:]

    print(len(grad_list))
    #print(grad_list)
    
    for n,i in enumerate(grad_list): #i --> [imgpath, grad]
        src = os.path.join(src_dir,i[0])    
        if n < len(grad_list)//3:
            dst = os.path.join(low_grad_dir,i[0])
        elif n < len(grad_list)//3*2:
            dst = os.path.join(mid_grad_dir,i[0])
        else:
            dst = os.path.join(high_grad_dir,i[0])
        copy(src,dst)
    

if __name__ == "__main__":
    #get_imgages_grad(r"E:\Code\Python\datas\RS\WHU-RS19-train\GT", "sobeltrain.csv", "sobel-train")
    #get_imgages_grad(r"E:\Code\Python\datas\RS\WHU-RS19-test\GT", "sobeltest.csv", "sobel-test")
    #sobel_grading(r"E:\Code\Python\datas\RS\WHU-RS19-train\GT","sobeltrain.csv","-sobel-train")
    #sobel_grading(r"E:\Code\Python\datas\RS\WHU-RS19-test\GT","sobeltest.csv","-sobel-test")
    sobel_grading(r"E:\Code\Python\datas\selfAID\AID","AIDtest.csv","-AID-test")


    #get_imgdir_grad("smooth_whurs_train", "smooth_whurs_train.csv" , "smooth_whurs_train_grad")



    #get_imgages_grad(r"E:\Code\Python\datas\RS\AID", "AID.csv", "AID")
    #sobel_grading(r"E:\Code\Python\datas\selfAID\AID","AID.csv","-sobel-AID")


    # grad_src_dir = r"smooth_whurs_test_grad"
    # src_dir = r"low-sobel-test"    
    # dst_dir = r"smooth_whurs_test_low_grad"

    # os.mkdir(dst_dir)
    # for img in os.listdir(src_dir):
    #     src_path = os.path.join(grad_src_dir,img)
    #     dst_path = os.path.join(dst_dir,img)
    #     copy(src_path,dst_path)
        
    pass


