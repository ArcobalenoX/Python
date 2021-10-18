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


def image_bicubic(img_path,scale=2):
    hr = Image.open(img_path)
    hrnp = np.array(hr)
    h,w,c = hrnp.shape
    lr = hr.resize((h//scale,w//scale),Image.BICUBIC)
    lrnp = np.array(lr)
    sr = lr.resize((h,w),Image.BICUBIC)
    srnp = np.array(sr)
    psnr = skpsnr(hrnp,srnp)
    ssim = skssim(hrnp,srnp, win_size=11 , multichannel=True)
    return psnr,ssim

if __name__ == "__main__":
    testset_dir = "mid-sobel-test"

    all_psnr = []
    all_ssim = [] 
    for i in os.listdir(testset_dir):
        img_path = os.path.join(testset_dir,i)
        if is_image_file(img_path):
            psnr, ssim = image_bicubic(img_path)
            all_psnr.append(psnr)
            all_ssim.append(ssim)
    average_psrn = np.mean(all_psnr)
    average_ssim = np.mean(all_ssim)
    print(f"psnr:{average_psrn:.4f}")
    print(f"ssim:{average_ssim:.4f}")



