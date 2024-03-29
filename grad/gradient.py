import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
import os
from PIL import Image
from grad import is_image_file
import csv

class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        # kernel_v = [[0, -1, 0],
        #             [0, 0, 0],
        #             [0, 1, 0]]
        # kernel_h = [[0, 0, 0],
        #             [-1, 0, 1],
        #             [0, 0, 0]]
        kernel_v = [[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]]
        kernel_h = [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]

        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x


class Get_lap_gradient_nopadding(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = [[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0 = F.conv2d(x0.unsqueeze(1), self.weight, padding=1)
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=1)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=1)
        x = torch.cat([x0, x1, x2], dim=1)
        return x



def imgdir_grad(dataset, csv_path , save_path):
    """
    获得一个目录下图片的sobel梯度图和平均梯度
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        for imgname in os.listdir(dataset): 
            if  is_image_file(imgname):
                imgpath = os.path.join(dataset,imgname)   
                img = Image.open(imgpath)
                timg = ToTensor()(img).unsqueeze(0).cuda()
                # g = Get_gradient_nopadding()(timg)
                g = Get_lap_gradient_nopadding()(timg)
                gimg = ToPILImage()(g.cpu().squeeze())             
                mgrad = torch.mean(g).item()*255
                gimg.save(os.path.join(save_path,imgname))                                  
                print(imgname,mgrad)
                writer.writerow([imgname,mgrad]) 



                