import torch
import torch.nn as nn
import math
import pickle
import imageio
import numpy as np

class WaveletTransform(nn.Module):
    def __init__(self, scale=1, dec=True, params_path='wavelet_weights_c2.pkl', transpose=True):
        super().__init__()

        self.scale = scale
        self.dec = dec
        self.transpose = transpose

        ks = int(math.pow(2, self.scale))
        nc = 3 * ks * ks

        if dec:
            self.conv = nn.Conv2d(in_channels=3, out_channels=nc, kernel_size=ks, stride=ks, padding=0, groups=3,
                                  bias=False)
        else:
            self.conv = nn.ConvTranspose2d(in_channels=nc, out_channels=3, kernel_size=ks, stride=ks, padding=0,
                                           groups=3, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                f = open(params_path, 'rb')
                dct = pickle.load(f,encoding='bytes')
                f.close()
                # print(type(dct))
                # for key in dct:
                #     print(key)
                m.weight.data = torch.from_numpy(dct[b'rec%d' % ks])
                m.weight.requires_grad = False

    def forward(self, x):
        if self.dec:
            output = self.conv(x)
            if self.transpose:
                osz = output.size()
                # print(osz)
                output = output.view(osz[0], 3, -1, osz[2], osz[3]).transpose(1, 2).contiguous().view(osz)
        else:
            if self.transpose:
                xx = x
                xsz = xx.size()
                xx = xx.view(xsz[0], -1, 3, xsz[2], xsz[3]).transpose(1, 2).contiguous().view(xsz)
            output = self.conv(xx)
        return output


# img_path = r'C:\Users\蒋凯\Desktop\WDSR&EDSR\DIV2K\DIV2K_train_HR\0002.png'
# img = imageio.imread(img_path)
# img = np.rollaxis(img,2,0)
#
# wavelet_dec = WaveletTransform(scale=2, dec=True)
# wavelet_rec = WaveletTransform(scale=1, dec=False)
# img = torch.from_numpy(img)
# img = img.view(1,3,1848,2040).float()
#
# rgb_mean = torch.FloatTensor([0.4488,0.4371,0.4040]).view([1, 3, 1, 1])
# img1 = img / 255
# target_wavelets = wavelet_dec(img)
# target_wavelets1 = wavelet_dec(img1)
# # rec = wavelet_rec(target_wavelets)
# # print(rec.shape)
# print(target_wavelets)
# print(target_wavelets1*255)
# print(target_wavelets1*255 == target_wavelets)


# tcoeffs = pw.dwt2(img, 'haar')
# tcA, (tcH, tcV, tcD) = tcoeffs
#
# lr = torch.from_numpy(np.concatenate((tcA, tcH, tcV, tcD), axis=1))
# print(lr.shape)
# print(target_wavelets)
# print('======================================')
# print(lr.dtype)
# print(torch.from_numpy(tcA) == target_wavelets[:,:3,:,:])

# coeffs = pw.wavedec2(img, 'haar', level=2)
# cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
# # print(torch.from_numpy(cA2) == target_wavelets[:,:3,:,:])
# print(cA2)
# print('=======================================')
# print(target_wavelets[:,:3,:,:])