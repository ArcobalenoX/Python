import torch
import torch.nn as nn
import torch.nn.functional as F
from deform_conv_v2 import DeformConv2d


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      reflection_padding = kernel_size // 2
      self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group,
                                   bias=False)
            self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)
            self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group,
                                   bias=False)
            self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)
            self.calayer=CALayer(channel_num)
            self.palayer=PALayer(channel_num)

    def forward(self, x):
            y = F.relu(self.norm1(self.conv1(x)))
            y = self.norm2(self.conv2(y))
            res = self.calayer(y)
            res = self.palayer(res)
            res += x
            return F.relu(res)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = float(sign) * torch.Tensor(rgb_mean) * rgb_range

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False

class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()
        assert kernel_size % 2 == 1, 'kernel size should be odd'
        self.padding = (kernel_size - 1)//2
        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, (kernel_size-1)//2, (kernel_size-1)//2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(inc, 1, self.kernel_size, self.kernel_size).contiguous()
        return F.conv2d(x, expand_weight,
                        None, 1, self.padding, 1, inc)


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class SmoothDilatedResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(SmoothDilatedResidualBlock, self).__init__()
        self.pre_conv1 = ShareSepConv(dilation*2-1)
        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)
        self.pre_conv2 = ShareSepConv(dilation*2-1)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)
        self.calayer=CALayer(channel_num)
        self.palayer=PALayer(channel_num)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(self.pre_conv1(x))))
        y = self.norm2(self.conv2(self.pre_conv2(y)))
        res=self.calayer(y)
        res=self.palayer(res)
        res += x
        return F.relu(res)

class Net(nn.Module):
    def __init__(self, in_channel = 4,layers = [64, 96, 128, 16, 32, 32]):
        super(Net, self).__init__()

        self.res1 = SmoothDilatedResidualBlock(256, dilation=2)
        self.res2 = SmoothDilatedResidualBlock(256, dilation=2)
        self.res3 = SmoothDilatedResidualBlock(256, dilation=2)
        self.res4 = SmoothDilatedResidualBlock(256, dilation=4)
        self.res5 = SmoothDilatedResidualBlock(256, dilation=4)
        self.res6 = SmoothDilatedResidualBlock(256, dilation=4)
        self.res7 = ResidualBlock(256, dilation=1)

        self.gate = nn.Conv2d(256 * 3, 3, 3, 1, 1, bias=True)

        self.num = 1

        rgb_mean = (0.5204, 0.5167, 0.5129)
        self.sub_mean = MeanShift(1., rgb_mean, -1)
        self.add_mean = MeanShift(1., rgb_mean, 1)

        self.branch_1 = nn.Sequential(
        nn.Conv2d(in_channel,layers[0],kernel_size=1,bias=False),
        nn.ReLU(inplace=True),
        )
        self.branch_2 = nn.Sequential(
        nn.Conv2d(in_channel,layers[1],kernel_size=1,bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(layers[1],layers[2],kernel_size=3,stride=1,padding=1,bias=False),
        nn.ReLU(inplace=True),
        )
        self.branch_3 = nn.Sequential(
        nn.Conv2d(in_channel,layers[3],kernel_size=1,stride=1,padding=0,bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(layers[3],layers[4],kernel_size=5,stride=1,padding=2,bias=False),
        nn.ReLU(inplace=True),
        )
        self.branch_4 = nn.Sequential(
        nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
        nn.Conv2d(in_channel,layers[5],kernel_size=1,stride=1,padding=0,bias=False),
        nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=(1, 1))

        self.conv_input = ConvLayer(3, 16, kernel_size=11, stride=1)

        self.conv2_1 = DeformConv2d(16, 32, kernel_size=3, stride=2)
        self.conv2_2 = nn.Conv2d(16,32,kernel_size=3,stride=2,dilation=2,padding=2)

        self.conv4_1 = DeformConv2d(32, 64, kernel_size=3, stride=2)
        self.conv4_2 = nn.Conv2d(32,64,kernel_size=3,stride=2,dilation=2,padding=2)

        self.conv8_1 = DeformConv2d(64, 128, kernel_size=3, stride=2)
        self.conv8_2 = nn.Conv2d(64,128,kernel_size=3,stride=2,dilation=2,padding=2)

        self.conv16_1 = DeformConv2d(128, 256, kernel_size=3, stride=2)
        self.conv16_2 = nn.Conv2d(128,256,kernel_size=3,stride=2,dilation=2,padding=2)



        self.convd16x = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)
        self.convd8x = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
        self.convd4x = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)
        self.convd2x = UpsampleConvLayer(32, 16, kernel_size=3, stride=2)

        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        #print(x.shape)
        b_1 = self.branch_1(x)
        #print(b_1.shape)
        b_2 = self.branch_2(x)
        #print(b_2.shape)
        b_3 = self.branch_3(x)
        #print(b_3.shape)
        b_4 = self.branch_4(x)
        #print(b_4.shape)
        y = torch.cat([b_1, b_2, b_3, b_4], dim=1)
        #print(y.shape)

        x = self.conv2(y)

        x = self.relu(self.conv_input(x))
        x1_1 = self.conv2_1(x)
        x1_2 = self.conv2_2(x)
        x1 = self.relu(torch.add(x1_1, x1_2))

        x2_1 = self.conv4_1(x1)
        x2_2 = self.conv4_2(x1)
        x2 = self.relu(torch.add(x2_1, x2_2))

        x3_1 = self.conv8_1(x2)
        x3_2 = self.conv8_2(x2)
        x3 = self.relu(torch.add(x3_1, x3_2))

        x4_1 = self.conv16_1(x3)
        x4_2 = self.conv16_2(x3)
        x4 = self.relu(torch.add(x4_1, x4_2))

        y = self.res1(x4)
        y = self.res2(y)
        y = self.res3(y)
        y2 = self.res4(y)
        y = self.res5(y2)
        y = self.res6(y)
        y3 = self.res7(y)

        gates = self.gate(torch.cat((x4, y2, y3), dim=1))
        gated_y = x4 * gates[:, [0], :, :] + y2 * gates[:, [1], :, :] + y3 * gates[:, [2], :, :]

        res16x = x4+gated_y

        res16x = self.relu(self.convd16x(res16x))
        res16x = F.upsample(res16x, x3.size()[2:], mode='bilinear')
        res8x = torch.add(res16x, x3)

        res8x = self.relu(self.convd8x(res8x))
        res8x = F.upsample(res8x, x2.size()[2:], mode='bilinear')
        res4x = torch.add(res8x, x2)

        res4x = self.relu(self.convd4x(res4x))
        res4x = F.upsample(res4x, x1.size()[2:], mode='bilinear')
        res2x = torch.add(res4x, x1)

        res2x = self.relu(self.convd2x(res2x))
        res2x = F.upsample(res2x, x.size()[2:], mode='bilinear')
        x = torch.add(res2x, x)

        x = self.conv_output(x)

        return x
