import argparse
import time
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import utils

# Source code of AlexNet
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(  # [50,3, 224, 224]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),     # [50, 96, 54, 54]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),                                   # [50,  96, 26, 26]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),    # [50, 256, 26, 26]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),                                   # [50, 384, 12, 12]

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),   # [50, 384, 12, 12]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),   # [50, 384, 12, 12]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),   # [50, 256, 12, 12]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)  # [50,256, 5, 5]
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),    # total class: 10
        )

    def forward(self, img):
        feature = self.conv(img)
        feature = feature.view(img.shape[0], -1)
        output = self.fc(feature)
        return output

# -------------------------------------------------------------------------CIFAR10-SVHN
# 1: AlexCapsNet-CIFAR10 32*32 done
class AlexCapsNet_CIFAR10(nn.Module):

    def __init__(self, device):
        super(AlexCapsNet_CIFAR10, self).__init__()
        self.conv = nn.Sequential(  # [50, 3,32,32]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),  # [50, 96,32,32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 96,16,16]
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数------------------------------------
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256,16,16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50,256, 8, 8]
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 8, 8]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 8, 8]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)  # [50,256, 6, 6]
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=3, stride=1, padding=1),
            # 32*[50,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=10, out_dim_caps=16,
                               device=device, routings=3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        return x

# 2: AlexNet-CIFAR10 32*32  done
class AlexNet_CIFAR10(nn.Module):
    def __init__(self):
        super(AlexNet_CIFAR10, self).__init__()
        self.conv = nn.Sequential(  # [50,  3,32,32]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),  # [50, 96,32,32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 96,16,16]
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数------------------------------------
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256,16,16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50,256, 8, 8]
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 8, 8]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 8, 8]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)  # [50,256, 6, 6]
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # total class: 10
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        feature = feature.view(img.shape[0], -1)
        output = self.fc(feature)
        return output

# 3: CapsAlexNet-CIFAR10 32*32,
class CapsAlexNet_CIFAR10(nn.Module):

    def __init__(self, device):
        super(CapsAlexNet_CIFAR10, self).__init__()
        self.conv = nn.Sequential(  # [50,  3,32,32]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),  # [50, 96,32,32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 96,16,16]
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数------------------------------------
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256,16,16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50,256, 8, 8]
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=3, stride=1, padding=0),
            # 32*[50,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]
            # utils.DigitCaps(area=1152, digit_cap_num=10, primary_cap_dim=8, digit_cap_dim=16, batch_size=batch_size,
            #                   device=device, rountin_num=3)  # [10,16]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=10, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.conv2 = nn.Sequential(  # [1, 10, 16]
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=1, padding=0),  # [50, 256, 8, 14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),  # [50, 256, 3, 6]
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(256*3*6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),  # total class: 10
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv(x)
        x = self.Cap(x)
        x = x.view(batch_size, 1, 10, 16)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 4: CapsNet-CIFAR10 32*32     done
class CapsNet_CIFAR10(nn.Module):

    def __init__(self, device):
        super(CapsNet_CIFAR10, self).__init__()
        self.conv = nn.Sequential(  # [50,  3,32,32]
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=9, stride=1, padding=0),  # [batch_size, 256, 24, 24]
            nn.ReLU()
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=9, stride=2, padding=0), # [batch, 8, 32, 8, 8]
            # 32*[bs,8,8,8]->[50,8,64,32]->[50,8,54*32(2048)]

            utils.DenseCapsule(in_num_caps=2048, in_dim_caps=8, out_num_caps=10, out_dim_caps=16,
                               device=device, routings=3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        return x

# 5: CapsNet with Reconstruction-CIFAR10
class CapsNet_Recon_CIFAR10(nn.Module):
    def __init__(self, device):
        super(CapsNet_Recon_CIFAR10, self).__init__()
        self.conv = nn.Sequential(  # [50,  3,32,32]
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=9, stride=1, padding=0),  # [batch_size, 256, 24, 24]
            nn.ReLU()
        )
        self.Cap = nn.Sequential(
            # 32*[bs,8,8,8]->[50,8,64,32]->[50,8,54*32(2048)]
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=9, stride=2, padding=0), # [batch, 8, 32, 8, 8]

            utils.DenseCapsule(in_num_caps=2048, in_dim_caps=8, out_num_caps=10, out_dim_caps=16,
                               device=device, routings=3)                                   # [bs,10,16]
        )
        self.reconstruction = utils.ReconstructionNet(num_dim=16, num_caps=10, img_size=32, original_chanel=3)

    def forward(self, x, targets):
        feature = self.conv(x)
        v = self.Cap(feature)
        reconstruction = self.reconstruction(v, targets)
        return v, reconstruction

# 6: AlexCapsNet with Reconstruction
class AlexCapsNet_Recon_CIFAR10(nn.Module):

    def __init__(self, device):
        super(AlexCapsNet_Recon_CIFAR10, self).__init__()
        self.conv = nn.Sequential(  # [50,  3,32,32]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),  # [50, 96,32,32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 96,16,16]
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数------------------------------------
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256,16,16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50,256, 8, 8]
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 8, 8]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 8, 8]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)  # [50,256, 6, 6]
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=3, stride=1, padding=1),
            # 32*[50,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]
            # utils.DigitCaps(area=1152, digit_cap_num=10, primary_cap_dim=8, digit_cap_dim=16, batch_size=batch_size,
            #                   device=device, rountin_num=3)  # [10,16]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=10, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.Recon = utils.ReconstructionNet(num_dim=16, num_caps=10, img_size=32, original_chanel=3)

    def forward(self, x, targets):
        feature = self.conv(x)
        v = self.Cap(feature)
        reconstruction = self.Recon(v, targets)
        return v, reconstruction

# 7: AlexCapsNet_Fully
class AlexCapsNet_Fully_CIFAR10(nn.Module):
    def __init__(self, device):
        super(AlexCapsNet_Fully_CIFAR10, self).__init__()
        self.conv = nn.Sequential(  # [50, 3,32,32]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),  # [50, 96,32,32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 96,16,16]
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数------------------------------------
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256,16,16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50,256, 8, 8]
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 8, 8]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 8, 8]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)  # [50,256, 6, 6]
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=3, stride=1, padding=1),
            # 32*[50,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=10, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(10 * 16, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),  # total class: 10
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 8: AlexCapsNet_Fully_Recon
class AlexCapsNet_F_Recon_CIFAR10(nn.Module):
    def __init__(self, device):
        super(AlexCapsNet_F_Recon_CIFAR10, self).__init__()
        self.conv = nn.Sequential(  # [50, 3,32,32]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),  # [50, 96,32,32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 96,16,16]
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数------------------------------------
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256,16,16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50,256, 8, 8]
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 8, 8]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 8, 8]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)  # [50,256, 6, 6]
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=3, stride=1, padding=1),
            # 32*[50,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=10, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(10 * 16, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),  # total class: 10
        )
        self.reconstruction = utils.ReconstructionNet(num_dim=16, num_caps=10, img_size=32, original_chanel=3)

    def forward(self, x, targets):
        x = self.conv(x)
        v = self.Cap(x)
        reconstruction = self.reconstruction(v, targets)
        x = self.flatten(v)
        x = self.fc(x)
        return x, reconstruction

# 9：Shallow AlexCapsNet_F
class S_AlexCapsNet_Fully_CIFAR10(nn.Module):
    def __init__(self, device):
        super(S_AlexCapsNet_Fully_CIFAR10, self).__init__()
        self.conv = nn.Sequential(  # [50, 3,32,32]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),  # [50, 96,32,32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 96,16,16]
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数------------------------------------
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256,16,16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50,256, 8, 8]
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 8, 8]
            nn.ReLU(),
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=384, out_channel=8, kersel_size=3, stride=1, padding=0),
            # 32*[50,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=10, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(10 * 16, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),  # total class: 10
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 10: Shallow AlexCapsNet
class S_AlexCapsNet_CIFAR10(nn.Module):

    def __init__(self, device):
        super(S_AlexCapsNet_CIFAR10, self).__init__()
        self.conv = nn.Sequential(  # [50, 3,32,32]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),  # [50, 96,32,32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 96,16,16]
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数------------------------------------
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256,16,16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50,256, 8, 8]
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 8, 8]
            nn.ReLU()
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=384, out_channel=8, kersel_size=3, stride=1, padding=0),
            # 32*[50,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=10, out_dim_caps=16,
                               device=device, routings=3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        return x

#----------------------------------------------------------------------------CIFAR100
# 1: AlexCapsNet-CIFAR100 32*32 done
class AlexCapsNet_CIFAR100(nn.Module):

    def __init__(self, device):
        super(AlexCapsNet_CIFAR100, self).__init__()
        self.conv = nn.Sequential(  # [50,  3,32,32]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),  # [50, 96,32,32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 96,16,16]
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数------------------------------------
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256,16,16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50,256, 8, 8]
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 8, 8]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 8, 8]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)  # [50,256, 6, 6]
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=3, stride=1, padding=1),
            # 32*[50,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]
            # utils.DigitCaps(area=1152, digit_cap_num=10, primary_cap_dim=8, digit_cap_dim=16, batch_size=batch_size,
            #                   device=device, rountin_num=3)  # [10,16]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=100, out_dim_caps=16,
                               device=device, routings=3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        return x

# 2: AlexNet-CIFAR100
class AlexNet_CIFAR100(nn.Module):
    def __init__(self):
        super(AlexNet_CIFAR100, self).__init__()
        self.conv = nn.Sequential(  # [50,  3,32,32]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),  # [50, 96,32,32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 96,16,16]
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数------------------------------------
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256,16,16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50,256, 8, 8]
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 8, 8]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 8, 8]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)  # [50,256, 6, 6]
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # total class: 100
            nn.Linear(4096, 100),
        )

    def forward(self, img):
        feature = self.conv(img)
        feature = feature.view(img.shape[0], -1)
        output = self.fc(feature)
        return output

# 3: CapsNet-CIFAR100 32*32
class CapsNet_CIFAR100(nn.Module):

    def __init__(self, device):
        super(CapsNet_CIFAR100, self).__init__()
        self.conv = nn.Sequential(  # [50,  3,32,32]
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=9, stride=1, padding=0),  # [batch_size, 256, 24, 24]
            nn.ReLU()
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=9, stride=2, padding=0), # [batch, 8, 32, 8, 8]
            # 32*[bs,8,8,8]->[50,8,64,32]->[50,8,54*32(2048)]

            utils.DenseCapsule(in_num_caps=2048, in_dim_caps=8, out_num_caps=100, out_dim_caps=16,
                               device=device, routings=3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        return x

# 4: CapsAlexNet-CIFAR100 32*32,
class CapsAlexNet_CIFAR100(nn.Module):

    def __init__(self, device):
        super(CapsAlexNet_CIFAR100, self).__init__()
        self.conv = nn.Sequential(  # [50,  3,32,32]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),  # [50, 96,32,32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 96,16,16]
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数------------------------------------
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256,16,16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50,256, 8, 8]
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=3, stride=1, padding=0),
            # 32*[50,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]
            # utils.DigitCaps(area=1152, digit_cap_num=10, primary_cap_dim=8, digit_cap_dim=16, batch_size=batch_size,
            #                   device=device, rountin_num=3)  # [10,16]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=100, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.conv2 = nn.Sequential(  # [1, 100, 16]
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=1, padding=0),  # [50, 256, 98, 14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),  # [50, 256, 48, 6]
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(256*48*6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 100),  # total class: 100
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        x = x.view(-1, 1, 100, 16)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 5: CapNet-Recon-CIFAR100
class CapsNet_Recon_CIFAR100(nn.Module):
    def __init__(self, device):
        super(CapsNet_Recon_CIFAR100, self).__init__()
        self.conv = nn.Sequential(  # [50,  3,32,32]
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=9, stride=1, padding=0),  # [batch_size, 256, 24, 24]
            nn.ReLU()
        )
        self.Cap = nn.Sequential(
            # 32*[bs,8,8,8]->[50,8,64,32]->[50,8,54*32(2048)]
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=9, stride=2, padding=0), # [batch, 8, 32, 8, 8]

            utils.DenseCapsule(in_num_caps=2048, in_dim_caps=8, out_num_caps=100, out_dim_caps=16,
                               device=device, routings=3)                                   # [bs,10,16]
        )
        self.reconstruction = utils.ReconstructionNet(num_dim=16, num_caps=100, img_size=32, original_chanel=3)

    def forward(self, x, targets):
        feature = self.conv(x)
        v = self.Cap(feature)
        reconstruction = self.reconstruction(v, targets)
        return v, reconstruction

# 6: AlexCapsNet with Reconstruction-CIFAR100 32*32
class AlexCapsNet_Recon_CIFAR100(nn.Module):
    def __init__(self, device):
        super(AlexCapsNet_Recon_CIFAR100, self).__init__()
        self.conv = nn.Sequential(  # [50,  3,32,32]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),  # [50, 96,32,32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 96,16,16]
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数------------------------------------
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256,16,16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50,256, 8, 8]
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 8, 8]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 8, 8]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)  # [50,256, 6, 6]
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=3, stride=1, padding=1),
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=100, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.Recon = utils.ReconstructionNet(num_dim=16, num_caps=100, img_size=32, original_chanel=3)

    def forward(self, x, targets):
        feature = self.conv(x)
        v = self.Cap(feature)
        reconstruction = self.Recon(v, targets)
        return v, reconstruction

# 7: AlexCapsNet_Fully
class AlexCapsNet_Fully_CIFAR100(nn.Module):
    def __init__(self, device):
        super(AlexCapsNet_Fully_CIFAR100, self).__init__()
        self.conv = nn.Sequential(  # [50, 3,32,32]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),  # [50, 96,32,32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 96,16,16]
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数------------------------------------
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256,16,16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50,256, 8, 8]
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 8, 8]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 8, 8]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)  # [50,256, 6, 6]
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=3, stride=1, padding=1),
            # 32*[50,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=100, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(100 * 16, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 100),  # total class: 100
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 8: AlexCapsNet_Fully_Recon
class AlexCapsNet_F_Recon_CIFAR100(nn.Module):
    def __init__(self, device):
        super(AlexCapsNet_F_Recon_CIFAR100, self).__init__()
        self.conv = nn.Sequential(  # [50, 3,32,32]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),  # [50, 96,32,32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 96,16,16]
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数------------------------------------
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256,16,16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50,256, 8, 8]
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 8, 8]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 8, 8]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)  # [50,256, 6, 6]
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=3, stride=1, padding=1),
            # 32*[50,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=100, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(100 * 16, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 100),  # total class: 100
        )
        self.reconstruction = utils.ReconstructionNet(num_dim=16, num_caps=100, img_size=32, original_chanel=3)

    def forward(self, x, targets):
        x = self.conv(x)
        v = self.Cap(x)
        reconstruction = self.reconstruction(v, targets)
        x = self.flatten(v)
        x = self.fc(x)
        return x, reconstruction

# 9: Shallow AlexCapsNet_F
class S_AlexCapsNet_Fully_CIFAR100(nn.Module):
    def __init__(self, device):
        super(S_AlexCapsNet_Fully_CIFAR100, self).__init__()
        self.conv = nn.Sequential(  # [50, 3,32,32]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),  # [50, 96,32,32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 96,16,16]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256,16,16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50,256, 8, 8]
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 8, 8]
            nn.ReLU(),
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=384, out_channel=8, kersel_size=3, stride=1, padding=0),
            # 32*[50,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=100, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(100 * 16, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 100),  # total class: 100
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 10: Shallow AlexCapsNet
class S_AlexCapsNet_CIFAR100(nn.Module):

    def __init__(self, device):
        super(S_AlexCapsNet_CIFAR100, self).__init__()
        self.conv = nn.Sequential(  # [50,  3,32,32]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1),  # [50, 96,32,32]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 96,16,16]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256,16,16]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50,256, 8, 8]

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 8, 8]
            nn.ReLU(),
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=384, out_channel=8, kersel_size=3, stride=1, padding=0),
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=100, out_dim_caps=16,
                               device=device, routings=3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        return x

# -----------------------------------------------------------------------------MINIST-FMNIST
# 1: AlexCpsNet-MNIST 28*28
class AlexCapsNet_MNIST(nn.Module):

    def __init__(self, device):
        super(AlexCapsNet_MNIST, self).__init__()
        self.conv = nn.Sequential(  # [50, 28, 28]
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=3, stride=1, padding=1),  # [50, 96,28,28]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 96,14,14]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256,14,14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50,256, 7, 7]
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 7, 7]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 7, 7]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256, 7, 7]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)  # [50,256, 5, 5]
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=2, stride=1, padding=0),
            # 32*[50,8,4,4]->[50,8,16,32]->[50,8,16*32(512)]
            utils.DenseCapsule(in_num_caps=512, in_dim_caps=8, out_num_caps=10, out_dim_caps=16,
                               device=device, routings=3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        return x

# 2: AlexNet-MNIST
class AlexNet_MNIST(nn.Module):
    def __init__(self, device):
        super(AlexNet_MNIST, self).__init__()
        self.conv = nn.Sequential(  # [50, 28, 28]
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=3, stride=1, padding=1),  # [50, 96,28,28]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 96,14,14]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256,14,14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50,256, 7, 7]
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 7, 7]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 7, 7]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256, 7, 7]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)  # [50,256, 5, 5]
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # total class: 10
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        feature = feature.view(img.shape[0], -1)
        output = self.fc(feature)
        return output

# 3: CapsNet-MNIST 32*32
class CapsNet_MNIST(nn.Module):

    def __init__(self, device):
        super(CapsNet_MNIST, self).__init__()
        self.conv = nn.Sequential(  # [50,  3,28,28]
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1, padding=0),  # [batch_size, 256, 20, 20]
            nn.ReLU()
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=9, stride=2, padding=0), # [batch, 8, 32, 6, 6]
            # 32*[bs,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=10, out_dim_caps=16,
                               device=device, routings=3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        return x

# 4: CapsAlexNet-MNIST, capsnet layer is inside Alexnet
class CapsAlexNet_MNIST(nn.Module):
    def __init__(self, device):
        super(CapsAlexNet_MNIST, self).__init__()
        self.conv = nn.Sequential(  # [50, 3,28,28]
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=3, stride=1, padding=1),  # [50, 96,28, 28]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 96,14,14]
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数------------------------------------
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256,14,14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50,256, 7, 7]
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=3, stride=1, padding=0),
            # 32*[50,8,5,5]->[50,8,25,32]->[50,8,25*32(800)]
            utils.DenseCapsule(in_num_caps=800, in_dim_caps=8, out_num_caps=10, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.conv2 = nn.Sequential(  # [50, 1, 10, 16]
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=1, padding=0),  # [50, 256, 8, 14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),  # [50, 256, 3, 6]
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(256*3*6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),  # total class: 10
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        x = x.view(-1, 1, 10, 16)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 5 CapNet-Recon-MNIST
class CapsNet_Recon_MNIST(nn.Module):
    def __init__(self, device):
        super(CapsNet_Recon_MNIST, self).__init__()
        self.conv = nn.Sequential(  # [50,  3,28,28]
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1, padding=0),  # [batch_size, 256, 20, 20]
            nn.ReLU()
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=9, stride=2, padding=0), # [batch, 8, 32, 6, 6]
            # 32*[bs,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=10, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.reconstruction = utils.ReconstructionNet(num_dim=16, num_caps=10, img_size=28, original_chanel=1)

    def forward(self, x, targets):
        feature = self.conv(x)
        v = self.Cap(feature)
        reconstruction = self.reconstruction(v, targets)
        return v, reconstruction

# 6: AlexCapsNet-MNIST
class AlexCapsNet_Recon_MNIST(nn.Module):
    def __init__(self, device):
        super(AlexCapsNet_Recon_MNIST, self).__init__()
        self.conv = nn.Sequential(  # [50, 28, 28]
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=3, stride=1, padding=1),  # [50, 96,28,28]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 96,14,14]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256,14,14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50,256, 7, 7]
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 7, 7]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 7, 7]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256, 7, 7]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)  # [50,256, 5, 5]
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=2, stride=1, padding=0),
            # 32*[50,8,4,4]->[50,8,16,32]->[50,8,16*32(512)]
            utils.DenseCapsule(in_num_caps=512, in_dim_caps=8, out_num_caps=10, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.reconstruction = utils.ReconstructionNet(num_dim=16, num_caps=10, img_size=28, original_chanel=1)

    def forward(self, x, targets):
        feature = self.conv(x)
        v = self.Cap(feature)
        reconstruction = self.reconstruction(v, targets)
        return v, reconstruction

# 7: AlexCapsNet_FullyOut-MNIST
class AlexCapsNet_Fully_MNIST(nn.Module):
    def __init__(self, device):
        super(AlexCapsNet_Fully_MNIST, self).__init__()
        self.conv = nn.Sequential(  # [50, 28, 28]
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=3, stride=1, padding=1),  # [50, 96,28,28]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 96,14,14]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256,14,14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50,256, 7, 7]
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 7, 7]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 7, 7]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256, 7, 7]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)  # [50,256, 5, 5]
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=2, stride=1, padding=0),
            # 32*[50,8,4,4]->[50,8,16,32]->[50,8,16*32(512)]
            utils.DenseCapsule(in_num_caps=512, in_dim_caps=8, out_num_caps=10, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(10*16, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),  # total class: 100
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 8: AlexCapsNet_FullyOut_Recon-MNIST
class AlexCapsNet_Fully_Recon_MNIST(nn.Module):
    def __init__(self, device):
        super(AlexCapsNet_Fully_Recon_MNIST, self).__init__()
        self.conv = nn.Sequential(  # [50, 28, 28]
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=3, stride=1, padding=1),  # [50, 96,28,28]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 96,14,14]
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数------------------------------------
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256,14,14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50,256, 7, 7]
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 7, 7]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50,384, 7, 7]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256, 7, 7]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1)  # [50,256, 5, 5]
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=2, stride=1, padding=0),
            # 32*[50,8,4,4]->[50,8,16,32]->[50,8,16*32(512)]
            utils.DenseCapsule(in_num_caps=512, in_dim_caps=8, out_num_caps=10, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(10*16, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),  # total class: 10
        )
        self.reconstruction = utils.ReconstructionNet(num_dim=16, num_caps=10, img_size=28, original_chanel=1)

    def forward(self, x, targets):
        x = self.conv(x)
        v = self.Cap(x)
        reconstruction = self.reconstruction(v, targets)
        x = self.flatten(v)
        x = self.fc(x)
        return x, reconstruction

# 9: Shollow AlexCapsNet_F
class S_AlexCapsNet_Fully_MNIST(nn.Module):
    def __init__(self, device):
        super(S_AlexCapsNet_Fully_MNIST, self).__init__()
        self.conv = nn.Sequential(  # [50, 28, 28]
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=3, stride=1, padding=1),  # [50, 96,28,28]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 96,14,14]
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数------------------------------------
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256,14,14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50,256, 7, 7]
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=0),  # [50,384, 5, 5]
            nn.ReLU(),
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=384, out_channel=8, kersel_size=2, stride=1, padding=0),
            # 32*[50,8,4,4]->[50,8,16,32]->[50,8,16*32(512)]
            utils.DenseCapsule(in_num_caps=512, in_dim_caps=8, out_num_caps=10, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(10*16, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),  # total class: 100
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 10: Shallow AlexCapsNet
class S_AlexCapsNet_MNIST(nn.Module):

    def __init__(self, device):
        super(S_AlexCapsNet_MNIST, self).__init__()
        self.conv = nn.Sequential(  # [50, 28, 28]
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=3, stride=1, padding=1),  # [50, 96,28,28]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 96,14,14]
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数------------------------------------
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50,256,14,14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50,256, 7, 7]
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=0),  # [50,384, 5, 5]
            nn.ReLU()
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=384, out_channel=8, kersel_size=2, stride=1, padding=0),
            # 32*[50,8,4,4]->[50,8,16,32]->[50,8,16*32(512)]
            utils.DenseCapsule(in_num_caps=512, in_dim_caps=8, out_num_caps=10, out_dim_caps=16,
                               device=device, routings=3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        return x

# ----------------------------------------------------------------------------- FOOD101 512*512->224
# 1: AlexCpsNet-FOOD101
class AlexCapsNet_FOOD101(nn.Module):

    def __init__(self, device):
        super(AlexCapsNet_FOOD101, self).__init__()
        self.conv = nn.Sequential(  # [50,3, 224, 224]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),     # [50, 96, 54, 54]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),                                   # [50,  96, 26, 26]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),    # [50, 256, 26, 26]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),                                   # [50, 384, 13, 13]

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),   # [50, 384, 13, 13]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),   # [50, 384, 13, 13]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),   # [50, 256, 13, 13]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)  # [50,256, 6, 6]
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=3, stride=1, padding=1),
            # 32*[50,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=101, out_dim_caps=16,
                               device=device, routings=3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        return x

# 2：AlexNet-FOOD101
class AlexNet_FOOD101(nn.Module):
    def __init__(self):
        super(AlexNet_FOOD101, self).__init__()
        self.conv = nn.Sequential(  # [50,3, 224, 224]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),     # [50, 96, 54, 54]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),                                   # [50,  96, 26, 26]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),    # [50, 256, 26, 26]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),                                   # [50, 384, 12, 12]

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),   # [50, 384, 12, 12]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),   # [50, 384, 12, 12]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),   # [50, 256, 12, 12]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)  # [50,256, 5, 5]
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 101),    # total class: 101
        )

    def forward(self, img):
        feature = self.conv(img)
        feature = feature.view(img.shape[0], -1)
        output = self.fc(feature)
        return output

# 3: CapsNet-FOOD101 512->112
class CapsNet_FOOD101(nn.Module):

    def __init__(self, device):
        super(CapsNet_FOOD101, self).__init__()
        self.conv = nn.Sequential(  # [50,  3, 112, 112] 112 34
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=11, stride=3, padding=0), # [batch_size, 256, 34, 34]
            nn.ReLU()
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=9, stride=3, padding=0), # [batch, 8, 32, 9, 9]
            # 32*[bs,8,9,9]->[50,8,81,32]->[50,8,81*32(2592)]

            utils.DenseCapsule(in_num_caps=2592, in_dim_caps=8, out_num_caps=101, out_dim_caps=16,
                               device=device, routings=3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        return x

# 4: CapsAlexNet-CIFAR100 32*32,
class CapsAlexNet_FOOD101(nn.Module):

    def __init__(self, device):
        super(CapsAlexNet_FOOD101, self).__init__()
        self.conv = nn.Sequential(          # [50,  3, 224, 224]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),     # [50, 96, 54, 54]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),                                   # [50,  96, 26, 26]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),    # [50, 256, 26, 26]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),                                   # [50, 384, 12, 12]
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=4, stride=1, padding=0),
            # 32*[50,8,9,9]->[50,8,81,32]->[50,8,81*32(2592)]
            utils.DenseCapsule(in_num_caps=2592, in_dim_caps=8, out_num_caps=101, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.conv2 = nn.Sequential(  # [1, 101, 16]
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=1, padding=0),  # [50, 256, 99, 14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),  # [50, 256, 49, 6]
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(256*49*6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 101),  # total class: 100
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        x = x.view(-1, 1, 100, 16)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 5: CapNet-Recon-CIFAR100 input is 112
class CapsNet_Recon_FOOD101(nn.Module):
    def __init__(self, device):
        super(CapsNet_Recon_FOOD101, self).__init__()
        self.conv = nn.Sequential(  # [50,  3, 112, 112] 112 34
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=11, stride=3, padding=0), # [batch_size, 256, 34, 34]
            nn.ReLU()
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=9, stride=3, padding=0), # [batch, 8, 32, 9, 9]
            # 32*[bs,8,9,9]->[50,8,81,32]->[50,8,81*32(2592)]

            utils.DenseCapsule(in_num_caps=2592, in_dim_caps=8, out_num_caps=101, out_dim_caps=16,
                               device=device, routings=3)           # [101, 16]
        )
        self.reconstruction = utils.ReconstructionNet(num_dim=16, num_caps=101, img_size=112, original_chanel=3)

    def forward(self, x, targets):
        feature = self.conv(x)
        v = self.Cap(feature)
        reconstruction = self.reconstruction(v, targets)
        return v, reconstruction

# 6: AlexCapsNet with Reconstruction-CIFAR100
class AlexCapsNet_Recon_FOOD101(nn.Module):
    def __init__(self, device):
        super(AlexCapsNet_Recon_FOOD101, self).__init__()
        self.conv = nn.Sequential(                                                              # [50,3, 224, 224]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),     # [50, 96, 54, 54]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),                                   # [50,  96, 26, 26]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),    # [50, 256, 26, 26]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),                                   # [50, 384, 13, 13]

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),   # [50, 384, 13, 13]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),   # [50, 384, 13, 13]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),   # [50, 256, 13, 13]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)  # [50,256, 6, 6]
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=3, stride=1, padding=1),
            # 32*[50,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=101, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.Recon = utils.ReconstructionNet(num_dim=16, num_caps=101, img_size=224, original_chanel=3)

    def forward(self, x, targets):
        feature = self.conv(x)
        v = self.Cap(feature)
        reconstruction = self.Recon(v, targets)
        return v, reconstruction

# 7: AlexCapsNet_FullyOut-MNIST
class AlexCapsNet_Fully_FOOD101(nn.Module):
    def __init__(self, device):
        super(AlexCapsNet_Fully_FOOD101, self).__init__()
        self.conv = nn.Sequential(  # [50,3, 224, 224]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),  # [50, 96, 54, 54]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),  # [50,  96, 26, 26]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),  # [50, 256, 26, 26]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 384, 13, 13]

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50, 384, 13, 13]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50, 384, 13, 13]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50, 256, 13, 13]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)  # [50,256, 6, 6]
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=3, stride=1, padding=1),
            # 32*[50,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=101, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(101*16, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 101),  # total class: 101
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 8: AlexCapsNet_Fully_Recon
class AlexCapsNet_F_Recon_FOOD101(nn.Module):
    def __init__(self, device):
        super(AlexCapsNet_F_Recon_FOOD101, self).__init__()
        self.conv = nn.Sequential(  # [50,3, 224, 224]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),  # [50, 96, 54, 54]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),  # [50,  96, 26, 26]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),  # [50, 256, 26, 26]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 384, 13, 13]

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50, 384, 13, 13]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50, 384, 13, 13]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50, 256, 13, 13]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)  # [50,256, 6, 6]
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=3, stride=1, padding=1),
            # 32*[50,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=101, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(101 * 16, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 101),  # total class: 101
        )
        self.reconstruction = utils.ReconstructionNet(num_dim=16, num_caps=101, img_size=224, original_chanel=3)

    def forward(self, x, targets):
        x = self.conv(x)
        v = self.Cap(x)
        reconstruction = self.reconstruction(v, targets)
        x = self.flatten(v)
        x = self.fc(x)
        return x, reconstruction

# 9: Shollow AlexCapsNet_F
class S_AlexCapsNet_Fully_FOOD101(nn.Module):
    def __init__(self, device):
        super(S_AlexCapsNet_Fully_FOOD101, self).__init__()
        self.conv = nn.Sequential(                                                              # [50,  3,224, 224]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),     # [50, 96, 54,  54]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),                                   # [50, 96,26,26]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=0),    # [50,256,22,22]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),                                   # [50,256, 10, 10]
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=0),   # [50,384,  8,  8]
            nn.ReLU(),
        )


        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=384, out_channel=8, kersel_size=3, stride=1, padding=0),
            # 32*[50,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=101, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(101*16, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 101),  # total class: 100
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 10: Shallow AlexCapsNet
class S_AlexCapsNet_FOOD101(nn.Module):

    def __init__(self, device):
        super(S_AlexCapsNet_FOOD101, self).__init__()
        self.conv = nn.Sequential(                                                              # [50,  3,224, 224]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),     # [50, 96, 54,  54]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),                                   # [50, 96,26,26]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=0),    # [50,256,22,22]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),                                   # [50,256, 10, 10]
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=0),   # [50,384,  8,  8]
            nn.ReLU(),
        )


        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=384, out_channel=8, kersel_size=3, stride=1, padding=0),
            # 32*[50,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=101, out_dim_caps=16,
                               device=device, routings=3)
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        return x

#--------------------------------------------------------------------Flower102 ->[224,224]

# 1: AlexCpsNet-FLOWER102
class AlexCapsNet_FLOWER102(nn.Module):

    def __init__(self, device):
        super(AlexCapsNet_FLOWER102, self).__init__()
        self.conv = nn.Sequential(  # [50,3, 224, 224]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),     # [50, 96, 54, 54]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),                                   # [50,  96, 26, 26]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),    # [50, 256, 26, 26]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),                                   # [50, 384, 13, 13]

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),   # [50, 384, 13, 13]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),   # [50, 384, 13, 13]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),   # [50, 256, 13, 13]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)  # [50,256, 6, 6]
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=3, stride=1, padding=1),
            # 32*[50,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=102, out_dim_caps=16,
                               device=device, routings=3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        return x

# 2：AlexNet-FLOWER102
class AlexNet_FLOWER102(nn.Module):
    def __init__(self):
        super(AlexNet_FLOWER102, self).__init__()
        self.conv = nn.Sequential(  # [50,3, 224, 224]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),     # [50, 96, 54, 54]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),                                   # [50,  96, 26, 26]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),    # [50, 256, 26, 26]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),                                   # [50, 384, 12, 12]

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),   # [50, 384, 12, 12]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),   # [50, 384, 12, 12]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),   # [50, 256, 12, 12]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)  # [50,256, 5, 5]
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 102),    # total class: 101
        )

    def forward(self, img):
        feature = self.conv(img)
        feature = feature.view(img.shape[0], -1)
        output = self.fc(feature)
        return output

# 3: CapsNet-FLOWER102 512->112
class CapsNet_FLOWER102(nn.Module):

    def __init__(self, device):
        super(CapsNet_FLOWER102, self).__init__()
        self.conv = nn.Sequential(  # [50,  3, 112, 112] 112 34
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=11, stride=3, padding=0), # [batch_size, 256, 34, 34]
            nn.ReLU()
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=9, stride=3, padding=0), # [batch, 8, 32, 9, 9]
            # 32*[bs,8,9,9]->[50,8,81,32]->[50,8,81*32(2592)]

            utils.DenseCapsule(in_num_caps=2592, in_dim_caps=8, out_num_caps=102, out_dim_caps=16,
                               device=device, routings=3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        return x

# 4: CapsAlexNet-FLOWER102 32*32,
class CapsAlexNet_FLOWER102(nn.Module):

    def __init__(self, device):
        super(CapsAlexNet_FLOWER102, self).__init__()
        self.conv = nn.Sequential(          # [50,  3, 224, 224]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),     # [50, 96, 54, 54]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),                                   # [50,  96, 26, 26]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),    # [50, 256, 26, 26]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),                                   # [50, 384, 12, 12]
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=4, stride=1, padding=0),
            # 32*[50,8,9,9]->[50,8,81,32]->[50,8,81*32(2592)]
            utils.DenseCapsule(in_num_caps=2592, in_dim_caps=8, out_num_caps=102, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.conv2 = nn.Sequential(  # [1, 102, 16]
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=2, padding=0),  # [50, 256, 50, 7]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),   # [50, 256, 24, 3]
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(256*24*3, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 102),  # total class: 102
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        x = x.view(-1, 1, 102, 16)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 5: CapNet-Recon-CIFAR100 input is 112
class CapsNet_Recon_FLOWER102(nn.Module):
    def __init__(self, device):
        super(CapsNet_Recon_FLOWER102, self).__init__()
        self.conv = nn.Sequential(  # [50,  3, 112, 112] 112 34
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=11, stride=3, padding=0), # [batch_size, 256, 34, 34]
            nn.ReLU()
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=9, stride=3, padding=0), # [batch, 8, 32, 9, 9]
            # 32*[bs,8,9,9]->[50,8,81,32]->[50,8,81*32(2592)]
            utils.DenseCapsule(in_num_caps=2592, in_dim_caps=8, out_num_caps=102, out_dim_caps=16,
                               device=device, routings=3)           # [101, 16]
        )
        self.reconstruction = utils.ReconstructionNet(num_dim=16, num_caps=102, img_size=112, original_chanel=3)

    def forward(self, x, targets):
        feature = self.conv(x)
        v = self.Cap(feature)
        reconstruction = self.reconstruction(v, targets)
        return v, reconstruction

# 6: AlexCapsNet with Reconstruction-CIFAR100
class AlexCapsNet_Recon_FLOWER102(nn.Module):
    def __init__(self, device):
        super(AlexCapsNet_Recon_FLOWER102, self).__init__()
        self.conv = nn.Sequential(                                                              # [50,3, 224, 224]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),     # [50, 96, 54, 54]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),                                   # [50,  96, 26, 26]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),    # [50, 256, 26, 26]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),                                   # [50, 384, 13, 13]

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),   # [50, 384, 13, 13]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),   # [50, 384, 13, 13]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),   # [50, 256, 13, 13]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)  # [50,256, 6, 6]
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=3, stride=1, padding=1),
            # 32*[50,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=102, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.Recon = utils.ReconstructionNet(num_dim=16, num_caps=102, img_size=224, original_chanel=3)

    def forward(self, x, targets):
        feature = self.conv(x)
        v = self.Cap(feature)
        reconstruction = self.Recon(v, targets)
        return v, reconstruction

# 7: AlexCapsNet_FullyOut-MNIST
class AlexCapsNet_Fully_FLOWER102(nn.Module):
    def __init__(self, device):
        super(AlexCapsNet_Fully_FLOWER102, self).__init__()
        self.conv = nn.Sequential(  # [50,3, 224, 224]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),  # [50, 96, 54, 54]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),  # [50,  96, 26, 26]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),  # [50, 256, 26, 26]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 384, 13, 13]

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50, 384, 13, 13]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50, 384, 13, 13]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50, 256, 13, 13]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)  # [50,256, 6, 6]
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=3, stride=1, padding=1),
            # 32*[50,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=102, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(102*16, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 102),  # total class: 101
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 8: AlexCapsNet_Fully_Recon
class AlexCapsNet_F_Recon_FLOWER102(nn.Module):
    def __init__(self, device):
        super(AlexCapsNet_F_Recon_FLOWER102, self).__init__()
        self.conv = nn.Sequential(  # [50,3, 224, 224]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),  # [50, 96, 54, 54]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),  # [50,  96, 26, 26]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),  # [50, 256, 26, 26]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # [50, 384, 13, 13]

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50, 384, 13, 13]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),  # [50, 384, 13, 13]
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # [50, 256, 13, 13]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)  # [50,256, 6, 6]
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=256, out_channel=8, kersel_size=3, stride=1, padding=1),
            # 32*[50,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=102, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(102 * 16, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 102),  # total class: 102
        )
        self.reconstruction = utils.ReconstructionNet(num_dim=16, num_caps=102, img_size=224, original_chanel=3)

    def forward(self, x, targets):
        x = self.conv(x)
        v = self.Cap(x)
        reconstruction = self.reconstruction(v, targets)
        x = self.flatten(v)
        x = self.fc(x)
        return x, reconstruction

# 9: Shollow AlexCapsNet_F
class S_AlexCapsNet_Fully_FLOWER102(nn.Module):
    def __init__(self, device):
        super(S_AlexCapsNet_Fully_FLOWER102, self).__init__()
        self.conv = nn.Sequential(                                                              # [50,  3,224, 224]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),     # [50, 96, 54,  54]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),                                   # [50, 96,26,26]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=0),    # [50,256,22,22]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),                                   # [50,256, 10, 10]
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=0),   # [50,384,  8,  8]
            nn.ReLU(),
        )
        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=384, out_channel=8, kersel_size=3, stride=1, padding=0),
            # 32*[50,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=102, out_dim_caps=16,
                               device=device, routings=3)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(102*16, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 102),  # total class: 100
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 10: Shallow AlexCapsNet
class S_AlexCapsNet_FLOWER102(nn.Module):

    def __init__(self, device):
        super(S_AlexCapsNet_FLOWER102, self).__init__()
        self.conv = nn.Sequential(                                                              # [50,  3,224, 224]
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),     # [50, 96, 54,  54]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),                                   # [50, 96,26,26]
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=0),    # [50,256,22,22]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),                                   # [50,256, 10, 10]
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=0),   # [50,384,  8,  8]
            nn.ReLU(),
        )

        self.Cap = nn.Sequential(
            utils.PrimaryCaps(num_caps=32, in_channel=384, out_channel=8, kersel_size=3, stride=1, padding=0),
            # 32*[50,8,6,6]->[50,8,36,32]->[50,8,36*32(1152)]
            utils.DenseCapsule(in_num_caps=1152, in_dim_caps=8, out_num_caps=102, out_dim_caps=16,
                               device=device, routings=3)
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.Cap(x)
        return x