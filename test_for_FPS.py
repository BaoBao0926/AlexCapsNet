import shutil
import time
import copy
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
import re

from thop import profile		 ## 导入thop模块
def cal_params_flopss(model, size, batch_size=1, d_model=3, channel_first=True):
    if torch.cuda.is_available():
        input = torch.randn(batch_size, d_model, size, size).cuda() if channel_first else torch.randn(1, size, size, d_model)
    else:
        input = torch.randn(batch_size, d_model, size, size) if channel_first else torch.randn(1, size, size, d_model)
    flops, params = profile(model, inputs=(input,))
    if flops < 1e6:
        print('flops K', flops/1e3)
    else:
        print('flops',flops/1e9)			## 打印计算量
    print('params',params/1e6)			## 打印参数量

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total/1e6))
    # logger.info(f'flops: {flops/1e9}, params: {params/1e6}, Total params: : {total/1e6:.4f}')
#
from AlexNet_Module import *


print("the image size is [3,32,32]. for CIFAR10 and SVHN")
ACN = AlexCapsNet_CIFAR10('cpu')
CN = CapsNet_CIFAR10('cpu')
AN = AlexNet_CIFAR10()
image = torch.zeros(1, 3, 32, 32)

start = time.time()
for i in range(100):
    out = ACN(image)
print(f'ACN: {time.time()-start}, {100/(time.time()-start)}')

start = time.time()
for i in range(100):
    out = CN(image)
print(f'CN: {time.time()-start}, {100/(time.time()-start)}')

start = time.time()
for i in range(100):
    out = AN(image)
print(f'AN: {time.time()-start}, {100/(time.time()-start)}')



print("the image size is [1,28,28]. for MNIST and F-MNIST")
ACN = AlexCapsNet_MNIST('cpu')
CN = CapsNet_MNIST('cpu')
AN = AlexNet_MNIST('cpu')
image = torch.zeros(1, 1, 28, 28)

start = time.time()
for i in range(100):
    out = ACN(image)
print(f'ACN: {time.time()-start}, {100/(time.time()-start)}')

start = time.time()
for i in range(100):
    out = CN(image)
print(f'CN: {time.time()-start}, {100/(time.time()-start)}')

start = time.time()
for i in range(100):
    out = AN(image)
print(f'AN: {time.time()-start}, {100/(time.time()-start)}')