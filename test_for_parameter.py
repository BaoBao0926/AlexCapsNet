import torch.cuda
from thop import profile		 ## 导入thop模块
from AlexNet_Module import *
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def cal_params_flopss(model, size, d_model=3, channel_first=True):
    if torch.cuda.is_available():
        input = torch.randn(1, d_model, size, size).cuda() if channel_first else torch.randn(1, size, size, d_model)
    else:
        input = torch.randn(1, d_model, size, size) if channel_first else torch.randn(1, size, size, d_model)
    flops, params = profile(model, inputs=(input,))
    # if flops < 1e6:
    #     print('flops K', flops/1e3)
    # else:
    print('flops',flops/1e6)			## 打印计算量
    print('params',params/1e6)			## 打印参数量

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total/1e6))
    # logger.info(f'flops: {flops/1e9}, params: {params/1e6}, Total params: : {total/1e6:.4f}')

# # print('this is for capsnet') flops 537.406464  params 5.401856  Total params: 39.24M
# cal_params_flopss(CapsNet_FLOWER102(device='cpu'), size=112, d_model=3)
# # print('this is for alexnet') flops 1006.26752 params 47.164902  Total params: 47.16M
# cal_params_flopss(AlexNet_FLOWER102(), size=224, d_model=3)
# # print('this is for alexcapsnet') flops 1061.506176        params 4.33728      Total params: 19.38M
# cal_params_flopss(AlexCapsNet_FLOWER102(device='cpu'), size=224, d_model=3)

# # print('this is for capsnet') flops 199.397376     params 5.329664     Total params: 6.80M
# cal_params_flopss(CapsNet_MNIST(device='cpu'), size=28, d_model=1)
# # print('this is for alexnet') flops 238.79424  params 46.360778    Total params: 46.36M
# cal_params_flopss(AlexNet_MNIST(device='cpu'), size=28, d_model=1)
# # print('this is for alexcapsnet') flops 199.955968     params 3.5824   Total params: 4.24M
# cal_params_flopss(AlexCapsNet_MNIST(device='cpu'), size=28, d_model=1)

# print('this is for capsnet') flops 375.570432     params 5.371136     Total params: 7.99M
cal_params_flopss(CapsNet_CIFAR10(device='cpu'), size=32, d_model=3)
# print('this is for alexnet') flops 312.025088     params 57.896842    Total params: 57.90M
cal_params_flopss(AlexNet_CIFAR10(), size=32, d_model=3)
# print('this is for alexcapsnet') flops 278.69184 params 3.911808      Total params: 5.39M
cal_params_flopss(AlexCapsNet_CIFAR10(device='cpu'), size=32, d_model=3)



