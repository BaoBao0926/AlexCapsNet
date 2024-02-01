import shutil
import time
import copy
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import re

# def get_data(file, mode):
#     # 用于匹配accuracy的正则表达式模式
#     accuracy_pattern = re.compile(f'{mode}: (\d+\.\d+)')
#
#     # 用于存储提取出的accuracy数据的列表
#     accuracy_values = []
#
#     # 打开文本文件并逐行读取内容
#     with open(file, 'r') as file:
#         # 遍历文件的每一行
#         for line in file:
#             # 尝试在当前行中匹配accuracy的模式
#             match = accuracy_pattern.search(line)
#
#             # 如果匹配成功，则提取accuracy的数值并添加到列表中
#             if match:
#                 accuracy_value = float(match.group(1))
#                 accuracy_values.append(accuracy_value)
#
#     # 打印提取出的所有accuracy数值
#     # if accuracy_values[0] > 1000:
#     #     accuracy_values = [x / 100 for x in accuracy_values]
#
#     print("所有accuracy的数值:", accuracy_values)
#     print('legth', len(accuracy_values))
#
#
# # accuracy的两个
# file_path = r'D:\Learning_Rescoure\extra\Nets\Result\CIFAR10\Alexnet\3.CapsNet\train\best_train' + r'\eval_result.txt'
# get_data(file_path, 'The accuracy')
# # loss的两个
# print("AlexCpasNet loss:")
# file_path = r'D:\Learning_Rescoure\extra\Nets\Result\CIFAR10\Alexnet\1.AlexCapsNet\train\best_train' + r'\train_result.txt'
# get_data(file_path, 'Total Loss')
#
# print('AlexNet loss:')
# file_path = r'D:\Learning_Rescoure\extra\Nets\Result\CIFAR10\Alexnet\2.AlexNet\train\best_train' + r'\train_result.txt'
# get_data(file_path, 'Total Loss')
#
# print('CapsNet loss:')
# file_path = r'D:\Learning_Rescoure\extra\Nets\Result\CIFAR10\Alexnet\3.CapsNet\train\best_train' + r'\train_result.txt'
# get_data(file_path, 'Total Loss')

a = [1,2,3,4]
b = [5,6,7,8]

for i in a:
    print(i)
    for j in b:
        print(j)