import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
# from scipy import interp
from scipy import interpolate as interp
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize

from AlexNet_Module import *
from utils import vtoNorm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 绘制混淆矩阵的函数
def plot_confusion_matrix(cm, labels_name, dataset_name=None, model_name=None, title="Confusion Matrix",  is_norm=True,
                          colorbar=True, cmap=plt.cm.Blues):
    """
    :param cm:  y_true
    :param labels_name:
    :param title:
    :param is_norm:
    :param colorbar:
    :param cmap:
    :return:
    """

    plt.figure(figsize=(8, 7))

    if is_norm==True:
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],2)  # 横轴归一化并保留2位小数

    plt.imshow(cm, interpolation='nearest', cmap=cmap)  # 在特定的窗口上显示图像
    for i in range(len(cm)):
        for j in range(len(cm)):
            # plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center') # 默认所有值均为黑色
            # for SVHN
            if (i == 0 and j == 0) or (i==5 and j==5) or (i == 6 and j == 6) or (i == 7 and j == 7) or (i == 8 and j == 8) or (i == 9 and j == 9):
                plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', color="black" if i==j else "black", verticalalignment='center', fontsize=15) # 将对角线值设为白色 "orange" “orange” "darkorange"
            else:
                plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', color="white" if i==j else "black", verticalalignment='center', fontsize=15) # 将对角线值设为白色 "orange" “orange” "darkorange"
    if colorbar:
        plt.colorbar() # 创建颜色条

    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=30)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
    plt.title(model_name + " in " + dataset_name, fontsize=20)  # 图像标题
    plt.ylabel('Ground-Truth Label', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=16)

    if is_norm==True:
        plt.savefig(r'D:\Learning_Rescoure\extra\Project\1.AlexCapsNets\picture\ROC_Confusion\\'+dataset_name + '_' +
                    model_name + '_confusion_matrix.png', format='png')
    else:
        plt.savefig(r'D:\Learning_Rescoure\extra\Project\1.AlexCapsNets\picture\ROC_Confusion\\cm_'+dataset_name + '_' +
                    model_name + '_confusion_matrix.png', format='png')
    plt.show() # plt.show()在plt.savefig()之后
    plt.close()


def plot_matrix(y_true, y_pred, label_name, dataset_name, model_name):
    """这里需要argumax之后的"""
    # y_true = [2, 0, 2, 2, 0, 1] # 真实标签
    # y_pred = [0, 0, 2, 2, 0, 2] # 预测标签
    # label_name = ['ant', 'bird', 'cat']
    cm = confusion_matrix(y_true, y_pred) # 调用库函数confusion_matrix
    plot_confusion_matrix(cm, label_name, dataset_name, model_name, "Confusion Matrix", is_norm=False,) # 调用上面编写的自定义函数
    # plot_confusion_matrix(cm, label_name, dataset_name, model_name, "Confusion Matrix", is_norm=True) # 经过归一化的混淆矩阵

def plot_ROC_ACU(y_test, y_score, n_classes, classes_name, dataset_name, model_name):
    """y_test是label，y_score为得分没有arguemax"""
    y_test = label_binarize(y_test, classes=np.arange(n_classes))  # 将标签进行二进制编码

    fpr_micro, tpr_micro, _ = roc_curve(y_test.ravel(), np.array(y_score).ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    # 计算 macro-average ROC-AUC
    n_classes = y_test.shape[1]
    fpr_macro = dict()
    tpr_macro = dict()
    roc_auc_macro = dict()

    for i in range(n_classes):
        fpr_macro[i], tpr_macro[i], _ = roc_curve(y_test[:, i], np.array(y_score)[:, i])
        roc_auc_macro[i] = auc(fpr_macro[i], tpr_macro[i])

    # 绘制 micro-average ROC 曲线
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_micro, tpr_micro, color='deeppink', lw=2,
             label='Average ROC curve (area = {0:0.2f})'.format(roc_auc_micro))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Micro-average ROC Curve')
    # plt.legend(loc="lower right")
    # plt.show()

    # 绘制每个类别的 ROC 曲线
    # plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, color in zip(range(n_classes), colors):
        if dataset_name == 'SVHN' or dataset_name == 'MNIST':
            plt.plot(fpr_macro[i], tpr_macro[i], color=color, lw=2,
                 label='Class {0} (area = {1:0.2f})'.format(classes_name[i], roc_auc_macro[i]))
        else:
            plt.plot(fpr_macro[i], tpr_macro[i], color=color, lw=2,
                 label='{0} (area = {1:0.2f})'.format(classes_name[i], roc_auc_macro[i]))

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=30)
    plt.ylabel('True Positive Rate', fontsize=30)
    plt.title(model_name + ' in ' + dataset_name, fontsize=30)
    plt.legend(loc="lower right", fontsize=18)
    plt.savefig(r'D:\Learning_Rescoure\extra\Project\1.AlexCapsNets\picture\ROC_Confusion\\' + dataset_name + '_'
                + model_name + '_ROCcurve.png', format='png')
    plt.show()



def main(model, dataset_name, label_name, batch_size, model_name):
    model = model.to(device)
    model.eval()
    test_loader, n_classes = None, 10
    if dataset_name == 'MNIST':
        transform_test = transforms.Compose([transforms.ToTensor()])
        test_data = torchvision.datasets.MNIST(root='./datasets/MNIST/',
                                               train=False, download=True, transform=transform_test)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    elif dataset_name == 'SVHN':
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_data = torchvision.datasets.SVHN(root='./datasets/SVHN/',
                                              split='test', download=True, transform=transform_test)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    elif dataset_name == 'CIFAR10':
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_data = torchvision.datasets.CIFAR10(root='./datasets/CIFAR10/',
                                                 train=False, download=True, transform=transform_test)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    elif dataset_name == 'FMNIST':
        transform_test = transforms.Compose([transforms.ToTensor()])
        test_data = torchvision.datasets.FashionMNIST(root='./datasets/FashionMNIST/',
                                                      train=False, download=True, transform=transform_test)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    # 测试模型
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for images, labels in test_loader:
            # 1. 使用模型进行前向传播，获取输出
            outputs = model(images)

            if model_name == 'CapsNet' or model_name == 'AlexCapsNet':
                outputs = vtoNorm(outputs)

            # 2. 使用 torch.max 获取每个样本的预测类别索引
            _, predicted = torch.max(outputs, 1)

            # 3. 将真实标签和预测标签转换为 numpy 数组，并添加到 y_true 和 y_pred 列表中
            y_true.extend(labels.numpy())  # 真实标签
            y_pred.extend(predicted.numpy())  # 预测标签

            # 4. 计算 softmax 得分，将结果转换为 numpy 数组并添加到 y_scores 列表中
            #    这里假设你希望获取每个类别的概率分布
            y_scores.extend(F.softmax(outputs, dim=1).numpy())

    plot_matrix(y_true, y_pred, label_name=label_name, dataset_name=dataset_name, model_name=model_name)
    # plot_ROC_ACU(y_true, y_scores, n_classes, classes_name=label_name, dataset_name=dataset_name, model_name=model_name)


def start(model_name, dataset_name, batch_size):
    if model_name == 'AlexNet':
        if dataset_name == 'MNIST':
            model = AlexNet_MNIST(device).to(device)
            # model.load_state_dict(torch.load('./result/MNIST/AlexNet/best_accuracy.pth'))  # 加载已经训练好的模型权重
            model.load_state_dict(torch.load('./result/MNIST/AlexNet/best_accuracy.pth', map_location=torch.device('cpu')))
            label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        if dataset_name == 'SVHN':
            model = AlexNet_CIFAR10().to(device)
            model.load_state_dict(torch.load('./result/SVHN/AlexNet/best_accuracy.pth', map_location=torch.device('cpu')))  # 加载已经训练好的模型权重
            label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        if dataset_name == 'FMNIST':
            model = AlexNet_MNIST(device).to(device)
            model.load_state_dict(torch.load('./result/FashionMNIST/AlexNet/best_accuracy.pth', map_location=torch.device('cpu')))
            label_name = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
        if dataset_name == 'CIFAR10':
            model = AlexNet_CIFAR10().to(device)
            model.load_state_dict(torch.load('./result/CIFAR10/AlexNet/best_accuracy.pth', map_location=torch.device('cpu')))  # 加载已经训练好的模型权重
            label_name = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    if model_name == 'CapsNet':
        if dataset_name == 'MNIST':
            model = CapsNet_MNIST(device).to(device)
            model.load_state_dict(torch.load('./result/MNIST/CapsNet/best_accuracy.pth', map_location=torch.device('cpu')))  # 加载已经训练好的模型权重
            label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        if dataset_name == 'SVHN':
            model = CapsNet_CIFAR10(device).to(device)
            model.load_state_dict(torch.load('./result/SVHN/CapsNet/best_accuracy.pth', map_location=torch.device('cpu')))  # 加载已经训练好的模型权重
            label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        if dataset_name == 'FMNIST':
            model = CapsNet_MNIST(device).to(device)
            model.load_state_dict(torch.load('./result/FashionMNIST/CapsNet/best_accuracy.pth', map_location=torch.device('cpu')))
            label_name = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
        if dataset_name == 'CIFAR10':
            model = CapsNet_CIFAR10(device).to(device)
            model.load_state_dict(torch.load('./result/CIFAR10/CapsNet/best_accuracy.pth', map_location=torch.device('cpu')))  # 加载已经训练好的模型权重
            label_name = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    if model_name == 'AlexCapsNet':
        if dataset_name == 'MNIST':
            model = AlexCapsNet_MNIST(device).to(device)
            model.load_state_dict(torch.load('./result/MNIST/AlexCapsNet/best_accuracy.pth', map_location=torch.device('cpu')))  # 加载已经训练好的模型权重
            label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        if dataset_name == 'SVHN':
            model = AlexCapsNet_CIFAR10(device).to(device)
            model.load_state_dict(torch.load('./result/SVHN/AlexCapsNet/best_accuracy.pth', map_location=torch.device('cpu')))  # 加载已经训练好的模型权重
            label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        if dataset_name == 'FMNIST':
            model = AlexCapsNet_MNIST(device).to(device)
            model.load_state_dict(torch.load('./result/FashionMNIST/AlexCapsNet/best_accuracy.pth', map_location=torch.device('cpu')))
            label_name = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
        if dataset_name == 'CIFAR10':
            model = AlexCapsNet_CIFAR10(device).to(device)
            model.load_state_dict(torch.load('./result/CIFAR10/AlexCapsNet/best_accuracy.pth', map_location=torch.device('cpu')))  # 加载已经训练好的模型权重
            label_name = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    main(model=model, dataset_name=dataset_name, label_name=label_name, batch_size=batch_size, model_name=model_name)

if __name__ == '__main__':
    m_name = ['AlexNet', 'CapsNet', 'AlexCapsNet']
    set_name = ['MNIST', 'SVHN', 'FMNIST', 'CIFAR10']

    # start(model_name=m_name[0], dataset_name=set_name[1], batch_size=16)
    # for i in range(3):
    #     for j in range(4):
    #         print(f'this is for {m_name[i], i} and in {set_name[j], j}')
    #         start(model_name=m_name[i], dataset_name=set_name[j], batch_size=16)

    j = 1
    for i in range(3):
        print(f'this is for {m_name[i], i} and in {set_name[j], j}')
        start(model_name=m_name[i], dataset_name=set_name[j], batch_size=16)
