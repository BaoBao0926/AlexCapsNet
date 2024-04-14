## Hey! This is [Muyi Bao](https://github.com/BaoBao0926/BaoBao0926.github.io/tree/main)! 👋👋

[English](./README.md) | [简体中文](./README-Chinese.md)

# 1.Introduction to AlexCapsNet

这是一个将AlexNet和CapsNet集成在一起，提高Capsule Network性能的项目，我们称之为AlexCapsNet，用于图像分类。我们提高了CapsNet的性能在5个数据集的评估中。此外，我们还评估了不同层数的特征提取层在具有大量类别和少量类别的数据集上的性能，以及七个数据集上具有和不具有噪声的数据集上的reconstruction module。

-所有权重文件都可以在“结果”文件夹中找到

-所有代码都可以在文件夹中找到:
- AlexNet.py:是储存训练模型的代码
- AlexNet_Module.py：储存模型网络的代码
- recon_visual.py用于将重建的图像可视化的代码
- utils.py是保存一些方法，如训练和评估代码，Dataloader和其他一些简单的方法。

-主要贡献如下:

  -我们提出了一种新的架构，即AlexCapsNet，它利用AlexNet作为特征提取层，为CapsNet捕获更深层次和更多的语义特征。与CapsNet的其他variants相比，这种集成提高了性能。

  -我们提出了Shallow-AlexNet (S-AlexNet)模块，其层数比AlexNet少。通过实验，我们发现S-AlexNet更适合于类别多的数据集，AlexNet模块在类别少的数据集上表现出色。这一观察结果强调了特征提取层的深度在CapsNet性能中的关键作用。

  -通过对不同特征提取层模块的进一步研究，我们发现重构模块在没有噪声的数据集中更有效，而在有背景噪声的数据集中表现较差。这一观察结果显示了重建模块的潜在脆弱性和局限性。


  # 2.Network Architecture

<!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  -->
<details>
  
<summary>
  AlexNet的网络架构:
</summary>

<br />

 ![image](https://github.com/BaoBao0926/AlexCapsNet/blob/main/picture/AlexNet.png)
 
</details>
<!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  -->
<details>
  
<summary>
  Capsule Network的网络架构:
</summary>

<br />

 ![image](https://github.com/BaoBao0926/AlexCapsNet/blob/main/picture/CapsNet.png)
 
</details>
  <!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  -->
<details>
  
<summary>
  AlexCapsNet的网络架构:
</summary>

<br />

 ![image](https://github.com/BaoBao0926/AlexCapsNet/blob/main/picture/AlexCapsNet.png)
 
</details>
 <!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  -->
<details>
  
<summary>
  Shallow-AlexCapsNet的网络架构:
</summary>

<br />

 ![image](https://github.com/BaoBao0926/AlexCapsNet/blob/main/picture/Shallow%20AlexCapsNet.png)

 更多关于Shallow-AlexNet Module的细节：
 
 ![image](https://github.com/BaoBao0926/AlexCapsNet/blob/main/picture/S-ACN-M.png)
</details>
 <!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  -->
<details>
  
<summary>
  带有Reconstruction Module的AlexCapsNet or S-AlexCapsNet的网络架构
</summary>

<br />

 ![image](https://github.com/BaoBao0926/AlexCapsNet/blob/main/picture/ACN-Recon.png)

</details>

# 3.Experimental Dataset

本文用到的所有数据集都是 built-in methods in [torchvision](https://pytorch.org/vision/stable/datasets.html#built-in-datasets)

| Dataset           | [MNIST](https://yann.lecun.com/exdb/mnist/)   | [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) | [SVHN](http://ufldl.stanford.edu/housenumbers/)   | [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) | [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) | [FOOD101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)   | [FLOWER102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)  |
| -------           | ------- | -------       |  ---   | ------- | -------  | -------   | -------    | 
| 类别数量 | 10      | 10            | 10     | 10      |  100     | 101       |  102       |
| 图片大小        | 28×28   | 28×28         | 32×32  | 32×32   |  32×32   | ＜512×512 |  ＞500×500 |
| 通道数量    | 1       | 1             |  3     | 3       |  3       | 3         |    3       |
| 训练样本  | 50,000  | 50,000        | 73,257 | 50,000  |  50,000  |  75.750   |   6149     |
| 测试样本   | 10,000  | 10,000        | 26,032 | 10,000  |  10,000  |  25,250   |   1020     |   
| 是否有noise           | No      | No            | Yes    | Yes     |  Yes     |  Yes      | Yes        |


<img src="https://github.com/BaoBao0926/AlexCapsNet/blob/main/picture/datasetImage.png" width="500">


# 4.Experiment Results

## 4.1 Performance Analysis of AlexCapsNet

我们使用了CapsNet相关文章中常用的四个数据集作为评估, MNIST, F-MNIST, SVHN, CIFAR10 and 使用 **准确率** 作为评价指标

| Model | MNIST | FMNIST | SVHN | CIFAR10 |
|------ | ----- | -----  | ---- |-------  |
|[AlexNet](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)|99.18|91.82|94.00|72.53|
|[CapsNet](https://proceedings.neurips.cc/paper_files/paper/2017/hash/2cad8fa47bbef282badbb8de5374b894-Abstract.html)|99.21|90.31|91.30|70.46|
|[LE-CapsNet](https://ieeexplore.ieee.org/abstract/document/9680004)|-|93.04|92.62|76.73|
|[MS-CapsNet](https://ieeexplore.ieee.org/abstract/document/8481393)|-|92.27|-|75.70|
|[CFC-CapsNet](https://dl.acm.org/doi/abs/10.1145/3441110.3441148)|99.63|92.86|93.29|73.15|
|[FSc-CapsNet](https://ieeexplore.ieee.org/abstract/document/8851924)|99.64|**94.03**|-|80.03|
|[BDARS-CapsNet](https://ieeexplore.ieee.org/abstract/document/9044823)|99.67|-|-|82.30|
|[Deeper-CapsNet](https://ieeexplore.ieee.org/abstract/document/8852020/)|**99.84**|-|-|82.30|
|**AlexCapsNet(ours)**|99.66|93.27|**95.33**|**83.67**|

-总的来说，AlexCapsNet改进了很多，相比较于baseline（AlexNet和CapsNet）。

—在相对简单的数据集MNIST和FMNIST上，AlexCapsNet的性能接近最佳性能。

-在相对复杂的数据集SVHN和CIFAR10上，AlexCapsNet的性能是比较模型中最好的

## 4.2 Validation on the Depth of Feature Extraction Layers

在本节中，我们使用AlexCapsNet模型和Shallow-AlexCapsNet模型来探讨特征提取层深度的影响。**准确率**作为评价指标。

| Model | MNIST | FMNIST | SVHN | CIFAR10 | CIFAR100 | FOOD101 | FLOWER102 |
|------ | ----- | -----  | ---- |-------  | ------   | ------  | -------   |
|AlexCapsNet| **99.66** | **93.27** | **95.53** | **83.67** | 49.86 | 27.68 | 40.00 |
|S-AlexCapsNet | 99.60 | 93.18 | 94.10 | 79.98 | **51.86** | **30.59** | **50.78** |

-在类别较少的数据集上，如MNIST, FMIST, SVHN和CIFAR10，更深的特征提取层(AlexNet Module)可以有更好的性能。

-在类别较大的数据集上，如CIFAR100, FOOD101和FLOWER102，浅层特征提取层可以有更好的性能。

-这可能是由于在具有大量类别的数据集中，类别之间的相似性很高。更深的特征提取层可能会混淆语义信息，因此表现不佳。

## 4.3 Performance Analysis of Reconstruction Model

Reconstruction Module是和Capsule Network一同提出的，是用于从Dynamic Routing中提取特征恢复原图的一种方法，更多细节看这篇[文章](https://proceedings.neurips.cc/paper_files/paper/2017/hash/2cad8fa47bbef282badbb8de5374b894-Abstract.html) 


在本节中，我们要探讨Reconstruction Module的效果。我们分别使用CapsNet和AlexCapsNet对重构模块进行了评估。数据(**准确率**作为评价指标)如下表所示:

| Model | MNIST | FMNIST | SVHN | CIFAR10 | CIFAR100 | FOOD101 | FLOWER102 |
|------ | ----- | -----  | ---- |-------  | ------   | ------  | -------   |
|CapsNet| 99.21 | 90.31 | **91.30** | **70.46** | **42.02** | **18.97** | **39.01** |
|CapsNet-R| **99.23** | **90.54** | 89.07 | 67.28 | 34.38 | 10.57 | 38.64 |
|AlexCapsNet| **99.66** | 93.27 | **95.33** | **83.67** | **49.86** | **27.68** | **40.00** |
|AlexCapsNet-R| **99.66** | **93.54** | 95.13 | 75.72 | 34.33 | 4.55 | 25.98 |


-在无噪声数据集，MINST和FMINST上，Reconstruction Module可以帮助提高性能。
-在噪声数据集SVHN, CIFAR10, CIFAR100, FOOD101和FLOWER102上，Reconstruction Module降低了性能。
-当遇到带有背景噪声的图像时，会显示出较差的重建性能。
-重建后的图像显示如下:

<img src="https://github.com/BaoBao0926/AlexCapsNet/blob/main/picture/reconstruction_image.png" width="500">

# 5.Display the Data

In this section, we show all data genereated in this project together. Moreover, recall, precision, F1 score and the epoch to reach the best performance are displayed. The maximum and minimun value is marked:

<details>
  
<summary>
 Accuracy(%):
</summary>

<br />

| Model        | MNIST | FMNIST | SVHN   |  CIFAR10 | CIFAR100 | FOOD101 | FLOWER102 |
|------        | ----- | -----  | ----   | -------  | ------   | ------  | -------   |
|AlexNet       | ``99.18`` | 91.82      | 94.00      | 72.53        | ``31.39``    | 22.49       | 44.02     |
|CapsNet       | 99.21     | ``90.31``  | 91.30      | 70.46        | 42.02        | 18.97       | 39.01     |
|CapsNet-R     | 99.23     | 90.54      | ``89.07``  | ``67.28``    | 34.38        | 10.57       | 38.62     |
|AlexCapsNet   | ``99.66`` | 93.27      | ``95.33``  | ``83.67``    | 49.86        | 27.68       | 40.00     |
|AlexCapsNet-R | ``99.66`` | ``93.54``  | 95.13      | 75.72        | 34.33        | ``4.55``    | ``25.98``     |
|S-AlexCapsNet | 99.60     | 93.18      | 94.10      | 79.98        | ``51.86``    | ``30.59``   | ``46.37``     |
 
</details>

<details>
  
<summary>
 Recall(%):
</summary>

<br />

| Model        | MNIST | FMNIST | SVHN   |  CIFAR10 | CIFAR100 | FOOD101 | FLOWER102 |
|------        | ----- | -----  | ----   | -------  | ------   | ------  | -------   |
|AlexNet       | ``99.19`` | 91.88      | 93.77      |  72.74       |  ``32.70``   |  22.63      |  46.65        |
|CapsNet       | 99.21     | ``90.27``  | ``91.28``  |  ``70.98``   |  43.19       |  18.63      |  48.78        |
|CapsNet-R     | 99.22     | 90.51      | 91.46      |  79.21       |  33.79       |   9.37      |  48.58        |
|AlexCapsNet   | ``99.66`` | 93.25      | ``95.07``  |  ``83.69``   |  51.08       |  27.30      |  45.47        |
|AlexCapsNet-R | ``99.66`` | ``93.50``  | 94.81      |  77.31       |  33.95       |  ``4.04``   |  ``32.57``    |
|S-AlexCapsNet | 99.60     | 92.93      | 94.12      |  80.14       |  ``52.90``   |  ``30.38``  |  ``52.52``    |
 
</details>


<details>

<summary>
 Precision(%):
</summary>

<br />

| Model        | MNIST | FMNIST | SVHN   |  CIFAR10 | CIFAR100 | FOOD101 | FLOWER102 |
|------        | ----- | -----  | ----   | -------  | ------   | ------  | -------   |
|AlexNet       | ``99.16`` | 91.82      |  93.52     |  72.53       |  ``31.38``   |  22.49      |  44.02        |
|CapsNet       | 99.20     | ``90.31``  |  ``90.70`` |  70.45       |  42.02       |  18.97      |  39.02        |  
|CapsNet-R     | 99.22     | 90.54      |  88.13     |  ``67.27``   |  34.38       |  10.57      |  38.63        |
|AlexCapsNet   | ``99.65`` | 93.27      |  ``95.02`` |  ``83.67``   |  49.86       |  27.68      |  40.00        |
|AlexCapsNet-R | ``99.65`` | ``93.54``  |  94.72     |  75.72       |  34.33       |   ``4.55``  |  ``25.98``    |
|S-AlexCapsNet | 99.59     | 93.18      |  93.79     |  79.98       |  ``51.86``   |  ``30.58``  |  ``46.37``    |
 
</details>


<details>

<summary>
 F1 score(%):
</summary>

<br />

| Model        | MNIST | FMNIST | SVHN   |  CIFAR10 | CIFAR100 | FOOD101 | FLOWER102 |
|------        | ----- | -----  | ----   | -------  | ------   | ------  | -------   |
|AlexNet       | ``99.18`` | 91.16      | 93.56      | 72.59        |  ``31.82``   |  22.50      |  45.19        |
|CapsNet       | 99.21     | ``90.29``  | 90.54      | 70.59        |  42.56       |  18.45      |  43.05        |
|CapsNet-R     | 99.22     | 90.53      | ``88.28``  | ``67.58``    |  34.08       |   9.90      |  42.70        |
|AlexCapsNet   | ``99.66`` | 93.26      | ``94.99``  | ``83.68``    |  50.40       |  27.34      |  42.46        |
|AlexCapsNet-R | ``99.66`` | ``93.52``  | 94.70      | 76.00        |  34.14       |   ``4.28``  |  ``28.74``    |
|S-AlexCapsNet | 99.60     | 93.17      | 93.72      | 80.04        |  ``52.38``   |  ``30.40``  |  ``48.98``    |
 
</details>


<details>

<summary>
 The epoch to reach the best accuracy(th):
</summary>

<br />

| Model        | MNIST | FMNIST | SVHN   |  CIFAR10 | CIFAR100 | FOOD101 | FLOWER102 |
|------        | ----- | -----  | ----   | -------  | ------   | ------  | -------   |
|AlexNet       | ``251``   | 74         |  194       |  ``298``     |  ``253``     |  11         |  70           |
|CapsNet       | 38        | ``255``    |  ``15``    |  ``7``       |  ``7``       |  ``4 ``     |  ``8 ``       |
|CapsNet-R     | ``15``    | 11         |  79        |  54          |  56          |  48         |  199          |
|AlexCapsNet   | 83        | 199        |  279       |  230         |  15          |  7          |  34           |
|AlexCapsNet-R | 35        | 152        |  237       |  73          |  130         |  ``50``     |  ``294``      |
|S-AlexCapsNet | 58        | ``10``     |  ``287``   |  9           |  ``7``       |  10         |  31           |
 
</details>

# 6. Acknowledge

Thank Prof.Nanlin Jin and Prof.Ming Xu for their guidance. Thank PhD.Borong Lin for his help to design this project.
