## Hey! This is [Muyi Bao](https://github.com/BaoBao0926/BaoBao0926.github.io/tree/main)! 👋👋
# 1.Introduction to AlexCapsNet
This is a project to improve the performance of Capsule Network by integrating the AlexNet and CapsNet, called AlexCapsNet, used in image classification. As results, we improve the CapsNet performance across the evaluation of 5 datasets. Moreover we evaluate the performance on the dataset with noise and the dataset with large number of categoires.
  
- All weighting file can be found in the "result" folder
  
- All code can be found in the folder:
  - AlexNet.py         is to train the Network
  - AlexNet_Module.py  is to create Networks
  - recon_visual.py    is to visualize the reconstructed images
  - utils.py           is to save methods, such as training and evaluating codes, Dataloader and some simple methods.

- Main Contribution are listed below:
   - We have proposed a novel architecture, namely AlexCapsNet, which utilizes the AlexNet as feature extraction layers to capture deeper and more semantic features for CapsNet. This integration achieves the state-of-the-art performance, compared to the other variants of CapsNet.

   - We have proposed the Shallow-AlexNet (S-AlexNet) module, which has fewer layers than AlexNet. Through our experiments, we find that S-AlexNet is more suitable for the datasets with many categories and AlexNet module excels in the datasets with few categories. This observation underscores the crucial role of the depth of the feature extraction layers in the performance of CapsNet.

   - From our further investigation on different feature extraction layer modules, we find that the reconstruction module is more effective in the datasets without noise, and performs poorly in the datasets with background noise. This observation shows the potential vulnerability and limitation of the reconstruction module.

# 2.Network Architecture

<!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  -->
<details>
  
<summary>
  Architecture of AlexNet:
</summary>

<br />

 ![image](https://github.com/BaoBao0926/AlexCapsNet/blob/main/picture/AlexNet.png)
 
</details>
<!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  -->
<details>
  
<summary>
  Architecture of Capsule Network:
</summary>

<br />

 ![image](https://github.com/BaoBao0926/AlexCapsNet/blob/main/picture/CapsNet.png)
 
</details>
  <!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  -->
<details>
  
<summary>
  Architecture of AlexCapsNet:
</summary>

<br />

 ![image](https://github.com/BaoBao0926/AlexCapsNet/blob/main/picture/AlexCapsNet.png)
 
</details>
 <!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  -->
<details>
  
<summary>
  Architecture of Shallow-AlexCapsNet:
</summary>

<br />

 ![image](https://github.com/BaoBao0926/AlexCapsNet/blob/main/picture/Shallow%20AlexCapsNet.png)

 And more details in Shallow-AlexNet Module
 
 ![image](https://github.com/BaoBao0926/AlexCapsNet/blob/main/picture/S-ACN-M.png)
</details>
 <!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  -->
<details>
  
<summary>
  Architecture of AlexCapsNet or S-AlexCapsNet with the reconstruction module
</summary>

<br />

 ![image](https://github.com/BaoBao0926/AlexCapsNet/blob/main/picture/ACN-Recon.png)

</details>


# 3.Experiment Dataset

All datasets are built-in methods in [torchvision](https://pytorch.org/vision/stable/datasets.html#built-in-datasets)

| Dataset           | [MNIST](https://yann.lecun.com/exdb/mnist/)   | [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) | [SVHN](http://ufldl.stanford.edu/housenumbers/)   | [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) | [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) | [FOOD101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)   | [FLOWER102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)  |
| -------           | ------- | -------       |  ---   | ------- | -------  | -------   | -------    | 
| Categories Number | 10      | 10            | 10     | 10      |  100     | 101       |  102       |
| Image Size        | 28×28   | 28×28         | 32×32  | 32×32   |  32×32   | ＜512×512 |  ＞500×500 |
| Channel Number    | 1       | 1             |  3     | 3       |  3       | 3         |    3       |
| Training Samples  | 50,000  | 50,000        | 73,257 | 50,000  |  50,000  |  75.750   |   6149     |
| Testing Samples   | 10,000  | 10,000        | 26,032 | 10,000  |  10,000  |  25,250   |   1020     |   
| Noise             | No      | No            | Yes    | Yes     |  Yes     |  Yes      | Yes        |


<img src="https://github.com/BaoBao0926/AlexCapsNet/blob/main/picture/datasetImage.png" width="500">



# 4.Experiment Results

## 4.1 Performance Analysis of AlexCapsNet

We use commonly used four datasets, MNIST, F-MNIST, SVHN, CIFAR10 and use accuracy as evaluate metric

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

- In general, AlexCapsNet improve a lot on comparision with baseline model, AlexNet and CapsNet.

- On the relativly simple dataset, MNIST and FMNIST, the performance of AlexCapsNet is close to best performance.

- On the relatively complex dataset, SVHN and CIFAR10, the performance of AlexCapsNet is the best among compared models.

## 4.2 Validation on the Depth of Feature Extraction Layers

In this section, we use AlexCapsNet model and Shallow-AlexCapsNet model to explore the effect of depth of the feature extraction layers.

| Model | MNIST | FMNIST | SVHN | CIFAR10 | CIFAR100 | FOOD101 | FLOWER102 |
|------ | ----- | -----  | ---- |-------  | ------   | ------  | -------   |
|AlexCapsNet| **99.66** | **93.27** | **95.53** | **83.67** | 49.86 | 27.68 | 40.00 |
|S-AlexCapsNet | 99.60 | 93.18 | 94.10 | 79.98 | **51.86** | **30.59** | **50.78** |

- On the datasets with fewer categories, such as MNIST, FMIST, SVHN and CIFAR10, deeper feature extraction layers (AlexNet Module) can have better performance.
- On the dataset with larger categories, such as CIFAR100, FOOD101 and FLOWER102, shallow feature extraction layers can have better performance.
- This may be due to the high similarity between categires in the dataset with large number of categires. Deeper feature extraction layers may confuse the semantic information and therefore perform poorly.

## 4.3 Performance Analysis of Reconstruction Model

The reconstruction model is proposed together with the Capsule Network, which is to recover the image from the features generated by the Dynamic routing. More details can see this [paper](https://proceedings.neurips.cc/paper_files/paper/2017/hash/2cad8fa47bbef282badbb8de5374b894-Abstract.html)



In this section, we want to explore the effect of the reconstruction model. We use CapsNet and AlexCapsNet with and without the reconstruction module to evaluate it. The data is shown in the following table:

| Model | MNIST | FMNIST | SVHN | CIFAR10 | CIFAR100 | FOOD101 | FLOWER102 |
|------ | ----- | -----  | ---- |-------  | ------   | ------  | -------   |
|CapsNet| 99.21 | 90.31 | **91.30** | **70.46** | **42.02** | **18.97** | **39.01** |
|CapsNet-R| **99.23** | **90.54** | 89.07 | 67.28 | 34.38 | 10.57 | 38.64 |
|AlexCapsNet| **99.66** | 93.27 | **95.33** | **83.67** | **49.86** | **27.68** | **40.00** |
|AlexCapsNet-R| **99.66** | **93.54** | 95.13 | 75.72 | 34.33 | 4.55 | 25.98 |

- On the noise-free dataset, MINST and FMINST, the reconstruction module can help to improve the performance.
- On the noisy dataset, SVHN, CIFAR10, CIFAR100, FOOD101 and FLOWER102, the reconstruction module degerates the performace.
- It shows the bad performance of the reconstruction when it encounters the images with background noise.
- The reconstucted images are visualized below:

<img src="https://github.com/BaoBao0926/AlexCapsNet/blob/main/picture/reconstruction_image.png" width="500">

# 5.Display the Data

In this section, we show all data genereated in this project together. Moreover, recall, precision, F1 score and the epoch to reach the best performance are displayed:

<details>
  
<summary>
 Accuracy(%):
</summary>

<br />

| Model        | MNIST | FMNIST | SVHN   |  CIFAR10 | CIFAR100 | FOOD101 | FLOWER102 |
|------        | ----- | -----  | ----   | -------  | ------   | ------  | -------   |
|AlexNet       | 99.18 | 91.82  | 94.00  | 72.53    | 31.39    | 22.49   | 44.02     |
|CapsNet       | 99.21 | 90.31  | 91.30  | 70.46    | 42.02    | 18.97   | 39.01     |
|CapsNet-R     | 99.23 | 90.54  | 89.07  | 67.28    | 34.38    | 10.57   | 38.62     |
|AlexCapsNet   | **99.66** | 93.27  | **95.33**  | **83.67**    | 49.86    | 27.68   | 40.00     |
|AlexCapsNet-R | **99.66** | **93.54**  | 95.13  | 75.72    | 34.33    | 4.55    | 25.98     |
|S-AlexCapsNet | 99.60 | 93.18  | 94.10  | 79.98    | **51.86**    | **30.59**   | **46.37**     |
 
</details>

<details>
  
<summary>
 Recall(%):
</summary>

<br />

| Model        | MNIST | FMNIST | SVHN   |  CIFAR10 | CIFAR100 | FOOD101 | FLOWER102 |
|------        | ----- | -----  | ----   | -------  | ------   | ------  | -------   |
|AlexNet       | 99.19 | 91.88  |   |     |     |    |      |
|CapsNet       | 99.21 | 90.27  |   |     |     |    |      |
|CapsNet-R     | 99.22 | 90.51  |   |     |     |    |      |
|AlexCapsNet   | 99.66 | 93.25  |   |     |     |    |      |
|AlexCapsNet-R | 99.66 | 93.50  |   |     |     |    |      |
|S-AlexCapsNet | 99.60 | 92.93  |   |     |     |    |      |
 
</details>


<details>

<summary>
 Precision(%):
</summary>

<br />

| Model        | MNIST | FMNIST | SVHN   |  CIFAR10 | CIFAR100 | FOOD101 | FLOWER102 |
|------        | ----- | -----  | ----   | -------  | ------   | ------  | -------   |
|AlexNet       | 99.16 |   |   |     |     |    |      |
|CapsNet       | 99.20 |   |   |     |     |    |      |
|CapsNet-R     | 99.22 |   |   |     |     |    |      |
|AlexCapsNet   | 99.65 |   |   |     |     |    |      |
|AlexCapsNet-R | 99.65 |   |   |     |     |    |      |
|S-AlexCapsNet | 99.59 |   |   |     |     |    |      |
 
</details>


<details>

<summary>
 F1 score(%):
</summary>

<br />

| Model        | MNIST | FMNIST | SVHN   |  CIFAR10 | CIFAR100 | FOOD101 | FLOWER102 |
|------        | ----- | -----  | ----   | -------  | ------   | ------  | -------   |
|AlexNet       | 99.18 |   |   |     |     |    |      |
|CapsNet       | 99.21 |   |   |     |     |    |      |
|CapsNet-R     | 99.22 |   |   |     |     |    |      |
|AlexCapsNet   | 99.66 |   |   |     |     |    |      |
|AlexCapsNet-R | 99.66 |   |   |     |     |    |      |
|S-AlexCapsNet | 99.60 |   |   |     |     |    |      |
 
</details>


<details>

<summary>
 The epoch to reach the best performance(th), evaluated by accuracy:
</summary>

<br />

| Model        | MNIST | FMNIST | SVHN   |  CIFAR10 | CIFAR100 | FOOD101 | FLOWER102 |
|------        | ----- | -----  | ----   | -------  | ------   | ------  | -------   |
|AlexNet       | 251   |   |   |     |     |    |      |
|CapsNet       | 38    |   |   |     |     |    |      |
|CapsNet-R     | 15    |   |   |     |     |    |      |
|AlexCapsNet   | 83    |   |   |     |     |    |      |
|AlexCapsNet-R | 35    |   |   |     |     |    |      |
|S-AlexCapsNet | 58    |   |   |     |     |    |      |
 
</details>

# 6. Acknowledge

Thank Prof.Nanlin Jin and Prof.Ming Xu for their guidance. Thank PhD.Borong Lin for his help to design this project.
