# AlexCapsNet: An Integrated Architecture for Image Classification With Background Noise

### Authors:
- **[Muyi Bao](https://github.com/BaoBao0926/BaoBao0926.github.io)**, **Ming Xu**, **Nanlin Jin**



### NEWS:
- 2025.02.24: The paper was published in [IEEE Access](https://ieeexplore.ieee.org/document/10900363) 
- 2024.02.03: The repository is created and the code is uploeaded.


# Abstract
Capsule networks (CapsNet) are a pioneering architecture that can encode image features into vectors rather than scalars, addressing the limitations of traditional Convolutional Neural Networks (CNNs). This process is achieved by the dynamic routing algorithm and can maintain the image’s spatial hierarchies. CapsNet has demonstrated the state-of-the-art performance in simple datasets such as MNIST, but its performance degrades in more complex datasets. To solve this problem, AlexCapsNet architecture is proposed in this paper, in which the classic classification model AlexNet is used as the feature extraction layer. This allows CapsNet to capture deeper and more semantic features. The comprehensive evaluation with four datasets shows AlexCapsNet has improved performance when compared with the baseline and other CapsNet variants. Besides, our experiments on seven datasets show the reconstruction module in the CapsNet degrades the performance in the datasets with background noise. AlexCapsNet removes the reconstruction module and therefore can adapt to these complicated datasets.

# The Code Files

- All weighting file can be found in the "result" folder
- All code can be found in the folder:
  - AlexNet.py         is to train the Network
  - AlexNet_Module.py  is to create Networks
  - recon_visual.py    is to visualize the reconstructed images
  - utils.py           is to save some methods, such as training and evaluating codes, Dataloader and some simple methods.


# Network Architecture

<img src="https://github.com/BaoBao0926/AlexCapsNet/blob/main/figure/AlexCapsNet.png" alt="AlexCapsNet" width="700"/> 

<img src="https://github.com/BaoBao0926/AlexCapsNet/blob/main/figure/CapsNet-R.png" alt="CapsNet-R" width="300"/> <img src="https://github.com/BaoBao0926/AlexCapsNet/blob/main/figure/AlexCapsNet-R.png" alt="AlexCapsNet-R" width="400"/> 


# Experimental Dataset

All datasets are built-in methods in [torchvision](https://pytorch.org/vision/stable/datasets.html#built-in-datasets). The information of datasets is shown in table below and is visualized below.

<img src="https://github.com/BaoBao0926/AlexCapsNet/blob/main/figure/dataset.png" alt="dataset" width="250"/> <img src="https://github.com/BaoBao0926/AlexCapsNet/blob/main/figure/dataset_table.png" alt="table_dataset" width="500"/> 






# 4.Experiment Results

## 4.1 Performance Analysis of AlexCapsNet

We use commonly used four datasets, MNIST, F-MNIST, SVHN, CIFAR10 and use **accuracy** as evaluate metric

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

In this section, we use AlexCapsNet model and Shallow-AlexCapsNet model to explore the effect of depth of the feature extraction layers. **Accuracy** is used as evaluation metric.

| Model | MNIST | FMNIST | SVHN | CIFAR10 | CIFAR100 | FOOD101 | FLOWER102 |
|------ | ----- | -----  | ---- |-------  | ------   | ------  | -------   |
|AlexCapsNet| **99.66** | **93.27** | **95.53** | **83.67** | 49.86 | 27.68 | 40.00 |
|S-AlexCapsNet | 99.60 | 93.18 | 94.10 | 79.98 | **51.86** | **30.59** | **50.78** |

- On the datasets with fewer categories, such as MNIST, FMIST, SVHN and CIFAR10, deeper feature extraction layers (AlexNet Module) can have better performance.
- On the dataset with larger categories, such as CIFAR100, FOOD101 and FLOWER102, shallow feature extraction layers can have better performance.
- This may be due to the high similarity between categires in the dataset with large number of categires. Deeper feature extraction layers may confuse the semantic information and therefore perform poorly.

## 4.3 Performance Analysis of Reconstruction Module

The reconstruction module is proposed together with the Capsule Network, which is to recover the image from the features generated by the Dynamic routing. More details can see this [paper](https://proceedings.neurips.cc/paper_files/paper/2017/hash/2cad8fa47bbef282badbb8de5374b894-Abstract.html) 


In this section, we want to explore the effect of the reconstruction module. We use CapsNet and AlexCapsNet with and without the reconstruction module to evaluate it. The data (**Accuracy** is used as evaluation metric) is shown in the following table:

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
