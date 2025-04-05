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


# Experimental Results

### Performance Comparison on Benchmark
<img src="https://github.com/BaoBao0926/AlexCapsNet/blob/main/figure/benchmark.png" alt="benchmark" width="700"/>

<img src="https://github.com/BaoBao0926/AlexCapsNet/blob/main/figure/table_metric.png" alt="table_metric" width="700"/>

### Curve of Loss and Accuracy 
<img src="https://github.com/BaoBao0926/AlexCapsNet/blob/main/figure/curve.png" alt="curve" width="700"/>

### Confusion Matrix and ROC curve
<img src="https://github.com/BaoBao0926/AlexCapsNet/blob/main/figure/confusion.png" alt="confusion" width="340"/> <img src="https://github.com/BaoBao0926/AlexCapsNet/blob/main/figure/ROC.png" alt="ROC" width="360"/>

###  Comparison of Model Size
<img src="https://github.com/BaoBao0926/AlexCapsNet/blob/main/figure/time.png" alt="time" width="400"/> <img src="https://github.com/BaoBao0926/AlexCapsNet/blob/main/figure/table_parameter.png" alt="table_parameter" width="400"/>

### Ablation Study of Reconstruction Module
<img src="https://github.com/BaoBao0926/AlexCapsNet/blob/main/figure/table_recon.png" alt="recon" width="700"/>

<img src="https://github.com/BaoBao0926/AlexCapsNet/blob/main/figure/recon_images.png" alt="reconimages" width="500"/>


# More Data

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

# Acknowledge

I personally thank Prof. Nanlin Jin and Prof. Ming Xu for their guidance, and thank PhD Borong Lin for his help to design this project.
