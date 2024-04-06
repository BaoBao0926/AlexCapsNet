## Hey! This is [Muyi Bao](https://github.com/BaoBao0926/BaoBao0926.github.io/tree/main)! 👋👋
# Introduction to AlexCapsNet
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

# Network Architecture

<!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  -->
<details>
  
<summary>
  This is AlexNet Architecture
</summary>

<br />

 ![image](https://github.com/BaoBao0926/AlexCapsNet/blob/main/picture/AlexNet.png)
 
</details>
<!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  -->
<details>
  
<summary>
  This is the Capsule Network Architecture
</summary>

<br />

 ![image](https://github.com/BaoBao0926/AlexCapsNet/blob/main/picture/CapsNet.png)
 
</details>
  <!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  -->
<details>
  
<summary>
  This is the AlexCapsNet Architecture
</summary>

<br />

 ![image](https://github.com/BaoBao0926/AlexCapsNet/blob/main/picture/AlexCapsNet.png)
 
</details>
 <!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  -->
<details>
  
<summary>
  This is the Shallow-AlexCapsNet Architecture
</summary>

<br />

 ![image](https://github.com/BaoBao0926/AlexCapsNet/blob/main/picture/Shallow%20AlexCapsNet.png)

 And more details in Shallow-AlexNet Module
 
 ![image](https://github.com/BaoBao0926/AlexCapsNet/blob/main/picture/S-ACN-M.png)
</details>
 <!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  --><!--  -->

# Experiment Dataset

<img src="https://github.com/BaoBao0926/AlexCapsNet/blob/main/picture/datasetImage.png" width="500">



# Experiment Results




