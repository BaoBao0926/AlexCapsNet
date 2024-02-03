import utils
import AlexNet_Module
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# get model and load parameters
def get_model(model_number, dataset_number):
    model = None
    if model_number == 1:   # capsnet_recon
        if dataset_number == 1:    # mnist
            model = AlexNet_Module.CapsNet_Recon_MNIST(device).to(device)
            model.load_state_dict(torch.load(r".\Result\MNIST\Alexnet\5CapsNet_Recon\train\best_train\best_weight\best_accuracy.pth"))
        if dataset_number == 2:      # f-mnist
            model = AlexNet_Module.CapsNet_Recon_MNIST(device).to(device)
            model.load_state_dict(
            torch.load(r".\Result\FashionMNIST\Alexnet\5CapsNet_Recon\train\best_train\best_weight\best_accuracy.pth"))
        if dataset_number == 3:     # CIFAR10
            model = AlexNet_Module.CapsNet_Recon_CIFAR10(device).to(device)
            model.load_state_dict(torch.load(r".\Result\CIFAR10\Alexnet\5.CapsNet_recon\train\best_train\best_weight\best_accuracy.pth"))
        if dataset_number == 4:     # CIFAR100
            model = AlexNet_Module.CapsNet_Recon_CIFAR100(device).to(device)
            model.load_state_dict(torch.load(r".\Result\CIFAR100\Alexnet\5.CapsNet_Recon\best_train\best_weight\best_accuracy.pth"))
        if dataset_number == 5:     # FOOD101
            model = AlexNet_Module.CapsNet_Recon_FOOD101(device).to(device)
            model.load_state_dict(torch.load(r".\Result\FOOD101\Alexnet\5.CapsNet_Recon\train\best_train\best_weight\best_accuracy.pth"))
        if dataset_number == 6:     # FLOWER102
            model = AlexNet_Module.CapsNet_Recon_FLOWER102(device).to(device)
            model.load_state_dict(torch.load(r".\Result\FLOWER102\Alexnet\5.CapsNet_Recon\train\best_train\best_weight\best_accuracy.pth"))
        if dataset_number == 7:     # SVHN
            model = AlexNet_Module.CapsNet_Recon_CIFAR10(device).to(device)
            model.load_state_dict(torch.load(r'.\Result\SVHN\Alexnet\5.CapsNet_recon\best_weight\best_accuracy.pth'))

    if model_number == 2:    # AlexCpasNet_recon
        if dataset_number == 1:  # mnist
            model = AlexNet_Module.AlexCapsNet_Recon_MNIST(device).to(device)
            model.load_state_dict(
                torch.load(r".\Result\MNIST\Alexnet\6AlexCapsNet_Recon\train\best_train\best_weight\best_accuracy.pth"))
        if dataset_number == 2: # f-minst
            model = AlexNet_Module.AlexCapsNet_Recon_MNIST(device).to(device)
            model.load_state_dict(
                torch.load(r".\Result\FashionMNIST\Alexnet\6AlexCapsNet_Recon\train\best_train\best_weight\best_accuracy.pth"))
        if dataset_number == 3: # CIFAR10
            model = AlexNet_Module.AlexCapsNet_Recon_CIFAR10(device).to(device)
            model.load_state_dict(
                torch.load(
                    r".\Result\CIFAR10\Alexnet\6.AlexCapsNet_recon\train\best_train\best_weight\best_accuracy.pth"))
        if dataset_number == 4: # CIFAR100
            model = AlexNet_Module.AlexCapsNet_Recon_CIFAR100(device).to(device)
            model.load_state_dict(torch.load(
                r".\Result\CIFAR100\Alexnet\6.AlexCapsNet_recon\train\best_train\best_weight\best_accuracy.pth"))
        if dataset_number == 5: # FOOD101
            model = AlexNet_Module.AlexCapsNet_Recon_FOOD101(device).to(device)
            model.load_state_dict(torch.load(
                r".\Result\FOOD101\Alexnet\6.AlexCapsNet_recon\train\best_train\best_weight\best_accuracy.pth"))
        if dataset_number == 6: # FLOWER102
            model = AlexNet_Module.AlexCapsNet_Recon_FLOWER102(device).to(device)
            model.load_state_dict(torch.load(
                r".\Result\FLOWER102\Alexnet\6.AlexCapsNet_recon\train\best_weight\best_accuracy.pth"))
        if dataset_number == 7: #SVHN
            model = AlexNet_Module.AlexCapsNet_Recon_CIFAR10(device).to(device)
            model.load_state_dict(
                torch.load(r'.\Result\SVHN\Alexnet\6.AlexCapsNet_recon\best_train\best_weight\best_accuracy.pth'))

    return model
# get dataset
def get_dataset(dataset_number, batch_size):
    dataloaders = utils.dataloaders_recon()
    train_loader, test_loader1, test_loader2 = None, None, None
    if dataset_number == 1:  # MNIST
        train_loader, test_loader1, test_loader2 = (dataloaders.MNIST(batch_size))
    if dataset_number == 2:     # F-MNIST
        train_loader, test_loader1, test_loader2 = (dataloaders.FashionMNIST(batch_size))
    if dataset_number == 3:     # CIFAR10
        train_loader, test_loader1, test_loader2 = (dataloaders.CIFAR10(batch_size))
    if dataset_number == 4:     # CIFAR100
        train_loader, test_loader1, test_loader2 = (dataloaders.CIFAR100(batch_size))
    if dataset_number == 5:     # FOOD101
        train_loader, test_loader1, test_loader2 = (dataloaders.FOOD101(batch_size))
    if dataset_number == 6:     # FLOWER102
        train_loader, test_loader1, test_loader2 = (dataloaders.FLOWER102(batch_size))
    if dataset_number == 7:     # SVHN
        train_loader, test_loader1, test_loader2 = dataloaders.SVHN(batch_size)
    return train_loader, test_loader1, test_loader2

# show the reconstruction image
def recon_visual(dataset_number, batch_size):
    """
    visualize the reconstruction image
    :param dataset_number: the dataset number determine which dataset image will be reconstructed
    :param batch_size: batch size
    """
    # get the model
    capsnet = get_model(1, dataset_number)
    capsnet.eval()  # turn to the eval mode
    alexcapsnet = get_model(2, dataset_number)
    # get the dataset
    train_loader, test_loader1, test_loader2 = get_dataset(dataset_number, batch_size)


    i = 0
    with torch.no_grad():
        for data in test_loader1:

            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)


            cn_vector, cn_reconstruction = capsnet(inputs, target)          # capsnet
            acn_vector, acn_reconstruction = alexcapsnet(inputs, target)    # alexcapsnet

            print(i)
            i = i+1
            print(f'this is vector {acn_vector}')


            cn_reconstruction = cn_reconstruction.view_as(inputs)
            acn_reconstruction = acn_reconstruction.view_as(inputs)

            plt.subplot(3, 1, 1)
            pil_image = transforms.ToPILImage()(inputs.cpu().squeeze(0))
            plt.imshow(pil_image)
            plt.axis('off')


            plt.subplot(3, 1, 2)
            pil_image_cn = transforms.ToPILImage()(cn_reconstruction.cpu().squeeze(0))
            plt.imshow(pil_image_cn)
            plt.axis('off')

            plt.subplot(3, 1, 3)
            pil_image_acn = transforms.ToPILImage()(acn_reconstruction.cpu().squeeze(0))
            plt.imshow(pil_image_acn)
            plt.axis('off')


            plt.show()

# show the original image
def visual():
    dataloaders_recon = utils.dataloaders_recon()
    train_loader, test_loader1, test_loader2 = dataloaders_recon.SVHN(1)

    # get 9 images
    images_to_display = []
    for i, (imgs, targets) in enumerate(test_loader2):
        # transform tersor into Numpy
        img_np = np.transpose(imgs.squeeze(0).numpy(), (1, 2, 0))
        images_to_display.append(img_np)

        if i >= 15:
            break

    # create 3*3 subplot platform to show 9 images
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        ax.imshow(images_to_display[i])
        ax.axis('off')

    # Adjust the spacing of subgraphs
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    # 1: MNIST              2: F-MNIST              3:  CIFAR10
    # 4：CIFAR100            5：FOOD101              6: FLOWER102     7: SVHN
    dataset_number = 7
    batch_size = 1 # 1 is ok
    recon_visual(dataset_number, batch_size)
    # visual()