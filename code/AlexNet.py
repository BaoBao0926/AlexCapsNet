import argparse
import time
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import utils
import AlexNet_Module
import matplotlib.pyplot as plt
import numpy as np

dataloaders = utils.dataloaders()
dataloaders_recon = utils.dataloaders_recon()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def _arparse(epoch, batch_size, lr, lr_decay, r, num_save_epoch, train_save_dir, train_eval,
             pretrained, pretrained_weight, num_class, save_all, reconstruction, recon_alpha):
    parser = argparse.ArgumentParser(description="")
    # trainning parameters
    parser.add_argument('--epochs', default=epoch, type=int)
    parser.add_argument('--batch_size', default=batch_size, type=int)   # batch_size    50(batch size)*1000(batch number)
    parser.add_argument('--lr', default=lr, type=float, help="Initial learning rate")
    parser.add_argument('--lr_decay', default=lr_decay, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('-r', '--routings', default=r, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('-num_save_epoch',  default=num_save_epoch, type=int, help="How many epochs to save the model ")
    parser.add_argument('--num_class', default=num_class, type=int)
    parser.add_argument('--reconstruction_alpha', default=recon_alpha, type=float)

    # directory
    parser.add_argument('--train_save_dir', default=train_save_dir, type=str,
                        help='the path directory of training')

    # choose mode
    parser.add_argument('--train_eval', default=train_eval, type=bool, help='Whether to do a test during training')
    parser.add_argument('--save_all', default=save_all, type=bool, help='whether to save all weights')
    parser.add_argument('--pretrained', default=pretrained, type=bool, help='Whether to do a pretrained learning')
    parser.add_argument('--pretrained_weight', default=pretrained_weight, type=str, help='Whether to do a pretrained learning')
    parser.add_argument('--reconstruction', default=reconstruction, type=bool, help='whether to use reconstruction')
    args = parser.parse_args()
    return args

def main(DATASET, NETWORK):

    # CIFAR10
    if DATASET == 1:

        # AlexCapsNet when capsnet as predict layer.
        if NETWORK == 1:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR10/Alexnet/1.AlexCapsNet/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,    # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=1,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/CIFAR10/Alexnet/AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.CIFAR10(args.batch_size)), 10

            if not args.pretrained:  # 不使用预训练结果
                capAlex = AlexNet_Module.AlexCapsNet_CIFAR10(device).to(device)
                print(f'The CapsAlexNet for CIFAR10 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)
            else:  # 使用预训练
                capAlex = AlexNet_Module.AlexCapsNet_CIFAR10(device).to(device)
                capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for CIFAR10 architecture is shown:\n {capAlex}')  # show architecture)
                utils.train_capOutput(capAlex, train_loader, test_loader, args)

        # Alexnet
        if NETWORK == 2:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, r=3, lr_decay=0.995, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR10/Alexnet/2.AlexNet/train',
                            train_eval=True,  # false is fine
                            save_all=True,
                            batch_size=50,  # 测试的时候，batch_size为1, 训练为50
                            num_save_epoch=1,  # 保存一次要20MB，训练300epoch的时候，酌情设置， 50
                            pretrained=False,
                            pretrained_weight='./Result/CIFAR10/Alexnet/AlexNet/train/train1/model_2.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.CIFAR10(args.batch_size)), 10

            if not args.pretrained:
                # get CapsAlexNet
                Alex = AlexNet_Module.AlexNet_CIFAR10().to(device)
                print(f'The AlexNet for CIFAR10 architecture is shown:\n {Alex}')  # show architecture
                utils.train(Alex, train_loader, test_loader, args)
            else:   # pretrained
                Alex = AlexNet_Module.AlexNet_CIFAR10().to(device)
                Alex.load_state_dict(torch.load(args.pretrained_weights))
                print(f'The AlexNet for CIFAR10 architecture is shown:\n {Alex}')  # show architecture
                utils.train(Alex, train_loader, test_loader, args)

        # CapsNet
        if NETWORK == 3:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR10/Alexnet/3.CapsNet/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,    # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/CIFAR10/Alexnet/AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.CIFAR10(args.batch_size)), 10

            if not args.pretrained:  # 不使用预训练结果
                capnet = AlexNet_Module.CapsNet_CIFAR10(device).to(device)
                print(f'The CapsAlexNet for CIFAR10 architecture is shown:\n {capnet}')  # show architecture
                utils.train_capOutput(capnet, train_loader, test_loader, args)
            else:  # 使用预训练
                capnet = AlexNet_Module.CapsNet_CIFAR10(device).to(device)
                capnet.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for CIFAR10 architecture is shown:\n {capnet}')  # show architecture)
                utils.train_capOutput(capnet, train_loader, test_loader, args)

        # CapsAlexNet, capsnet layer is inside Alexnet·
        if NETWORK == 4:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, r=3, lr_decay=0.995, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR10/Alexnet/4.CapsAlexNet/train',
                            train_eval=True,  # false is fine
                            save_all=True,
                            batch_size=50,  # 测试的时候，batch_size为1, 训练为50
                            num_save_epoch=50,  # 保存一次要20MB，训练300epoch的时候，酌情设置， 50
                            pretrained=False,
                            pretrained_weight='./Result/CIFAR10/Alexnet/CapsAlexNet/train/train1/model_2.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.CIFAR10(args.batch_size)), 10

            if not args.pretrained:
                # get CapsAlexNet
                capsalex = AlexNet_Module.CapsAlexNet_CIFAR10(device).to(device)
                print(f'The CapsAlexNet for CIFAR10 architecture is shown:\n {capsalex}')  # show architecture
                utils.train(capsalex, train_loader, test_loader, args)
            else:  # pretrained
                capsalex = AlexNet_Module.CapsAlexNet_CIFAR10(device).to(device)
                capsalex.load_state_dict(torch.load(args.pretrained_weights))
                print(f'The CapsAlexNet for CIFAR10 architecture is shown:\n {capsalex}')  # show architecture
                utils.train(capsalex, train_loader, test_loader, args)

        # CapsNet with reconstruction
        if NETWORK == 5:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR10/Alexnet/5.CapsNet_recon/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=1,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/CIFAR10/Alexnet/CapsNet_recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )
            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.CIFAR10(args.batch_size)), 10

            if not args.pretrained:  # 不使用预训练结果
                capnet_recon = AlexNet_Module.CapsNet_Recon_CIFAR10(device).to(device)
                print(f'The CapsAlexNet for CIFAR10 architecture is shown:\n {capnet_recon}')  # show architecture
                utils.train_capOutput(capnet_recon, train_loader, test_loader, args)
            else:  # 使用预训练
                capnet_recon = AlexNet_Module.CapsNet_Recon_CIFAR10(device).to(device)
                capnet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for CIFAR10 architecture is shown:\n {capnet_recon}')  # show architecture)
                utils.train_capOutput(capnet_recon, train_loader, test_loader, args)

        # AlexCapsNet with reconstruction
        if NETWORK == 6:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR10/Alexnet/6.AlexCapsNet_recon/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=1,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/CIFAR10/Alexnet/AlexCapsNet_recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )
            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.CIFAR10(args.batch_size)), 10

            if not args.pretrained:  # 不使用预训练结果
                alexcapnet_recon = AlexNet_Module.AlexCapsNet_Recon_CIFAR10(device).to(device)
                print(f'The CapsAlexNet for CIFAR10 architecture is shown:\n {alexcapnet_recon}')  # show architecture
                utils.train_capOutput(alexcapnet_recon, train_loader, test_loader, args)
            else:  # 使用预训练
                alexcapnet_recon = AlexNet_Module.AlexCapsNet_Recon_CIFAR10(device).to(device)
                alexcapnet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for CIFAR10 architecture is shown:\n {alexcapnet_recon}')  # show architecture)
                utils.train_capOutput(alexcapnet_recon, train_loader, test_loader, args)

        # AlexCpasNet fully contected output
        if NETWORK == 7:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR10/Alexnet/7.AlexCapsNet_FullyOut/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=1,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/CIFAR10/Alexnet/AlexCapsNet_FullyOut/train/best_train/model_299.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.CIFAR10(args.batch_size)), 10

            if not args.pretrained:  # 不使用预训练结果
                alexcapnet_fullyOut = AlexNet_Module.AlexCapsNet_Fully_CIFAR10(device).to(device)
                print(f'The CapsAlexNet_FullyOut for CIFAR10 architecture is shown:\n {alexcapnet_fullyOut}')  # show architecture
                utils.train(alexcapnet_fullyOut, train_loader, test_loader, args)
            else:  # 使用预训练
                alexcapnet_fullyOut = AlexNet_Module.AlexCapsNet_Fully_CIFAR10(device).to(device)
                alexcapnet_fullyOut.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet_FullyOut for CIFAR10 architecture is shown:\n {alexcapnet_fullyOut}')
                utils.train(alexcapnet_fullyOut, train_loader, test_loader, args)

        # AlexCapsNet_FullyOut_Recon
        if NETWORK == 8:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR10/Alexnet/8.AlexCapsNet_FullyOut_Recon/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=1,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/CIFAR10/Alexnet/AlexCapsNet_FullyOut_Recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )
            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.CIFAR10(args.batch_size)), 10

            if not args.pretrained:  # 不使用预训练结果
                alexcapnet_F_recon = AlexNet_Module.AlexCapsNet_F_Recon_CIFAR10(device).to(device)
                print(f'The CapsAlexNet_Fully_Rencon for CIFAR10 architecture is shown:\n {alexcapnet_F_recon}')  # show architecture
                utils.train(alexcapnet_F_recon, train_loader, test_loader, args)
            else:  # 使用预训练
                alexcapnet_F_recon = AlexNet_Module.AlexCapsNet_Fully_CIFAR10(device).to(device)
                alexcapnet_F_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet_FullyOut_Recon for CIFAR10 architecture is shown:\n {alexcapnet_F_recon}')
                utils.train(alexcapnet_F_recon, train_loader, test_loader, args)

        # Shallow AlexCapsNet_F
        if NETWORK == 9:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR10/Alexnet/9.S_AlexCapsNet_F/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=1,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/CIFAR10/Alexnet/9.S_AlexCapsNet_F/train/best_train/model_299.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.CIFAR10(args.batch_size)), 10

            if not args.pretrained:  # 不使用预训练结果
                alexcapnet_fullyOut = AlexNet_Module.S_AlexCapsNet_Fully_CIFAR10(device).to(device)
                print(f'The S_CapsAlexNet_F for CIFAR10 architecture is shown:\n {alexcapnet_fullyOut}')  # show architecture
                utils.train(alexcapnet_fullyOut, train_loader, test_loader, args)
            else:  # 使用预训练
                alexcapnet_fullyOut = AlexNet_Module.S_AlexCapsNet_Fully_CIFAR10(device).to(device)
                alexcapnet_fullyOut.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet_F for CIFAR10 architecture is shown:\n {alexcapnet_fullyOut}')
                utils.train(alexcapnet_fullyOut, train_loader, test_loader, args)

        # Shallow AlexCapsNet_F
        if NETWORK == 10:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR10/Alexnet/10.S_AlexCapsNet/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,    # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=1,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/CIFAR10/Alexnet/10.S_AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.CIFAR10(args.batch_size)), 10

            if not args.pretrained:  # 不使用预训练结果
                capAlex = AlexNet_Module.S_AlexCapsNet_CIFAR10(device).to(device)
                print(f'The CapsAlexNet for CIFAR10 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)
            else:  # 使用预训练
                capAlex = AlexNet_Module.S_AlexCapsNet_CIFAR10(device).to(device)
                capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for CIFAR10 architecture is shown:\n {capAlex}')  # show architecture)
                utils.train_capOutput(capAlex, train_loader, test_loader, args)

    # CIFAR100
    if DATASET == 2:

        # AlexCapsNet
        if NETWORK == 1:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=100, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR100/Alexnet/1.AlexCapsNet/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/CIFAR100/Alexnet/AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.CIFAR100(args.batch_size)), 100

            if not args.pretrained:  # 不使用预训练结果
                capAlex = AlexNet_Module.AlexCapsNet_CIFAR100(device).to(device)
                print(f'The CapsAlexNet for CIFAR10 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)
            else:                    # 使用预训练
                capAlex = AlexNet_Module.AlexCapsNet_CIFAR100(device).to(device)
                capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for CIFAR10 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)

        # Alexnet
        if NETWORK == 2:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, r=3, lr_decay=0.995, num_class=100, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR100/Alexnet/2.AlexNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,  # 测试的时候，batch_size为1, 训练为50
                            num_save_epoch=50,  # 保存一次要20MB，训练300epoch的时候，酌情设置， 50
                            pretrained=False,
                            pretrained_weight='./Result/CIFAR100/Alexnet/AlexNet/train/train1/model_2.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.CIFAR100(args.batch_size)), 100

            if not args.pretrained:
                # get CapsAlexNet
                Alex = AlexNet_Module.AlexNet_CIFAR100().to(device)
                print(f'The AlexNet for CIFAR10 architecture is shown:\n {Alex}')  # show architecture
                utils.train(Alex, train_loader, test_loader, args)
            else:   # pretrained
                Alex = AlexNet_Module.AlexNet_CIFAR100().to(device)
                Alex.load_state_dict(torch.load(args.pretrained_weights))
                print(f'The AlexNet for CIFAR10 architecture is shown:\n {Alex}')  # show architecture
                utils.train(Alex, train_loader, test_loader, args)

        # CapsNet
        if NETWORK == 3:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=100, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR100/Alexnet/3.CapsNet/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,    # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/CIFAR100/Alexnet/AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.CIFAR100(args.batch_size)), 100

            if not args.pretrained:  # 不使用预训练结果
                capnet = AlexNet_Module.CapsNet_CIFAR100(device).to(device)
                print(f'The CapsAlexNet for CIFAR10 architecture is shown:\n {capnet}')  # show architecture
                utils.train_capOutput(capnet, train_loader, test_loader, args)
            else:  # 使用预训练
                capnet = AlexNet_Module.CapsNet_CIFAR100(device).to(device)
                capnet.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for CIFAR10 architecture is shown:\n {capnet}')  # show architecture)
                utils.train_capOutput(capnet, train_loader, test_loader, args)

        # CapsAlexNet, capsnet layer is inside Alexnet·
        if NETWORK == 4:

                # get parameter
                args = _arparse(epoch=300, lr=0.001, r=3, lr_decay=0.995, num_class=100, recon_alpha=0.0005,
                                train_save_dir='./Result/CIFAR100/Alexnet/4.CapsAlexNet/train',
                                train_eval=True,  # false is fine
                                save_all=True,
                                batch_size=50,  # 测试的时候，batch_size为1, 训练为50
                                num_save_epoch=50,  # 保存一次要20MB，训练300epoch的时候，酌情设置， 50
                                pretrained=False,
                                pretrained_weight='./Result/CIFAR100/Alexnet/CapsAlexNet/train/train1/model_2.pth',
                                reconstruction=False
                                )
                # get corresponding dataset
                # CIFAR100   train images: 50000  50(batch size)*1000(batch number)
                (train_loader, test_loader), num_class = (dataloaders.CIFAR100(args.batch_size)), 100

                if not args.pretrained:
                    # get CapsAlexNet
                    capsalex = AlexNet_Module.CapsAlexNet_CIFAR100(device).to(device)
                    print(f'The AlexNet for CIFAR100 architecture is shown:\n {capsalex}')  # show architecture
                    utils.train(capsalex, train_loader, test_loader, args)
                else:  # pretrained
                    capsalex = AlexNet_Module.CapsAlexNet_CIFAR100(device).to(device)
                    capsalex.load_state_dict(torch.load(args.pretrained_weights))
                    print(f'The AlexNet for CIFAR100 architecture is shown:\n {capsalex}')  # show architecture
                    utils.train(capsalex, train_loader, test_loader, args)

        # CapsNet with reconstruction
        if NETWORK == 5:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=100, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR100/Alexnet/5.CapsNet_recon/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=1,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/CIFAR100/Alexnet/CapsNet_recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )
            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.CIFAR100(args.batch_size)), 100

            if not args.pretrained:  # 不使用预训练结果
                capnet_recon = AlexNet_Module.CapsNet_Recon_CIFAR100(device).to(device)
                print(f'The CapsAlexNet for CIFAR100 architecture is shown:\n {capnet_recon}')  # show architecture
                utils.train_capOutput(capnet_recon, train_loader, test_loader, args)
            else:  # 使用预训练
                capnet_recon = AlexNet_Module.CapsNet_Recon_CIFAR100(device).to(device)
                capnet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for CIFAR100 architecture is shown:\n {capnet_recon}')  # show architecture)
                utils.train_capOutput(capnet_recon, train_loader, test_loader, args)

        # AlexCapsNet with reconstruction
        if NETWORK == 6:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=100, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR100/Alexnet/6.AlexCapsNet_recon/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=1,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/CIFAR100/Alexnet/AlexCapsNet_recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )
            # get corresponding dataset
            # CIFAR100   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.CIFAR100(args.batch_size)), 100

            if not args.pretrained:  # 不使用预训练结果
                alexcapnet_recon = AlexNet_Module.AlexCapsNet_Recon_CIFAR100(device).to(device)
                print(f'The CapsAlexNet for CIFAR100 architecture is shown:\n {alexcapnet_recon}')  # show architecture
                utils.train_capOutput(alexcapnet_recon, train_loader, test_loader, args)
            else:  # 使用预训练
                alexcapnet_recon = AlexNet_Module.AlexCapsNet_Recon_CIFAR100(device).to(device)
                alexcapnet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for CIFAR100 architecture is shown:\n {alexcapnet_recon}')  # show architecture)
                utils.train_capOutput(alexcapnet_recon, train_loader, test_loader, args)

        # AlexCpasNet fully contected output
        if NETWORK == 7:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=100, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR100/Alexnet/7.AlexCapsNet_F/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/CIFAR100/Alexnet/AlexCapsNet_FullyOut/train/best_train/model_299.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.CIFAR100(args.batch_size)), 100

            if not args.pretrained:  # 不使用预训练结果
                alexcapnet_fullyOut = AlexNet_Module.AlexCapsNet_Fully_CIFAR100(device).to(device)
                print(f'The CapsAlexNet_FullyOut for CIFAR10 architecture is shown:\n {alexcapnet_fullyOut}')  # show architecture
                utils.train(alexcapnet_fullyOut, train_loader, test_loader, args)
            else:  # 使用预训练
                alexcapnet_fullyOut = AlexNet_Module.AlexCapsNet_Fully_CIFAR100(device).to(device)
                alexcapnet_fullyOut.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet_FullyOut for CIFAR10 architecture is shown:\n {alexcapnet_fullyOut}')
                utils.train(alexcapnet_fullyOut, train_loader, test_loader, args)

        # AlexCapsNet_FullyOut_Recon
        if NETWORK == 8:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR100/Alexnet/8.AlexCapsNet_F_Recon/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=1,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/CIFAR100/Alexnet/AlexCapsNet_FullyOut_Recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )
            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.CIFAR100(args.batch_size)), 100

            if not args.pretrained:  # 不使用预训练结果
                alexcapnet_F_recon = AlexNet_Module.AlexCapsNet_F_Recon_CIFAR100(device).to(device)
                print(f'The CapsAlexNet_Fully_Rencon for CIFAR100 architecture is shown:\n {alexcapnet_F_recon}')  # show architecture
                utils.train(alexcapnet_F_recon, train_loader, test_loader, args)
            else:  # 使用预训练
                alexcapnet_F_recon = AlexNet_Module.AlexCapsNet_Fully_CIFAR100(device).to(device)
                alexcapnet_F_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet_FullyOut_Recon for CIFAR100 architecture is shown:\n {alexcapnet_F_recon}')
                utils.train(alexcapnet_F_recon, train_loader, test_loader, args)

        # Shallow AlexCpasNet fully contected output
        if NETWORK == 9:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=100, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR100/Alexnet/9.S_AlexCapsNet_F/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/CIFAR100/Alexnet/9.S_AlexCapsNet_F/train/best_train/model_299.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.CIFAR100(args.batch_size)), 100

            if not args.pretrained:  # 不使用预训练结果
                alexcapnet_fullyOut = AlexNet_Module.S_AlexCapsNet_Fully_CIFAR100(device).to(device)
                print(f'The S_AlexCapsNet_Fully for CIFAR10 architecture is shown:\n {alexcapnet_fullyOut}')  # show architecture
                utils.train(alexcapnet_fullyOut, train_loader, test_loader, args)
            else:  # 使用预训练
                alexcapnet_fullyOut = AlexNet_Module.AlexCapsNet_Fully_CIFAR100(device).to(device)
                alexcapnet_fullyOut.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The S_AlexCapsNet_Fully for CIFAR10 architecture is shown:\n {alexcapnet_fullyOut}')
                utils.train(alexcapnet_fullyOut, train_loader, test_loader, args)

        # Shaollow AlexCapsNet
        if NETWORK == 10:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=100, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR100/Alexnet/10.S_AlexCapsNet/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/CIFAR100/Alexnet/10.S_AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.CIFAR100(args.batch_size)), 100

            if not args.pretrained:  # 不使用预训练结果
                capAlex = AlexNet_Module.S_AlexCapsNet_CIFAR100(device).to(device)
                print(f'The S_CapsAlexNet for CIFAR10 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)
            else:                    # 使用预训练
                capAlex = AlexNet_Module.S_AlexCapsNet_CIFAR100(device).to(device)
                capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The S_CapsAlexNet for CIFAR10 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)

    # MINIST 28*28
    if DATASET == 3:

        # AlexCapsNet when capsnet as predict layer.
        if NETWORK == 1:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/MNIST/Alexnet/1AlexCapsNet/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/MNIST/Alexnet/1AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            # MNIST 28*28
            (train_loader, test_loader), num_class = (dataloaders.MNIST(args.batch_size)), 10


            if not args.pretrained:  # 不使用预训练结果
                capAlex = AlexNet_Module.AlexCapsNet_MNIST(device).to(device)
                print(f'The AlexCapsNet for MNIST architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)
            else:  # 使用预训练
                capAlex = AlexNet_Module.AlexCapsNet_MNIST(device).to(device)
                capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The AlexCapsNet for MNIST architecture is shown:\n {capAlex}')  # show architecture)
                utils.train_capOutput(capAlex, train_loader, test_loader, args)

        # AlexNet
        if NETWORK == 2:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/MNIST/Alexnet/2AlexNet/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/MNIST/Alexnet/2AlexNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            # MNIST 28*28
            (train_loader, test_loader), num_class = (dataloaders.MNIST(args.batch_size)), 10


            if not args.pretrained:
                AlexNet = AlexNet_Module.AlexNet_MNIST(device).to(device)
                print(f'The AlexNet for MNIST architecture is shown:\n {AlexNet}')  # show architecture
                utils.train(AlexNet, train_loader, test_loader, args)
            else:  # 使用预训练
                AlexNet = AlexNet_Module.AlexNet_MNIST(device).to(device)
                AlexNet.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The AlexNet for MNIST architecture is shown:\n {AlexNet}')  # show architecture)
                utils.train(AlexNet, train_loader, test_loader, args)

        # CapsNet
        if NETWORK == 3:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/MNIST/Alexnet/3CapsNet/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/MNIST/Alexnet/3CapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            # MNIST 28*28
            (train_loader, test_loader), num_class = (dataloaders.MNIST(args.batch_size)), 10

            if not args.pretrained:  # 不使用预训练结果
                capnet = AlexNet_Module.CapsNet_MNIST(device).to(device)
                print(f'The capnet for MNIST architecture is shown:\n {capnet}')  # show architecture
                utils.train_capOutput(capnet, train_loader, test_loader, args)
            else:  # 使用预训练
                capnet = AlexNet_Module.CapsNet_MNIST(device).to(device)
                capnet.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsNet for MNIST architecture is shown:\n {capnet}')  # show architecture)
                utils.train_capOutput(capnet, train_loader, test_loader, args)

        # CapsAlexNet
        if NETWORK == 4:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/MNIST/Alexnet/4CapsAlexNet/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/MNIST/Alexnet/4CapsAlexNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            # MNIST 28*28
            (train_loader, test_loader), num_class = (dataloaders.MNIST(args.batch_size)), 10

            if not args.pretrained:
                CapsAlexNet = AlexNet_Module.CapsAlexNet_MNIST(device).to(device)
                print(f'The CapsAlexNet for MNIST architecture is shown:\n {CapsAlexNet}')  # show architecture
                utils.train(CapsAlexNet, train_loader, test_loader, args)
            else:  # 使用预训练
                CapsAlexNet = AlexNet_Module.CapsAlexNet_MNIST(device).to(device)
                CapsAlexNet.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for MNIST architecture is shown:\n {CapsAlexNet}')  # show architecture)
                utils.train(CapsAlexNet, train_loader, test_loader, args)

        # CapsNet-Recon
        if NETWORK == 5:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/MNIST/Alexnet/5CapsNet_Recon/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/MNIST/Alexnet/5CapsNet_Recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )

            # get corresponding dataset
            # MNIST 28*28
            (train_loader, test_loader), num_class = (dataloaders.MNIST(args.batch_size)), 10

            if not args.pretrained:
                CapsNet_recon = AlexNet_Module.CapsNet_Recon_MNIST(device).to(device)
                print(f'The CapsAlexNet for MNIST architecture is shown:\n {CapsNet_recon}')  # show architecture
                utils.train_capOutput(CapsNet_recon, train_loader, test_loader, args)
            else:  # 使用预训练
                CapsNet_recon = AlexNet_Module.CapsNet_Recon_MNIST(device).to(device)
                CapsNet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for MNIST architecture is shown:\n {CapsNet_recon}')  # show architecture)
                utils.train_capOutput(CapsNet_recon, train_loader, test_loader, args)

        # AlexCapsNet-Recon
        if NETWORK == 6:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/MNIST/Alexnet/6AlexCapsNet_Recon/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/MNIST/Alexnet/6AlexCapsNet_Recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )

            # get corresponding dataset
            # MNIST 28*28
            (train_loader, test_loader), num_class = (dataloaders.MNIST(args.batch_size)), 10

            if not args.pretrained:
                AlexCapsNet_recon = AlexNet_Module.AlexCapsNet_Recon_MNIST(device).to(device)
                print(f'The CapsAlexNet for MNIST architecture is shown:\n {AlexCapsNet_recon}')  # show architecture
                utils.train_capOutput(AlexCapsNet_recon, train_loader, test_loader, args)
            else:  # 使用预训练
                AlexCapsNet_recon = AlexNet_Module.AlexCapsNet_Recon_MNIST(device).to(device)
                AlexCapsNet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for MNIST architecture is shown:\n {AlexCapsNet_recon}')  # show architecture)
                utils.train_capOutput(AlexCapsNet_recon, train_loader, test_loader, args)

        # AlexCapsNet_FullyOut
        if NETWORK == 7:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/MNIST/Alexnet/7AlexCapsNet_FullyOut/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/MNIST/Alexnet/7AlexCapsNet_FullyOut/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            # MNIST 28*28
            (train_loader, test_loader), num_class = (dataloaders.MNIST(args.batch_size)), 10

            if not args.pretrained:
                AlexCapsNet_fullyOut = AlexNet_Module.AlexCapsNet_Fully_MNIST(device).to(device)
                print(f'The AlexCapsNet_fullyOut for MNIST architecture is shown:\n {AlexCapsNet_fullyOut}')  # show architecture
                utils.train(AlexCapsNet_fullyOut, train_loader, test_loader, args)
            else:  # 使用预训练
                AlexCapsNet_fullyOut = AlexNet_Module.AlexCapsNet_Fully_MNIST(device).to(device)
                AlexCapsNet_fullyOut.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The AlexCapsNet_fullyOut for MNIST architecture is shown:\n {AlexCapsNet_fullyOut}')  # show architecture)
                utils.train(AlexCapsNet_fullyOut, train_loader, test_loader, args)

        # AlexCapsNet_FullyOut_Recon
        if NETWORK == 8:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/MNIST/Alexnet/8AlexCapsNet_FullyOut_Recon/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/MNIST/Alexnet/8AlexCapsNet_FullyOut_Recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )

            # get corresponding dataset
            # MNIST 28*28
            (train_loader, test_loader), num_class = (dataloaders.MNIST(args.batch_size)), 10

            if not args.pretrained:
                AlexCapsNet_fullyOut_recon = AlexNet_Module.AlexCapsNet_Fully_Recon_MNIST(device).to(device)
                print(
                    f'The AlexCapsNet_fullyOut for MNIST architecture is shown:\n {AlexCapsNet_fullyOut_recon}')  # show architecture
                utils.train(AlexCapsNet_fullyOut_recon, train_loader, test_loader, args)
            else:  # 使用预训练
                AlexCapsNet_fullyOut_recon = AlexNet_Module.AlexCapsNet_Fully_Recon_MNIST(device).to(device)
                AlexCapsNet_fullyOut_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(
                    f'The AlexCapsNet_fullyOut for MNIST architecture is shown:\n {AlexCapsNet_fullyOut_recon}')  # show architecture)
                utils.train(AlexCapsNet_fullyOut_recon, train_loader, test_loader, args)

        # Shallow AlexCapsNet Full out
        if NETWORK == 9:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/MNIST/Alexnet/9S_AlexCapsNet_F/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/MNIST/Alexnet/9S_AlexCapsNet_F/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            # MNIST 28*28
            (train_loader, test_loader), num_class = (dataloaders.MNIST(args.batch_size)), 10

            if not args.pretrained:
                AlexCapsNet_fullyOut = AlexNet_Module.S_AlexCapsNet_Fully_MNIST(device).to(device)
                print(f'The S_AlexCapsNet_Fully for MNIST architecture is shown:\n {AlexCapsNet_fullyOut}')  # show architecture
                utils.train(AlexCapsNet_fullyOut, train_loader, test_loader, args)
            else:  # 使用预训练
                AlexCapsNet_fullyOut = AlexNet_Module.S_AlexCapsNet_Fully_MNIST(device).to(device)
                AlexCapsNet_fullyOut.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The S_AlexCapsNet_Fully for MNIST architecture is shown:\n {AlexCapsNet_fullyOut}')  # show architecture)
                utils.train(AlexCapsNet_fullyOut, train_loader, test_loader, args)

        # Shallow AlexCapsNet
        if NETWORK == 10:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/MNIST/Alexnet/10S_AlexCapsNet/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/MNIST/Alexnet/10S_AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            # MNIST 28*28
            (train_loader, test_loader), num_class = (dataloaders.MNIST(args.batch_size)), 10


            if not args.pretrained:  # 不使用预训练结果
                s_capAlex = AlexNet_Module.S_AlexCapsNet_MNIST(device).to(device)
                print(f'The S_AlexCapsNet for MNIST architecture is shown:\n {s_capAlex}')  # show architecture
                utils.train_capOutput(s_capAlex, train_loader, test_loader, args)
            else:  # 使用预训练
                s_capAlex = AlexNet_Module.S_AlexCapsNet_MNIST(device).to(device)
                s_capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The S_AlexCapsNet for MNIST architecture is shown:\n {s_capAlex}')  # show architecture)
                utils.train_capOutput(s_capAlex, train_loader, test_loader, args)

    # FashionMINIST 28*28
    if DATASET == 4:

        # AlexCapsNet when capsnet as predict layer.
        if NETWORK == 1:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/FashionMNIST/Alexnet/1AlexCapsNet/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FashionMNIST/Alexnet/1AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            # MNIST 28*28
            (train_loader, test_loader), num_class = (dataloaders.FashionMNIST(args.batch_size)), 10


            if not args.pretrained:  # 不使用预训练结果
                capAlex = AlexNet_Module.AlexCapsNet_MNIST(device).to(device)
                print(f'The AlexCapsNet for FashionMNIST architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)
            else:  # 使用预训练
                capAlex = AlexNet_Module.AlexCapsNet_MNIST(device).to(device)
                capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The AlexCapsNet for FashionMNIST architecture is shown:\n {capAlex}')  # show architecture)
                utils.train_capOutput(capAlex, train_loader, test_loader, args)

        # AlexNet
        if NETWORK == 2:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/FashionMNIST/Alexnet/2AlexNet/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FashionMNIST/Alexnet/2AlexNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            # MNIST 28*28
            (train_loader, test_loader), num_class = (dataloaders.FashionMNIST(args.batch_size)), 10


            if not args.pretrained:
                AlexNet = AlexNet_Module.AlexNet_MNIST(device).to(device)
                print(f'The AlexNet for FashionMNIST architecture is shown:\n {AlexNet}')  # show architecture
                utils.train(AlexNet, train_loader, test_loader, args)
            else:  # 使用预训练
                AlexNet = AlexNet_Module.AlexNet_MNIST(device).to(device)
                AlexNet.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The AlexNet for FashionMNIST architecture is shown:\n {AlexNet}')  # show architecture)
                utils.train(AlexNet, train_loader, test_loader, args)

        # CapsNet
        if NETWORK == 3:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/FashionMNIST/Alexnet/3CapsNet/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FashionMNIST/Alexnet/3CapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            # MNIST 28*28
            (train_loader, test_loader), num_class = (dataloaders.FashionMNIST(args.batch_size)), 10

            if not args.pretrained:  # 不使用预训练结果
                capnet = AlexNet_Module.CapsNet_MNIST(device).to(device)
                print(f'The capnet for FashionMNIST architecture is shown:\n {capnet}')  # show architecture
                utils.train_capOutput(capnet, train_loader, test_loader, args)
            else:  # 使用预训练
                capnet = AlexNet_Module.CapsNet_MNIST(device).to(device)
                capnet.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsNet for FashionMNIST architecture is shown:\n {capnet}')  # show architecture)
                utils.train_capOutput(capnet, train_loader, test_loader, args)

        # CapsAlexNet
        if NETWORK == 4:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/FashionMNIST/Alexnet/4CapsAlexNet/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FashionMNIST/Alexnet/4CapsAlexNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            # MNIST 28*28
            (train_loader, test_loader), num_class = (dataloaders.FashionMNIST(args.batch_size)), 10


            if not args.pretrained:
                CapsAlexNet = AlexNet_Module.CapsAlexNet_MNIST(device).to(device)
                print(f'The CapsAlexNet for FashionMNIST architecture is shown:\n {CapsAlexNet}')  # show architecture
                utils.train(CapsAlexNet, train_loader, test_loader, args)
            else:  # 使用预训练
                CapsAlexNet = AlexNet_Module.CapsAlexNet_MNIST(device).to(device)
                CapsAlexNet.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for FashionMNIST architecture is shown:\n {CapsAlexNet}')  # show architecture)
                utils.train(CapsAlexNet, train_loader, test_loader, args)

        # CapsNet-Recon
        if NETWORK == 5:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/FashionMNIST/Alexnet/5CapsNet_Recon/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FashionMNIST/Alexnet/5CapsNet_Recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )

            # get corresponding dataset
            # MNIST 28*28
            (train_loader, test_loader), num_class = (dataloaders.FashionMNIST(args.batch_size)), 10

            if not args.pretrained:
                CapsNet_recon = AlexNet_Module.CapsNet_Recon_MNIST(device).to(device)
                print(f'The CapsAlexNet for FashionMNIST architecture is shown:\n {CapsNet_recon}')  # show architecture
                utils.train_capOutput(CapsNet_recon, train_loader, test_loader, args)
            else:  # 使用预训练
                CapsNet_recon = AlexNet_Module.CapsNet_Recon_MNIST(device).to(device)
                CapsNet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for FashionMNIST architecture is shown:\n {CapsNet_recon}')  # show architecture)
                utils.train_capOutput(CapsNet_recon, train_loader, test_loader, args)

        # AlexCapsNet-Recon
        if NETWORK == 6:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/FashionMNIST/Alexnet/6AlexCapsNet_Recon/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FashionMNIST/Alexnet/6AlexCapsNet_Recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )

            # get corresponding dataset
            # MNIST 28*28
            (train_loader, test_loader), num_class = (dataloaders.FashionMNIST(args.batch_size)), 10

            if not args.pretrained:
                AlexCapsNet_recon = AlexNet_Module.AlexCapsNet_Recon_MNIST(device).to(device)
                print(f'The CapsAlexNet for FashionMNIST architecture is shown:\n {AlexCapsNet_recon}')  # show architecture
                utils.train_capOutput(AlexCapsNet_recon, train_loader, test_loader, args)
            else:  # 使用预训练
                AlexCapsNet_recon = AlexNet_Module.AlexCapsNet_Recon_MNIST(device).to(device)
                AlexCapsNet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for FashionMNIST architecture is shown:\n {AlexCapsNet_recon}')  # show architecture)
                utils.train_capOutput(AlexCapsNet_recon, train_loader, test_loader, args)

        # AlexCapsNet_FullyOut
        if NETWORK == 7:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/FashionMNIST/Alexnet/7AlexCapsNet_FullyOut/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FashionMNIST/Alexnet/7AlexCapsNet_FullyOut/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            # MNIST 28*28
            (train_loader, test_loader), num_class = (dataloaders.FashionMNIST(args.batch_size)), 10

            if not args.pretrained:
                AlexCapsNet_fullyOut = AlexNet_Module.AlexCapsNet_Fully_MNIST(device).to(device)
                print(f'The AlexCapsNet_fullyOut for FashionMNIST architecture is shown:\n {AlexCapsNet_fullyOut}')  # show architecture
                utils.train(AlexCapsNet_fullyOut, train_loader, test_loader, args)
            else:  # 使用预训练
                AlexCapsNet_fullyOut = AlexNet_Module.AlexCapsNet_Fully_MNIST(device).to(device)
                AlexCapsNet_fullyOut.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The AlexCapsNet_fullyOut for FashionMNIST architecture is shown:\n {AlexCapsNet_fullyOut}')  # show architecture)
                utils.train(AlexCapsNet_fullyOut, train_loader, test_loader, args)

        # AlexCapsNet_FullyOut_Recon
        if NETWORK == 8:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/FashionMNIST/Alexnet/8AlexCapsNet_FullyOut_Recon/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FashionMNIST/Alexnet/8AlexCapsNet_FullyOut_Recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )
            # get corresponding dataset
            # FashionMNIST 28*28
            (train_loader, test_loader), num_class = (dataloaders.FashionMNIST(args.batch_size)), 10

            if not args.pretrained:
                AlexCapsNet_fullyOut_recon = AlexNet_Module.AlexCapsNet_Fully_Recon_MNIST(device).to(device)
                print(
                    f'The AlexCapsNet_fullyOut for FashionMNIST architecture is shown:\n {AlexCapsNet_fullyOut_recon}')  # show architecture
                utils.train(AlexCapsNet_fullyOut_recon, train_loader, test_loader, args)
            else:  # 使用预训练
                AlexCapsNet_fullyOut_recon = AlexNet_Module.AlexCapsNet_Fully_Recon_MNIST(device).to(device)
                AlexCapsNet_fullyOut_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(
                    f'The AlexCapsNet_fullyOut for FashionMNIST architecture is shown:\n {AlexCapsNet_fullyOut_recon}')  # show architecture)
                utils.train(AlexCapsNet_fullyOut_recon, train_loader, test_loader, args)

        # Shallow AlexCapsNet Full out
        if NETWORK == 9:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/FashionMNIST/Alexnet/9S_AlexCapsNet_F/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FashionMNIST/Alexnet/9S_AlexCapsNet_F/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            # FashionMNIST 28*28
            (train_loader, test_loader), num_class = (dataloaders.FashionMNIST(args.batch_size)), 10

            if not args.pretrained:
                AlexCapsNet_fullyOut = AlexNet_Module.S_AlexCapsNet_Fully_MNIST(device).to(device)
                print(f'The S_AlexCapsNet_Fully for MNIST architecture is shown:\n {AlexCapsNet_fullyOut}')  # show architecture
                utils.train(AlexCapsNet_fullyOut, train_loader, test_loader, args)
            else:  # 使用预训练
                AlexCapsNet_fullyOut = AlexNet_Module.S_AlexCapsNet_Fully_MNIST(device).to(device)
                AlexCapsNet_fullyOut.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The S_AlexCapsNet_Fully for MNIST architecture is shown:\n {AlexCapsNet_fullyOut}')  # show architecture)
                utils.train(AlexCapsNet_fullyOut, train_loader, test_loader, args)

        # Shallow AlexCapsNet
        if NETWORK == 10:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/FashionMNIST/Alexnet/10S_AlexCapsNet/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FashionMNIST/Alexnet/10S_AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            # FashionMNIST 28*28
            (train_loader, test_loader), num_class = (dataloaders.FashionMNIST(args.batch_size)), 10

            if not args.pretrained:  # 不使用预训练结果
                s_capAlex = AlexNet_Module.S_AlexCapsNet_MNIST(device).to(device)
                print(f'The S_AlexCapsNet for FashionMNIST architecture is shown:\n {s_capAlex}')  # show architecture
                utils.train_capOutput(s_capAlex, train_loader, test_loader, args)
            else:  # 使用预训练
                s_capAlex = AlexNet_Module.S_AlexCapsNet_MNIST(device).to(device)
                s_capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The S_AlexCapsNet for FashionMNIST architecture is shown:\n {s_capAlex}')  # show architecture)
                utils.train_capOutput(s_capAlex, train_loader, test_loader, args)

    # FOOD101   [512 512]->[224 224] batch 1500(50)
    if DATASET == 5:
        # AlexCapsNet when capsnet as predict layer.
        if NETWORK == 1:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=101, recon_alpha=0.0005,
                            train_save_dir='./Result/FOOD101/Alexnet/1AlexCapsNet/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FOOD101/Alexnet/1AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            # FODD101
            (train_loader, test_loader), num_class = (dataloaders.FOOD101(args.batch_size)), 100

            if not args.pretrained:  # 不使用预训练结果
                capAlex = AlexNet_Module.AlexCapsNet_FOOD101(device).to(device)
                print(f'The AlexCapsNet for AlexCapsNet_FOOD101 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)
            else:  # 使用预训练
                capAlex = AlexNet_Module.AlexCapsNet_FOOD101(device).to(device)
                capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The AlexCapsNet for AlexCapsNet_FOOD101 architecture is shown:\n {capAlex}')  # show architecture)
                utils.train_capOutput(capAlex, train_loader, test_loader, args)

        # Alexnet
        if NETWORK == 2:
            # get parameter
            args = _arparse(epoch=300, lr=0.0005, r=3, lr_decay=0.995, num_class=101, recon_alpha=0.0005,
                            train_save_dir='./Result/FOOD101/Alexnet/2.AlexNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,      # 测试的时候，batch_size为1, 训练为50
                            num_save_epoch=50,  # 保存一次要20MB，训练300epoch的时候，酌情设置， 50
                            pretrained=False,
                            pretrained_weight='./Result/FOOD101/Alexnet/2.AlexNet/train/train1/model_2.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FOOD101(args.batch_size)), 101

            if not args.pretrained:
                Alex = AlexNet_Module.AlexNet_FOOD101().to(device)
                print(f'The AlexNet for FOOD101 architecture is shown:\n {Alex}')  # show architecture
                utils.train(Alex, train_loader, test_loader, args)
            else:   # pretrained
                Alex = AlexNet_Module.AlexNet_FOOD101().to(device)
                Alex.load_state_dict(torch.load(args.pretrained_weights))
                print(f'The AlexNet for FOOD101 architecture is shown:\n {Alex}')  # show architecture
                utils.train(Alex, train_loader, test_loader, args)

        # CapsNet
        if NETWORK == 3:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=101, recon_alpha=0.0005,
                            train_save_dir='./Result/FOOD101/Alexnet/3.CapsNet/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,    # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FOOD101/Alexnet/AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FOOD101(args.batch_size)), 101

            if not args.pretrained:  # 不使用预训练结果
                capnet = AlexNet_Module.CapsNet_FOOD101(device).to(device)
                print(f'The CapsAlexNet for FOOD101 architecture is shown:\n {capnet}')  # show architecture
                utils.train_capOutput(capnet, train_loader, test_loader, args)
            else:  # 使用预训练
                capnet = AlexNet_Module.CapsNet_FOOD101(device).to(device)
                capnet.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for FOOD101 architecture is shown:\n {capnet}')  # show architecture)
                utils.train_capOutput(capnet, train_loader, test_loader, args)

        # CapsAlexNet
        if NETWORK == 4:

                # get parameter
                args = _arparse(epoch=300, lr=0.001, r=3, lr_decay=0.995, num_class=101, recon_alpha=0.0005,
                                train_save_dir='./Result/FOOD101/Alexnet/4.CapsAlexNet/train',
                                train_eval=True,  # false is fine
                                save_all=True,
                                batch_size=50,  # 测试的时候，batch_size为1, 训练为50
                                num_save_epoch=50,  # 保存一次要20MB，训练300epoch的时候，酌情设置， 50
                                pretrained=False,
                                pretrained_weight='./Result/FOOD101/Alexnet/CapsAlexNet/train/train1/model_2.pth',
                                reconstruction=False
                                )
                # get corresponding dataset
                (train_loader, test_loader), num_class = (dataloaders.FOOD101(args.batch_size)), 101

                if not args.pretrained:
                    # get CapsAlexNet
                    capsalex = AlexNet_Module.CapsAlexNet_FOOD101(device).to(device)
                    print(f'The AlexNet for FOOD101 architecture is shown:\n {capsalex}')  # show architecture
                    utils.train(capsalex, train_loader, test_loader, args)
                else:  # pretrained
                    capsalex = AlexNet_Module.CapsAlexNet_FOOD101(device).to(device)
                    capsalex.load_state_dict(torch.load(args.pretrained_weights))
                    print(f'The AlexNet for FOOD101 architecture is shown:\n {capsalex}')  # show architecture
                    utils.train(capsalex, train_loader, test_loader, args)

        # CapsNet with reconstruction
        if NETWORK == 5:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=101, recon_alpha=0.00005,
                            train_save_dir='./Result/FOOD101/Alexnet/5.CapsNet_recon/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FOOD101/Alexnet/CapsNet_recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )
            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.FOOD101(args.batch_size)), 101

            if not args.pretrained:  # 不使用预训练结果
                capnet_recon = AlexNet_Module.CapsNet_Recon_FOOD101(device).to(device)
                print(f'The CapsAlexNet for FOOD101 architecture is shown:\n {capnet_recon}')  # show architecture
                utils.train_capOutput(capnet_recon, train_loader, test_loader, args)
            else:  # 使用预训练
                capnet_recon = AlexNet_Module.CapsNet_Recon_FOOD101(device).to(device)
                capnet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for FOOD101 architecture is shown:\n {capnet_recon}')  # show architecture)
                utils.train_capOutput(capnet_recon, train_loader, test_loader, args)

        # AlexCapsNet with reconstruction
        if NETWORK == 6:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=101, recon_alpha=0.00005,
                            train_save_dir='./Result/FOOD101/Alexnet/6.AlexCapsNet_recon/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FOOD101/Alexnet/AlexCapsNet_recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FOOD101(args.batch_size)), 101

            if not args.pretrained:  # 不使用预训练结果
                alexcapnet_recon = AlexNet_Module.AlexCapsNet_Recon_FOOD101(device).to(device)
                print(f'The CapsAlexNet for FOOD101 architecture is shown:\n {alexcapnet_recon}')  # show architecture
                utils.train_capOutput(alexcapnet_recon, train_loader, test_loader, args)
            else:  # 使用预训练
                alexcapnet_recon = AlexNet_Module.AlexCapsNet_Recon_FOOD101(device).to(device)
                alexcapnet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for FOOD101 architecture is shown:\n {alexcapnet_recon}')  # show architecture)
                utils.train_capOutput(alexcapnet_recon, train_loader, test_loader, args)

        # AlexCpasNet fully contected output
        if NETWORK == 7:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=101, recon_alpha=0.0005,
                            train_save_dir='./Result/FOOD101/Alexnet/7.AlexCapsNet_F/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FOOD101/Alexnet/AlexCapsNet_FullyOut/train/best_train/model_299.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FOOD101(args.batch_size)), 101

            if not args.pretrained:  # 不使用预训练结果
                alexcapnet_fullyOut = AlexNet_Module.AlexCapsNet_Fully_FOOD101(device).to(device)
                print(f'The CapsAlexNet_FullyOut for FOOD101 architecture is shown:\n {alexcapnet_fullyOut}')  # show architecture
                utils.train(alexcapnet_fullyOut, train_loader, test_loader, args)
            else:  # 使用预训练
                alexcapnet_fullyOut = AlexNet_Module.AlexCapsNet_Fully_FOOD101(device).to(device)
                alexcapnet_fullyOut.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet_FullyOut for FOOD101 architecture is shown:\n {alexcapnet_fullyOut}')
                utils.train(alexcapnet_fullyOut, train_loader, test_loader, args)

        # AlexCapsNet_FullyOut_Recon
        if NETWORK == 8:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=101, recon_alpha=0.00005,
                            train_save_dir='./Result/FOOD101/Alexnet/8.AlexCapsNet_F_Recon/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FOOD101/Alexnet/AlexCapsNet_FullyOut_Recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FOOD101(args.batch_size)), 101

            if not args.pretrained:  # 不使用预训练结果
                alexcapnet_F_recon = AlexNet_Module.AlexCapsNet_F_Recon_FOOD101(device).to(device)
                print(f'The CapsAlexNet_Fully_Rencon for FOOD101 architecture is shown:\n {alexcapnet_F_recon}')  # show architecture
                utils.train(alexcapnet_F_recon, train_loader, test_loader, args)
            else:  # 使用预训练
                alexcapnet_F_recon = AlexNet_Module.AlexCapsNet_Fully_FOOD101(device).to(device)
                alexcapnet_F_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet_FullyOut_Recon for FOOD101 architecture is shown:\n {alexcapnet_F_recon}')
                utils.train(alexcapnet_F_recon, train_loader, test_loader, args)

        # Shallow AlexCpasNet fully contected output
        if NETWORK == 9:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=101, recon_alpha=0.0005,
                            train_save_dir='./Result/FOOD101/Alexnet/9.S_AlexCapsNet_F/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FOOD101/Alexnet/9.S_AlexCapsNet_F/train/best_train/model_299.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.FOOD101(args.batch_size)), 101

            if not args.pretrained:  # 不使用预训练结果
                alexcapnet_fullyOut = AlexNet_Module.S_AlexCapsNet_Fully_FOOD101(device).to(device)
                print(f'The S_AlexCapsNet_Fully for FOOD101 architecture is shown:\n {alexcapnet_fullyOut}')  # show architecture
                utils.train(alexcapnet_fullyOut, train_loader, test_loader, args)
            else:  # 使用预训练
                alexcapnet_fullyOut = AlexNet_Module.S_AlexCapsNet_Fully_FOOD101(device).to(device)
                alexcapnet_fullyOut.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The S_AlexCapsNet_Fully for FOOD101 architecture is shown:\n {alexcapnet_fullyOut}')
                utils.train(alexcapnet_fullyOut, train_loader, test_loader, args)

        # Shaollow AlexCapsNet
        if NETWORK == 10:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=101, recon_alpha=0.0005,
                            train_save_dir='./Result/FOOD101/Alexnet/10.S_AlexCapsNet/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FOOD101/Alexnet/10.S_AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.FOOD101(args.batch_size)), 101

            if not args.pretrained:  # 不使用预训练结果
                capAlex = AlexNet_Module.S_AlexCapsNet_FOOD101(device).to(device)
                print(f'The S_CapsAlexNet for FOOD101 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)
            else:                    # 使用预训练
                capAlex = AlexNet_Module.S_AlexCapsNet_FOOD101(device).to(device)
                capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The S_CapsAlexNet for FOOD101 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)

    # FLOWER102
    if DATASET == 6:
        # AlexCapsNet when capsnet as predict layer.
        if NETWORK == 1:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=102, recon_alpha=0.0005,
                            train_save_dir='./Result/FLOWER102/Alexnet/1AlexCapsNet/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FLOWER102/Alexnet/1AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            # FODD101
            (train_loader, test_loader), num_class = (dataloaders.FOOD101(args.batch_size)), 102

            if not args.pretrained:  # 不使用预训练结果
                capAlex = AlexNet_Module.AlexCapsNet_FLOWER102(device).to(device)
                print(
                    f'The AlexCapsNet for AlexCapsNet_FLOWER102 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)
            else:  # 使用预训练
                capAlex = AlexNet_Module.AlexCapsNet_FLOWER102(device).to(device)
                capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(
                    f'The AlexCapsNet for AlexCapsNet_FLOWER102 architecture is shown:\n {capAlex}')  # show architecture)
                utils.train_capOutput(capAlex, train_loader, test_loader, args)

        # Alexnet
        if NETWORK == 2:
            # get parameter
            args = _arparse(epoch=300, lr=0.0005, r=3, lr_decay=0.995, num_class=102, recon_alpha=0.0005,
                            train_save_dir='./Result/FLOWER102/Alexnet/2.AlexNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,      # 测试的时候，batch_size为1, 训练为50
                            num_save_epoch=50,  # 保存一次要20MB，训练300epoch的时候，酌情设置， 50
                            pretrained=False,
                            pretrained_weight='./Result/FLOWER102/Alexnet/2.AlexNet/train/train1/model_2.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FLOWER102(args.batch_size)), 102

            if not args.pretrained:
                Alex = AlexNet_Module.AlexNet_FLOWER102().to(device)
                print(f'The AlexNet for FLOWER102 architecture is shown:\n {Alex}')  # show architecture
                utils.train(Alex, train_loader, test_loader, args)
            else:   # pretrained
                Alex = AlexNet_Module.AlexNet_FLOWER102().to(device)
                Alex.load_state_dict(torch.load(args.pretrained_weights))
                print(f'The AlexNet for FLOWER102 architecture is shown:\n {Alex}')  # show architecture
                utils.train(Alex, train_loader, test_loader, args)

        # CapsNet
        if NETWORK == 3:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=102, recon_alpha=0.0005,
                            train_save_dir='./Result/FLOWER102/Alexnet/3.CapsNet/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,    # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FLOWER102/Alexnet/AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FLOWER102(args.batch_size)), 102

            if not args.pretrained:  # 不使用预训练结果
                capnet = AlexNet_Module.CapsNet_FLOWER102(device).to(device)
                print(f'The CapsAlexNet for FLOWER102 architecture is shown:\n {capnet}')  # show architecture
                utils.train_capOutput(capnet, train_loader, test_loader, args)
            else:  # 使用预训练
                capnet = AlexNet_Module.CapsNet_FLOWER102(device).to(device)
                capnet.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for FLOWER102 architecture is shown:\n {capnet}')  # show architecture)
                utils.train_capOutput(capnet, train_loader, test_loader, args)

        # CapsAlexNet
        if NETWORK == 4:

                # get parameter
                args = _arparse(epoch=300, lr=0.001, r=3, lr_decay=0.995, num_class=102, recon_alpha=0.0005,
                                train_save_dir='./Result/FLOWER102/Alexnet/4.CapsAlexNet/train',
                                train_eval=True,  # false is fine
                                save_all=True,
                                batch_size=50,  # 测试的时候，batch_size为1, 训练为50
                                num_save_epoch=50,  # 保存一次要20MB，训练300epoch的时候，酌情设置， 50
                                pretrained=False,
                                pretrained_weight='./Result/FLOWER102/Alexnet/CapsAlexNet/train/train1/model_2.pth',
                                reconstruction=False
                                )
                # get corresponding dataset
                (train_loader, test_loader), num_class = (dataloaders.FLOWER102(args.batch_size)), 102

                if not args.pretrained:
                    # get CapsAlexNet
                    capsalex = AlexNet_Module.CapsAlexNet_FLOWER102(device).to(device)
                    print(f'The AlexNet for FLOWER102 architecture is shown:\n {capsalex}')  # show architecture
                    utils.train(capsalex, train_loader, test_loader, args)
                else:  # pretrained
                    capsalex = AlexNet_Module.CapsAlexNet_FLOWER102(device).to(device)
                    capsalex.load_state_dict(torch.load(args.pretrained_weights))
                    print(f'The AlexNet for FLOWER102 architecture is shown:\n {capsalex}')  # show architecture
                    utils.train(capsalex, train_loader, test_loader, args)

        # CapsNet with reconstruction
        if NETWORK == 5:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=102, recon_alpha=0.00005,
                            train_save_dir='./Result/FLOWER102/Alexnet/5.CapsNet_recon/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FLOWER102/Alexnet/CapsNet_recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FLOWER102(args.batch_size)), 102

            if not args.pretrained:  # 不使用预训练结果
                capnet_recon = AlexNet_Module.CapsNet_Recon_FLOWER102(device).to(device)
                print(f'The CapsAlexNet for FLOWER102 architecture is shown:\n {capnet_recon}')  # show architecture
                utils.train_capOutput(capnet_recon, train_loader, test_loader, args)
            else:  # 使用预训练
                capnet_recon = AlexNet_Module.CapsNet_Recon_FLOWER102(device).to(device)
                capnet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for FLOWER102 architecture is shown:\n {capnet_recon}')  # show architecture)
                utils.train_capOutput(capnet_recon, train_loader, test_loader, args)

        # AlexCapsNet with reconstruction
        if NETWORK == 6:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=102, recon_alpha=0.0005,
                            train_save_dir='./Result/FLOWER102/Alexnet/6.AlexCapsNet_recon/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FLOWER102/Alexnet/AlexCapsNet_recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FLOWER102(args.batch_size)), 102

            if not args.pretrained:  # 不使用预训练结果
                alexcapnet_recon = AlexNet_Module.AlexCapsNet_Recon_FLOWER102(device).to(device)
                print(f'The CapsAlexNet for FLOWER102 architecture is shown:\n {alexcapnet_recon}')  # show architecture
                utils.train_capOutput(alexcapnet_recon, train_loader, test_loader, args)
            else:  # 使用预训练
                alexcapnet_recon = AlexNet_Module.AlexCapsNet_Recon_FLOWER102(device).to(device)
                alexcapnet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for FLOWER102 architecture is shown:\n {alexcapnet_recon}')  # show architecture)
                utils.train_capOutput(alexcapnet_recon, train_loader, test_loader, args)

        # AlexCpasNet fully contected output
        if NETWORK == 7:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=102, recon_alpha=0.0005,
                            train_save_dir='./Result/FLOWER102/Alexnet/7.AlexCapsNet_F/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FLOWER102/Alexnet/AlexCapsNet_FullyOut/train/best_train/model_299.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FLOWER102(args.batch_size)), 102

            if not args.pretrained:  # 不使用预训练结果
                alexcapnet_fullyOut = AlexNet_Module.AlexCapsNet_Fully_FLOWER102(device).to(device)
                print(f'The CapsAlexNet_FullyOut for FLOWER102 architecture is shown:\n {alexcapnet_fullyOut}')  # show architecture
                utils.train(alexcapnet_fullyOut, train_loader, test_loader, args)
            else:  # 使用预训练
                alexcapnet_fullyOut = AlexNet_Module.AlexCapsNet_Fully_FLOWER102(device).to(device)
                alexcapnet_fullyOut.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet_FullyOut for FLOWER102 architecture is shown:\n {alexcapnet_fullyOut}')
                utils.train(alexcapnet_fullyOut, train_loader, test_loader, args)

        # AlexCapsNet_FullyOut_Recon
        if NETWORK == 8:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=102, recon_alpha=0.00005,
                            train_save_dir='./Result/FLOWER102/Alexnet/8.AlexCapsNet_F_Recon/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FLOWER102/Alexnet/AlexCapsNet_FullyOut_Recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FLOWER102(args.batch_size)), 102

            if not args.pretrained:  # 不使用预训练结果
                alexcapnet_F_recon = AlexNet_Module.AlexCapsNet_F_Recon_FLOWER102(device).to(device)
                print(f'The CapsAlexNet_Fully_Rencon for FLOWER102 architecture is shown:\n {alexcapnet_F_recon}')  # show architecture
                utils.train(alexcapnet_F_recon, train_loader, test_loader, args)
            else:  # 使用预训练
                alexcapnet_F_recon = AlexNet_Module.AlexCapsNet_Fully_FLOWER102(device).to(device)
                alexcapnet_F_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet_FullyOut_Recon for FLOWER102 architecture is shown:\n {alexcapnet_F_recon}')
                utils.train(alexcapnet_F_recon, train_loader, test_loader, args)

        # Shallow AlexCpasNet fully contected output
        if NETWORK == 9:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=102, recon_alpha=0.0005,
                            train_save_dir='./Result/FLOWER102/Alexnet/9.S_AlexCapsNet_F/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FLOWER102/Alexnet/9.S_AlexCapsNet_F/train/best_train/model_299.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.FLOWER102(args.batch_size)), 102

            if not args.pretrained:  # 不使用预训练结果
                alexcapnet_fullyOut = AlexNet_Module.S_AlexCapsNet_Fully_FLOWER102(device).to(device)
                print(f'The S_AlexCapsNet_Fully for FLOWER102 architecture is shown:\n {alexcapnet_fullyOut}')  # show architecture
                utils.train(alexcapnet_fullyOut, train_loader, test_loader, args)
            else:  # 使用预训练
                alexcapnet_fullyOut = AlexNet_Module.S_AlexCapsNet_Fully_FLOWER102(device).to(device)
                alexcapnet_fullyOut.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The S_AlexCapsNet_Fully for FLOWER102 architecture is shown:\n {alexcapnet_fullyOut}')
                utils.train(alexcapnet_fullyOut, train_loader, test_loader, args)

        # Shaollow AlexCapsNet
        if NETWORK == 10:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=102, recon_alpha=0.0005,
                            train_save_dir='./Result/FLOWER102/Alexnet/10.S_AlexCapsNet/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/FLOWER102/Alexnet/10.S_AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FLOWER102(args.batch_size)), 102

            if not args.pretrained:  # 不使用预训练结果
                capAlex = AlexNet_Module.S_AlexCapsNet_FLOWER102(device).to(device)
                print(f'The S_CapsAlexNet for FLOWER102 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)
            else:                    # 使用预训练
                capAlex = AlexNet_Module.S_AlexCapsNet_FLOWER102(device).to(device)
                capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The S_CapsAlexNet for FLOWER102 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)

    # SVHN
    if DATASET == 7:

        # AlexCapsNet when capsnet as predict layer.
        if NETWORK == 1:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/SVHN/Alexnet/1AlexCapsNet/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/SVHN/Alexnet/1AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            train_loader, test_loader = dataloaders.SVHN(args.batch_size)

            if not args.pretrained:  # 不使用预训练结果
                capAlex = AlexNet_Module.AlexCapsNet_CIFAR10(device).to(device)
                print(
                    f'The AlexCapsNet for AlexCapsNet_SVHN architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)
            else:  # 使用预训练
                capAlex = AlexNet_Module.AlexCapsNet_CIFAR10(device).to(device)
                capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(
                    f'The AlexCapsNet for AlexCapsNet_SVHN architecture is shown:\n {capAlex}')  # show architecture)
                utils.train_capOutput(capAlex, train_loader, test_loader, args)

        # Alexnet
        if NETWORK == 2:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, r=3, lr_decay=0.995, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/SVHN/Alexnet/2.AlexNet/train',
                            train_eval=True,  # false is fine
                            save_all=True,
                            batch_size=50,  # 测试的时候，batch_size为1, 训练为50
                            num_save_epoch=1,  # 保存一次要20MB，训练300epoch的时候，酌情设置， 50
                            pretrained=False,
                            pretrained_weight='./Result/SVHN/Alexnet/AlexNet/train/train1/model_2.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.SVHN(args.batch_size)), 10

            if not args.pretrained:
                # get CapsAlexNet
                Alex = AlexNet_Module.AlexNet_CIFAR10().to(device)
                print(f'The AlexNet for SVHN architecture is shown:\n {Alex}')  # show architecture
                utils.train(Alex, train_loader, test_loader, args)
            else:   # pretrained
                Alex = AlexNet_Module.AlexNet_CIFAR10().to(device)
                Alex.load_state_dict(torch.load(args.pretrained_weights))
                print(f'The AlexNet for SVHN architecture is shown:\n {Alex}')  # show architecture
                utils.train(Alex, train_loader, test_loader, args)

        # CapsNet
        if NETWORK == 3:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/SVHN/Alexnet/3.CapsNet/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,    # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/SVHN/Alexnet/AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.SVHN(args.batch_size)), 10

            if not args.pretrained:  # 不使用预训练结果
                capnet = AlexNet_Module.CapsNet_CIFAR10(device).to(device)
                print(f'The CapsAlexNet for SVHN architecture is shown:\n {capnet}')  # show architecture
                utils.train_capOutput(capnet, train_loader, test_loader, args)
            else:  # 使用预训练
                capnet = AlexNet_Module.CapsNet_CIFAR10(device).to(device)
                capnet.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for SVHN architecture is shown:\n {capnet}')  # show architecture)
                utils.train_capOutput(capnet, train_loader, test_loader, args)

        # CapsAlexNet, capsnet layer is inside Alexnet·
        if NETWORK == 4:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, r=3, lr_decay=0.995, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/SVHN/Alexnet/4.CapsAlexNet/train',
                            train_eval=True,  # false is fine
                            save_all=True,
                            batch_size=50,  # 测试的时候，batch_size为1, 训练为50
                            num_save_epoch=50,  # 保存一次要20MB，训练300epoch的时候，酌情设置， 50
                            pretrained=False,
                            pretrained_weight='./Result/SVHN/Alexnet/CapsAlexNet/train/train1/model_2.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.SVHN(args.batch_size)), 10

            if not args.pretrained:
                # get CapsAlexNet
                capsalex = AlexNet_Module.CapsAlexNet_CIFAR10(device).to(device)
                print(f'The CapsAlexNet for SVHN architecture is shown:\n {capsalex}')  # show architecture
                utils.train(capsalex, train_loader, test_loader, args)
            else:  # pretrained
                capsalex = AlexNet_Module.CapsAlexNet_CIFAR10(device).to(device)
                capsalex.load_state_dict(torch.load(args.pretrained_weights))
                print(f'The CapsAlexNet for SVHN architecture is shown:\n {capsalex}')  # show architecture
                utils.train(capsalex, train_loader, test_loader, args)

        # CapsNet with reconstruction
        if NETWORK == 5:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/SVHN/Alexnet/5.CapsNet_recon/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=1,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/SVHN/Alexnet/CapsNet_recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )
            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.SVHN(args.batch_size)), 10

            if not args.pretrained:  # 不使用预训练结果
                capnet_recon = AlexNet_Module.CapsNet_Recon_CIFAR10(device).to(device)
                print(f'The CapsAlexNet for CIFAR10 architecture is shown:\n {capnet_recon}')  # show architecture
                utils.train_capOutput(capnet_recon, train_loader, test_loader, args)
            else:  # 使用预训练
                capnet_recon = AlexNet_Module.CapsNet_Recon_CIFAR10(device).to(device)
                capnet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for CIFAR10 architecture is shown:\n {capnet_recon}')  # show architecture)
                utils.train_capOutput(capnet_recon, train_loader, test_loader, args)

        # AlexCapsNet with reconstruction
        if NETWORK == 6:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/SVHN/Alexnet/6.AlexCapsNet_recon/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,  # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=1,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/SVHN/Alexnet/AlexCapsNet_recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )
            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.SVHN(args.batch_size)), 10

            if not args.pretrained:  # 不使用预训练结果
                alexcapnet_recon = AlexNet_Module.AlexCapsNet_Recon_CIFAR10(device).to(device)
                print(f'The CapsAlexNet for CIFAR10 architecture is shown:\n {alexcapnet_recon}')  # show architecture
                utils.train_capOutput(alexcapnet_recon, train_loader, test_loader, args)
            else:  # 使用预训练
                alexcapnet_recon = AlexNet_Module.AlexCapsNet_Recon_CIFAR10(device).to(device)
                alexcapnet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for CIFAR10 architecture is shown:\n {alexcapnet_recon}')  # show architecture)
                utils.train_capOutput(alexcapnet_recon, train_loader, test_loader, args)

        # Shallow AlexCapsNet_F
        if NETWORK == 9:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR10/Alexnet/9.S_AlexCapsNet_F/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=1,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/CIFAR10/Alexnet/9.S_AlexCapsNet_F/train/best_train/model_299.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.CIFAR10(args.batch_size)), 10

            if not args.pretrained:  # 不使用预训练结果
                alexcapnet_fullyOut = AlexNet_Module.S_AlexCapsNet_Fully_CIFAR10(device).to(device)
                print(f'The S_CapsAlexNet_F for CIFAR10 architecture is shown:\n {alexcapnet_fullyOut}')  # show architecture
                utils.train(alexcapnet_fullyOut, train_loader, test_loader, args)
            else:  # 使用预训练
                alexcapnet_fullyOut = AlexNet_Module.S_AlexCapsNet_Fully_CIFAR10(device).to(device)
                alexcapnet_fullyOut.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet_F for CIFAR10 architecture is shown:\n {alexcapnet_fullyOut}')
                utils.train(alexcapnet_fullyOut, train_loader, test_loader, args)

        # Shallow AlexCapsNet_F
        if NETWORK == 10:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR10/Alexnet/10.S_AlexCapsNet/train',
                            train_eval=True,  # 训练的时候会评测
                            save_all=True,    # True则保存所有的pth文件
                            batch_size=50,
                            num_save_epoch=1,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/CIFAR10/Alexnet/10.S_AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            # CIFAR10   train images: 50000  50(batch size)*1000(batch number)
            (train_loader, test_loader), num_class = (dataloaders.CIFAR10(args.batch_size)), 10

            if not args.pretrained:  # 不使用预训练结果
                capAlex = AlexNet_Module.S_AlexCapsNet_CIFAR10(device).to(device)
                print(f'The CapsAlexNet for CIFAR10 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)
            else:  # 使用预训练
                capAlex = AlexNet_Module.S_AlexCapsNet_CIFAR10(device).to(device)
                capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for CIFAR10 architecture is shown:\n {capAlex}')  # show architecture)
                utils.train_capOutput(capAlex, train_loader, test_loader, args)




if __name__ == '__main__':
    # CIFAR10: 1    CIFAR100: 2     MINIST：     3   FashionMNIST: 4
    # FOOD101: 5    FLOWER102:  6   SVHN    :   7
    DATASET = 7

    # AlexCapsNet: 1      AlexNet: 2            CapsNet:3               CapsAlexNet: 4
    # CapsNet_Recon: 5    AlexCapsNet_Recon: 6  AlexCapsNet_Fully: 7    AlexCapsNet_Fully_Recon: 8
    # S_AlexCapsNet_F:9   S_AlexCapsNet:10
    NETWORK = 2

    main(DATASET, NETWORK)





