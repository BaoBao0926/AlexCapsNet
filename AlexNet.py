import argparse
import torch
import utils
import AlexNet_Module

dataloaders = utils.dataloaders()
dataloaders_recon = utils.dataloaders_recon()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _arparse(epoch, batch_size, lr, lr_decay, r, num_save_epoch, train_save_dir, train_eval,
             pretrained, pretrained_weight, num_class, save_all, reconstruction, recon_alpha):
    parser = argparse.ArgumentParser(description="")
    # trainning parameters
    parser.add_argument('--epochs', default=epoch, type=int)
    parser.add_argument('--batch_size', default=batch_size, type=int)
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

        # AlexCapsNet
        if NETWORK == 1:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR10/Alexnet/1.AlexCapsNet/train',
                            train_eval=True,  # train while testing
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=1,
                            pretrained=False,  # whether to use the pretrained weight
                            pretrained_weight='./Result/CIFAR10/Alexnet/AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.CIFAR10(args.batch_size)), 10

            if not args.pretrained:
                capAlex = AlexNet_Module.AlexCapsNet_CIFAR10(device).to(device)
                print(f'The AlexCapsNet for CIFAR10 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)
            else:
                capAlex = AlexNet_Module.AlexCapsNet_CIFAR10(device).to(device)
                capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The AlexCapsNet for CIFAR10 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)

        # Alexnet
        if NETWORK == 2:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, r=3, lr_decay=0.995, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR10/Alexnet/2.AlexNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=1,
                            pretrained=False,
                            pretrained_weight='./Result/CIFAR10/Alexnet/AlexNet/train/train1/model_2.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.CIFAR10(args.batch_size)), 10

            if not args.pretrained:
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
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/CIFAR10/Alexnet/AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.CIFAR10(args.batch_size)), 10

            if not args.pretrained:
                capnet = AlexNet_Module.CapsNet_CIFAR10(device).to(device)
                print(f'The CapsNet for CIFAR10 architecture is shown:\n {capnet}')  # show architecture
                utils.train_capOutput(capnet, train_loader, test_loader, args)
            else:
                capnet = AlexNet_Module.CapsNet_CIFAR10(device).to(device)
                capnet.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsNet for CIFAR10 architecture is shown:\n {capnet}')  # show architecture)
                utils.train_capOutput(capnet, train_loader, test_loader, args)

        # CapsNet with reconstruction
        if NETWORK == 4:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR10/Alexnet/4.CapsNet_Recon/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=1,
                            pretrained=False,
                            pretrained_weight='./Result/CIFAR10/Alexnet/4.CapsNet_recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.CIFAR10(args.batch_size)), 10

            if not args.pretrained:
                capnet_recon = AlexNet_Module.CapsNet_Recon_CIFAR10(device).to(device)
                print(f'The CapsNet-Recon for CIFAR10 architecture is shown:\n {capnet_recon}')  # show architecture
                utils.train_capOutput(capnet_recon, train_loader, test_loader, args)
            else:
                capnet_recon = AlexNet_Module.CapsNet_Recon_CIFAR10(device).to(device)
                capnet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for CIFAR10 architecture is shown:\n {capnet_recon}')  # show architecture)
                utils.train_capOutput(capnet_recon, train_loader, test_loader, args)

        # AlexCapsNet-Reconstruction
        if NETWORK == 5:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR10/Alexnet/5.AlexCapsNet_Recon/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=1,
                            pretrained=False,
                            pretrained_weight='./Result/CIFAR10/Alexnet/5.AlexCapsNet_Recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.CIFAR10(args.batch_size)), 10

            if not args.pretrained:
                alexcapnet_recon = AlexNet_Module.AlexCapsNet_Recon_CIFAR10(device).to(device)
                print(f'The AlexCapsNet-Recon for CIFAR10 architecture is shown:\n {alexcapnet_recon}')  # show architecture
                utils.train_capOutput(alexcapnet_recon, train_loader, test_loader, args)
            else:
                alexcapnet_recon = AlexNet_Module.AlexCapsNet_Recon_CIFAR10(device).to(device)
                alexcapnet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for CIFAR10 architecture is shown:\n {alexcapnet_recon}')  # show architecture
                utils.train_capOutput(alexcapnet_recon, train_loader, test_loader, args)

        # Shallow AlexCapsNet
        if NETWORK == 6:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR10/Alexnet/6.S_AlexCapsNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=1,
                            pretrained=False,
                            pretrained_weight='./Result/CIFAR10/Alexnet/6.S_AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.CIFAR10(args.batch_size)), 10

            if not args.pretrained:
                capAlex = AlexNet_Module.S_AlexCapsNet_CIFAR10(device).to(device)
                print(f'The S-AlexCapsNet for CIFAR10 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)
            else:
                capAlex = AlexNet_Module.S_AlexCapsNet_CIFAR10(device).to(device)
                capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for CIFAR10 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)

    # CIFAR100
    if DATASET == 2:

        # AlexCapsNet
        if NETWORK == 1:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=100, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR100/Alexnet/1.AlexCapsNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/CIFAR100/Alexnet/1.AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.CIFAR100(args.batch_size)), 100

            if not args.pretrained:
                capAlex = AlexNet_Module.AlexCapsNet_CIFAR100(device).to(device)
                print(f'The AlexCapsNet for CIFAR100 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)
            else:
                capAlex = AlexNet_Module.AlexCapsNet_CIFAR100(device).to(device)
                capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The AlexCapsNet for CIFAR100 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)

        # Alexnet
        if NETWORK == 2:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, r=3, lr_decay=0.995, num_class=100, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR100/Alexnet/2.AlexNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/CIFAR100/Alexnet/2.AlexNet/train/train1/model_2.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.CIFAR100(args.batch_size)), 100

            if not args.pretrained:
                # get CapsAlexNet
                Alex = AlexNet_Module.AlexNet_CIFAR100().to(device)
                print(f'The AlexNet for CIFAR100 architecture is shown:\n {Alex}')  # show architecture
                utils.train(Alex, train_loader, test_loader, args)
            else:   # pretrained
                Alex = AlexNet_Module.AlexNet_CIFAR100().to(device)
                Alex.load_state_dict(torch.load(args.pretrained_weights))
                print(f'The AlexNet for CIFAR100 architecture is shown:\n {Alex}')  # show architecture
                utils.train(Alex, train_loader, test_loader, args)

        # CapsNet
        if NETWORK == 3:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=100, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR100/Alexnet/3.CapsNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/CIFAR100/Alexnet/3.CapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.CIFAR100(args.batch_size)), 100

            if not args.pretrained:
                capnet = AlexNet_Module.CapsNet_CIFAR100(device).to(device)
                print(f'The CapsNet for CIFAR100 architecture is shown:\n {capnet}')  # show architecture
                utils.train_capOutput(capnet, train_loader, test_loader, args)
            else:
                capnet = AlexNet_Module.CapsNet_CIFAR100(device).to(device)
                capnet.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsNet for CIFAR100 architecture is shown:\n {capnet}')  # show architecture
                utils.train_capOutput(capnet, train_loader, test_loader, args)

        # CapsNet with reconstruction
        if NETWORK == 4:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=100, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR100/Alexnet/4.CapsNet_Recon/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=1,
                            pretrained=False,
                            pretrained_weight='./Result/CIFAR100/Alexnet/4.CapsNet_Recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.CIFAR100(args.batch_size)), 100

            if not args.pretrained:
                capnet_recon = AlexNet_Module.CapsNet_Recon_CIFAR100(device).to(device)
                print(f'The CapsxNet-Recon for CIFAR100 architecture is shown:\n {capnet_recon}')  # show architecture
                utils.train_capOutput(capnet_recon, train_loader, test_loader, args)
            else:
                capnet_recon = AlexNet_Module.CapsNet_Recon_CIFAR100(device).to(device)
                capnet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsNet-Recon for CIFAR100 architecture is shown:\n {capnet_recon}')  # show architecture
                utils.train_capOutput(capnet_recon, train_loader, test_loader, args)

        # AlexCapsNet with reconstruction
        if NETWORK == 5:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=100, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR100/Alexnet/5.AlexCapsNet_Recon/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=1,
                            pretrained=False,
                            pretrained_weight='./Result/CIFAR100/Alexnet/5.AlexCapsNet_Recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.CIFAR100(args.batch_size)), 100

            if not args.pretrained:
                alexcapnet_recon = AlexNet_Module.AlexCapsNet_Recon_CIFAR100(device).to(device)
                print(f'The AlexCapsNet-Recon for CIFAR100 architecture is shown:\n {alexcapnet_recon}')  # show architecture
                utils.train_capOutput(alexcapnet_recon, train_loader, test_loader, args)
            else:
                alexcapnet_recon = AlexNet_Module.AlexCapsNet_Recon_CIFAR100(device).to(device)
                alexcapnet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The AlexCapsNet-Recon for CIFAR100 architecture is shown:\n {alexcapnet_recon}')  # show architecture
                utils.train_capOutput(alexcapnet_recon, train_loader, test_loader, args)

        # Shaollow AlexCapsNet
        if NETWORK == 6:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=100, recon_alpha=0.0005,
                            train_save_dir='./Result/CIFAR100/Alexnet/6.S_AlexCapsNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/CIFAR100/Alexnet/6.S_AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.CIFAR100(args.batch_size)), 100

            if not args.pretrained:
                capAlex = AlexNet_Module.S_AlexCapsNet_CIFAR100(device).to(device)
                print(f'The S-AlexCapsNet for CIFAR100 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)
            else:
                capAlex = AlexNet_Module.S_AlexCapsNet_CIFAR100(device).to(device)
                capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The S_AlexCapsNet for CIFAR100 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)

    # MINIST 28*28
    if DATASET == 3:

        # AlexCapsNet
        if NETWORK == 1:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/MNIST/Alexnet/1.AlexCapsNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/MNIST/Alexnet/1.AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.MNIST(args.batch_size)), 10


            if not args.pretrained:
                capAlex = AlexNet_Module.AlexCapsNet_MNIST(device).to(device)
                print(f'The AlexCapsNet for MNIST architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)
            else:
                capAlex = AlexNet_Module.AlexCapsNet_MNIST(device).to(device)
                capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The AlexCapsNet for MNIST architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)

        # AlexNet
        if NETWORK == 2:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/MNIST/Alexnet/2.AlexNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,  # 是否要使用使用预训练
                            pretrained_weight='./Result/MNIST/Alexnet/2.AlexNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.MNIST(args.batch_size)), 10

            if not args.pretrained:
                AlexNet = AlexNet_Module.AlexNet_MNIST(device).to(device)
                print(f'The AlexNet for MNIST architecture is shown:\n {AlexNet}')  # show architecture
                utils.train(AlexNet, train_loader, test_loader, args)
            else:
                AlexNet = AlexNet_Module.AlexNet_MNIST(device).to(device)
                AlexNet.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The AlexNet for MNIST architecture is shown:\n {AlexNet}')  # show architecture)
                utils.train(AlexNet, train_loader, test_loader, args)

        # CapsNet
        if NETWORK == 3:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/MNIST/Alexnet/3.CapsNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/MNIST/Alexnet/3.CapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.MNIST(args.batch_size)), 10

            if not args.pretrained:
                capnet = AlexNet_Module.CapsNet_MNIST(device).to(device)
                print(f'The Capsnet for MNIST architecture is shown:\n {capnet}')  # show architecture
                utils.train_capOutput(capnet, train_loader, test_loader, args)
            else:
                capnet = AlexNet_Module.CapsNet_MNIST(device).to(device)
                capnet.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsNet for MNIST architecture is shown:\n {capnet}')  # show architecture
                utils.train_capOutput(capnet, train_loader, test_loader, args)

        # CapsNet-Recon
        if NETWORK == 4:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/MNIST/Alexnet/4.CapsNet_Recon/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/MNIST/Alexnet/4.CapsNet_Recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )

            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.MNIST(args.batch_size)), 10

            if not args.pretrained:
                CapsNet_recon = AlexNet_Module.CapsNet_Recon_MNIST(device).to(device)
                print(f'The CapsNet-Recon for MNIST architecture is shown:\n {CapsNet_recon}')  # show architecture
                utils.train_capOutput(CapsNet_recon, train_loader, test_loader, args)
            else:  # 使用预训练
                CapsNet_recon = AlexNet_Module.CapsNet_Recon_MNIST(device).to(device)
                CapsNet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsNet-Recon for MNIST architecture is shown:\n {CapsNet_recon}')  # show architecture
                utils.train_capOutput(CapsNet_recon, train_loader, test_loader, args)

        # AlexCapsNet-Recon
        if NETWORK == 5:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/MNIST/Alexnet/5.AlexCapsNet_Recon/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/MNIST/Alexnet/5.AlexCapsNet_Recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )

            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.MNIST(args.batch_size)), 10

            if not args.pretrained:
                AlexCapsNet_recon = AlexNet_Module.AlexCapsNet_Recon_MNIST(device).to(device)
                print(f'The AlexCapsNet-Recon for MNIST architecture is shown:\n {AlexCapsNet_recon}')  # show architecture
                utils.train_capOutput(AlexCapsNet_recon, train_loader, test_loader, args)
            else:
                AlexCapsNet_recon = AlexNet_Module.AlexCapsNet_Recon_MNIST(device).to(device)
                AlexCapsNet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The AlexCapsNet-Recon for MNIST architecture is shown:\n {AlexCapsNet_recon}')  # show architecture
                utils.train_capOutput(AlexCapsNet_recon, train_loader, test_loader, args)

        # Shallow AlexCapsNet
        if NETWORK == 6:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/MNIST/Alexnet/6.S_AlexCapsNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/MNIST/Alexnet/6.S_AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            # MNIST 28*28
            (train_loader, test_loader), num_class = (dataloaders.MNIST(args.batch_size)), 10

            if not args.pretrained:
                s_capAlex = AlexNet_Module.S_AlexCapsNet_MNIST(device).to(device)
                print(f'The S_AlexCapsNet for MNIST architecture is shown:\n {s_capAlex}')  # show architecture
                utils.train_capOutput(s_capAlex, train_loader, test_loader, args)
            else:
                s_capAlex = AlexNet_Module.S_AlexCapsNet_MNIST(device).to(device)
                s_capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The S_AlexCapsNet for MNIST architecture is shown:\n {s_capAlex}')  # show architecture
                utils.train_capOutput(s_capAlex, train_loader, test_loader, args)

    # FashionMINIST 28*28
    if DATASET == 4:

        # AlexCapsNet when capsnet as predict layer.
        if NETWORK == 1:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/FashionMNIST/Alexnet/1.AlexCapsNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/FashionMNIST/Alexnet/1.AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FashionMNIST(args.batch_size)), 10


            if not args.pretrained:
                capAlex = AlexNet_Module.AlexCapsNet_MNIST(device).to(device)
                print(f'The AlexCapsNet for FashionMNIST architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)
            else:
                capAlex = AlexNet_Module.AlexCapsNet_MNIST(device).to(device)
                capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The AlexCapsNet for FashionMNIST architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)

        # AlexNet
        if NETWORK == 2:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/FashionMNIST/Alexnet/2.AlexNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/FashionMNIST/Alexnet/2.AlexNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FashionMNIST(args.batch_size)), 10

            if not args.pretrained:
                AlexNet = AlexNet_Module.AlexNet_MNIST(device).to(device)
                print(f'The AlexNet for FashionMNIST architecture is shown:\n {AlexNet}')  # show architecture
                utils.train(AlexNet, train_loader, test_loader, args)
            else:
                AlexNet = AlexNet_Module.AlexNet_MNIST(device).to(device)
                AlexNet.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The AlexNet for FashionMNIST architecture is shown:\n {AlexNet}')  # show architecture
                utils.train(AlexNet, train_loader, test_loader, args)

        # CapsNet
        if NETWORK == 3:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/FashionMNIST/Alexnet/3.CapsNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/FashionMNIST/Alexnet/3.CapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FashionMNIST(args.batch_size)), 10

            if not args.pretrained:
                capnet = AlexNet_Module.CapsNet_MNIST(device).to(device)
                print(f'The Capsnet for FashionMNIST architecture is shown:\n {capnet}')  # show architecture
                utils.train_capOutput(capnet, train_loader, test_loader, args)
            else:
                capnet = AlexNet_Module.CapsNet_MNIST(device).to(device)
                capnet.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsNet for FashionMNIST architecture is shown:\n {capnet}')  # show architecture
                utils.train_capOutput(capnet, train_loader, test_loader, args)

        # CapsNet-Recon
        if NETWORK == 4:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/FashionMNIST/Alexnet/4.CapsNet_Recon/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/FashionMNIST/Alexnet/4.CapsNet_Recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )

            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FashionMNIST(args.batch_size)), 10

            if not args.pretrained:
                CapsNet_recon = AlexNet_Module.CapsNet_Recon_MNIST(device).to(device)
                print(f'The CapsNet-Recon for FashionMNIST architecture is shown:\n {CapsNet_recon}')  # show architecture
                utils.train_capOutput(CapsNet_recon, train_loader, test_loader, args)
            else:
                CapsNet_recon = AlexNet_Module.CapsNet_Recon_MNIST(device).to(device)
                CapsNet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsNet-Recon for FashionMNIST architecture is shown:\n {CapsNet_recon}')  # show architecture
                utils.train_capOutput(CapsNet_recon, train_loader, test_loader, args)

        # AlexCapsNet-Recon
        if NETWORK == 5:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/FashionMNIST/Alexnet/5.AlexCapsNet_Recon/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/FashionMNIST/Alexnet/5.AlexCapsNet_Recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )

            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FashionMNIST(args.batch_size)), 10

            if not args.pretrained:
                AlexCapsNet_recon = AlexNet_Module.AlexCapsNet_Recon_MNIST(device).to(device)
                print(f'The AlexCapsNet-Recon for FashionMNIST architecture is shown:\n {AlexCapsNet_recon}')  # show architecture
                utils.train_capOutput(AlexCapsNet_recon, train_loader, test_loader, args)
            else:
                AlexCapsNet_recon = AlexNet_Module.AlexCapsNet_Recon_MNIST(device).to(device)
                AlexCapsNet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The AlexCapsNet-Recon for FashionMNIST architecture is shown:\n {AlexCapsNet_recon}')  # show architecture
                utils.train_capOutput(AlexCapsNet_recon, train_loader, test_loader, args)

        # Shallow AlexCapsNet
        if NETWORK == 6:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/FashionMNIST/Alexnet/6.S_AlexCapsNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/FashionMNIST/Alexnet/6.S_AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            # FashionMNIST 28*28
            (train_loader, test_loader), num_class = (dataloaders.FashionMNIST(args.batch_size)), 10

            if not args.pretrained:
                s_capAlex = AlexNet_Module.S_AlexCapsNet_MNIST(device).to(device)
                print(f'The S_AlexCapsNet for FashionMNIST architecture is shown:\n {s_capAlex}')  # show architecture
                utils.train_capOutput(s_capAlex, train_loader, test_loader, args)
            else:
                s_capAlex = AlexNet_Module.S_AlexCapsNet_MNIST(device).to(device)
                s_capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The S_AlexCapsNet for FashionMNIST architecture is shown:\n {s_capAlex}')  # show architecture
                utils.train_capOutput(s_capAlex, train_loader, test_loader, args)

    # FOOD101   [512 512]->[224 224]
    if DATASET == 5:
        # AlexCapsNet
        if NETWORK == 1:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=101, recon_alpha=0.0005,
                            train_save_dir='./Result/FOOD101/Alexnet/1.AlexCapsNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/FOOD101/Alexnet/1.AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            # FODD101
            (train_loader, test_loader), num_class = (dataloaders.FOOD101(args.batch_size)), 100

            if not args.pretrained:
                capAlex = AlexNet_Module.AlexCapsNet_FOOD101(device).to(device)
                print(f'The AlexCapsNet for FOOD101 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)
            else:
                capAlex = AlexNet_Module.AlexCapsNet_FOOD101(device).to(device)
                capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The AlexCapsNet for FOOD101 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)

        # Alexnet
        if NETWORK == 2:
            # get parameter
            args = _arparse(epoch=300, lr=0.0005, r=3, lr_decay=0.995, num_class=101, recon_alpha=0.0005,
                            train_save_dir='./Result/FOOD101/Alexnet/2.AlexNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
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
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/FOOD101/Alexnet/3.CapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FOOD101(args.batch_size)), 101

            if not args.pretrained:
                capnet = AlexNet_Module.CapsNet_FOOD101(device).to(device)
                print(f'The CapsNet for FOOD101 architecture is shown:\n {capnet}')  # show architecture
                utils.train_capOutput(capnet, train_loader, test_loader, args)
            else:
                capnet = AlexNet_Module.CapsNet_FOOD101(device).to(device)
                capnet.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsNet for FOOD101 architecture is shown:\n {capnet}')  # show architecture
                utils.train_capOutput(capnet, train_loader, test_loader, args)

        # CapsNet with reconstruction
        if NETWORK == 4:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=101, recon_alpha=0.00005,
                            train_save_dir='./Result/FOOD101/Alexnet/4.CapsNet_Recon/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/FOOD101/Alexnet/4.CapsNet_Recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FOOD101(args.batch_size)), 101

            if not args.pretrained:
                capnet_recon = AlexNet_Module.CapsNet_Recon_FOOD101(device).to(device)
                print(f'The CapsNet for FOOD101 architecture is shown:\n {capnet_recon}')  # show architecture
                utils.train_capOutput(capnet_recon, train_loader, test_loader, args)
            else:
                capnet_recon = AlexNet_Module.CapsNet_Recon_FOOD101(device).to(device)
                capnet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsNet for FOOD101 architecture is shown:\n {capnet_recon}')  # show architecture
                utils.train_capOutput(capnet_recon, train_loader, test_loader, args)

        # AlexCapsNet with reconstruction
        if NETWORK == 5:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=101, recon_alpha=0.00005,
                            train_save_dir='./Result/FOOD101/Alexnet/5.AlexCapsNet_Recon/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/FOOD101/Alexnet/5.AlexCapsNet_Recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FOOD101(args.batch_size)), 101

            if not args.pretrained:
                alexcapnet_recon = AlexNet_Module.AlexCapsNet_Recon_FOOD101(device).to(device)
                print(f'The AlexCapsNet-Recon for FOOD101 architecture is shown:\n {alexcapnet_recon}')  # show architecture
                utils.train_capOutput(alexcapnet_recon, train_loader, test_loader, args)
            else:
                alexcapnet_recon = AlexNet_Module.AlexCapsNet_Recon_FOOD101(device).to(device)
                alexcapnet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The AlexCapsNet-Recon for FOOD101 architecture is shown:\n {alexcapnet_recon}')  # show architecture
                utils.train_capOutput(alexcapnet_recon, train_loader, test_loader, args)

        # Shaollow AlexCapsNet
        if NETWORK == 6:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=101, recon_alpha=0.0005,
                            train_save_dir='./Result/FOOD101/Alexnet/6.S_AlexCapsNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/FOOD101/Alexnet/6.S_AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FOOD101(args.batch_size)), 101

            if not args.pretrained:
                capAlex = AlexNet_Module.S_AlexCapsNet_FOOD101(device).to(device)
                print(f'The S_CapsAlexNet for FOOD101 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)
            else:
                capAlex = AlexNet_Module.S_AlexCapsNet_FOOD101(device).to(device)
                capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The S_CapsAlexNet for FOOD101 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)

    # FLOWER102
    if DATASET == 6:
        # AlexCapsNet
        if NETWORK == 1:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=102, recon_alpha=0.0005,
                            train_save_dir='./Result/FLOWER102/Alexnet/1.AlexCapsNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/FLOWER102/Alexnet/1.AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FOOD101(args.batch_size)), 102

            if not args.pretrained:
                capAlex = AlexNet_Module.AlexCapsNet_FLOWER102(device).to(device)
                print(
                    f'The AlexCapsNet for FLOWER102 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)
            else:
                capAlex = AlexNet_Module.AlexCapsNet_FLOWER102(device).to(device)
                capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(
                    f'The AlexCapsNet for FLOWER102 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)

        # Alexnet
        if NETWORK == 2:
            # get parameter
            args = _arparse(epoch=300, lr=0.0005, r=3, lr_decay=0.995, num_class=102, recon_alpha=0.0005,
                            train_save_dir='./Result/FLOWER102/Alexnet/2.AlexNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
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
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/FLOWER102/Alexnet/3.CapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FLOWER102(args.batch_size)), 102

            if not args.pretrained:
                capnet = AlexNet_Module.CapsNet_FLOWER102(device).to(device)
                print(f'The CapsNet for FLOWER102 architecture is shown:\n {capnet}')  # show architecture
                utils.train_capOutput(capnet, train_loader, test_loader, args)
            else:
                capnet = AlexNet_Module.CapsNet_FLOWER102(device).to(device)
                capnet.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsNet for FLOWER102 architecture is shown:\n {capnet}')  # show architecture
                utils.train_capOutput(capnet, train_loader, test_loader, args)

        # CapsNet with reconstruction
        if NETWORK == 4:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=102, recon_alpha=0.00005,
                            train_save_dir='./Result/FLOWER102/Alexnet/4.CapsNet_Recon/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/FLOWER102/Alexnet/4.CapsNet_Recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FLOWER102(args.batch_size)), 102

            if not args.pretrained:
                capnet_recon = AlexNet_Module.CapsNet_Recon_FLOWER102(device).to(device)
                print(f'The CapsNet-Recon for FLOWER102 architecture is shown:\n {capnet_recon}')  # show architecture
                utils.train_capOutput(capnet_recon, train_loader, test_loader, args)
            else:
                capnet_recon = AlexNet_Module.CapsNet_Recon_FLOWER102(device).to(device)
                capnet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsNet-Recon for FLOWER102 architecture is shown:\n {capnet_recon}')  # show architecture
                utils.train_capOutput(capnet_recon, train_loader, test_loader, args)

        # AlexCapsNet with reconstruction
        if NETWORK == 5:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=102, recon_alpha=0.0005,
                            train_save_dir='./Result/FLOWER102/Alexnet/5.AlexCapsNet_Recon/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/FLOWER102/Alexnet/5.AlexCapsNet_Recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )

            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FLOWER102(args.batch_size)), 102

            if not args.pretrained:
                alexcapnet_recon = AlexNet_Module.AlexCapsNet_Recon_FLOWER102(device).to(device)
                print(f'The AlexCapsNet_Recon for FLOWER102 architecture is shown:\n {alexcapnet_recon}')  # show architecture
                utils.train_capOutput(alexcapnet_recon, train_loader, test_loader, args)
            else:
                alexcapnet_recon = AlexNet_Module.AlexCapsNet_Recon_FLOWER102(device).to(device)
                alexcapnet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The AlexCapsNet_Recon for FLOWER102 architecture is shown:\n {alexcapnet_recon}')  # show architecture
                utils.train_capOutput(alexcapnet_recon, train_loader, test_loader, args)

        # Shaollow AlexCapsNet
        if NETWORK == 6:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=102, recon_alpha=0.0005,
                            train_save_dir='./Result/FLOWER102/Alexnet/6.S_AlexCapsNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/FLOWER102/Alexnet/6.S_AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.FLOWER102(args.batch_size)), 102

            if not args.pretrained:
                capAlex = AlexNet_Module.S_AlexCapsNet_FLOWER102(device).to(device)
                print(f'The S_CapsAlexNet for FLOWER102 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)
            else:
                capAlex = AlexNet_Module.S_AlexCapsNet_FLOWER102(device).to(device)
                capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The S_CapsAlexNet for FLOWER102 architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)

    # SVHN
    if DATASET == 7:

        # AlexCapsNet
        if NETWORK == 1:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/SVHN/Alexnet/1.AlexCapsNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/SVHN/Alexnet/1.AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            train_loader, test_loader = dataloaders.SVHN(args.batch_size)

            if not args.pretrained:
                capAlex = AlexNet_Module.AlexCapsNet_CIFAR10(device).to(device)
                print(
                    f'The AlexCapsNet for SVHN architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)
            else:
                capAlex = AlexNet_Module.AlexCapsNet_CIFAR10(device).to(device)
                capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(
                    f'The AlexCapsNet for SVHN architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)

        # Alexnet
        if NETWORK == 2:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, r=3, lr_decay=0.995, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/SVHN/Alexnet/2.AlexNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=1,
                            pretrained=False,
                            pretrained_weight='./Result/SVHN/Alexnet/2.AlexNet/train/train1/model_2.pth',
                            reconstruction=False
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.SVHN(args.batch_size)), 10

            if not args.pretrained:
                # get AlexNet
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
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=50,
                            pretrained=False,
                            pretrained_weight='./Result/SVHN/Alexnet/3.CapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.SVHN(args.batch_size)), 10

            if not args.pretrained:
                capnet = AlexNet_Module.CapsNet_CIFAR10(device).to(device)
                print(f'The CapsNet for SVHN architecture is shown:\n {capnet}')  # show architecture
                utils.train_capOutput(capnet, train_loader, test_loader, args)
            else:
                capnet = AlexNet_Module.CapsNet_CIFAR10(device).to(device)
                capnet.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsNet for SVHN architecture is shown:\n {capnet}')  # show architecture
                utils.train_capOutput(capnet, train_loader, test_loader, args)

        # CapsNet-Recon
        if NETWORK == 4:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/SVHN/Alexnet/4.CapsNet_Recon/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=1,
                            pretrained=False,
                            pretrained_weight='./Result/SVHN/Alexnet/4.CapsNet_Recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.SVHN(args.batch_size)), 10

            if not args.pretrained:
                capnet_recon = AlexNet_Module.CapsNet_Recon_CIFAR10(device).to(device)
                print(f'The CapsNet-Recon for SVHN architecture is shown:\n {capnet_recon}')  # show architecture
                utils.train_capOutput(capnet_recon, train_loader, test_loader, args)
            else:
                capnet_recon = AlexNet_Module.CapsNet_Recon_CIFAR10(device).to(device)
                capnet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsNet-Recon for SVHN architecture is shown:\n {capnet_recon}')  # show architecture
                utils.train_capOutput(capnet_recon, train_loader, test_loader, args)

        # AlexCapsNet-recon
        if NETWORK == 5:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/SVHN/Alexnet/6.AlexCapsNet_Recon/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=1,
                            pretrained=False,
                            pretrained_weight='./Result/SVHN/Alexnet/6.AlexCapsNet_Recon/train/best_train/model_299.pth',
                            reconstruction=True
                            )
            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.SVHN(args.batch_size)), 10

            if not args.pretrained:
                alexcapnet_recon = AlexNet_Module.AlexCapsNet_Recon_CIFAR10(device).to(device)
                print(f'The AlexCapsNet-Recon for SVHN architecture is shown:\n {alexcapnet_recon}')  # show architecture
                utils.train_capOutput(alexcapnet_recon, train_loader, test_loader, args)
            else:
                alexcapnet_recon = AlexNet_Module.AlexCapsNet_Recon_CIFAR10(device).to(device)
                alexcapnet_recon.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The AlexCapsNet-Recon for SVHN architecture is shown:\n {alexcapnet_recon}')  # show architecture
                utils.train_capOutput(alexcapnet_recon, train_loader, test_loader, args)

        # Shallow AlexCapsNet
        if NETWORK == 6:
            # get parameter
            args = _arparse(epoch=300, lr=0.001, lr_decay=0.995, r=3, num_class=10, recon_alpha=0.0005,
                            train_save_dir='./Result/SVHN/Alexnet/6.S_AlexCapsNet/train',
                            train_eval=True,
                            save_all=True,
                            batch_size=50,
                            num_save_epoch=1,
                            pretrained=False,
                            pretrained_weight='./Result/SVHN/Alexnet/6.S_AlexCapsNet/train/best_train/model_299.pth',
                            reconstruction=False
                            )

            # get corresponding dataset
            (train_loader, test_loader), num_class = (dataloaders.SVHN(args.batch_size)), 10

            if not args.pretrained:
                capAlex = AlexNet_Module.S_AlexCapsNet_CIFAR10(device).to(device)
                print(f'The CapsAlexNet for SVHN architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)
            else:
                capAlex = AlexNet_Module.S_AlexCapsNet_CIFAR10(device).to(device)
                capAlex.load_state_dict(torch.load(args.pretrained_weight))
                print(f'The CapsAlexNet for SVHN architecture is shown:\n {capAlex}')  # show architecture
                utils.train_capOutput(capAlex, train_loader, test_loader, args)

if __name__ == '__main__':
    # CIFAR10: 1    CIFAR100: 2     MINIST：3    FashionMNIST: 4
    # FOOD101: 5    FLOWER102: 6    SVHN:   7
    DATASET = 7

    # AlexCapsNet: 1      AlexNet: 2                CapsNet:3
    # CapsNet_Recon: 4    AlexCapsNet_Recon: 5      S_AlexCapsNet: 6
    NETWORK = 2

    main(DATASET, NETWORK)





