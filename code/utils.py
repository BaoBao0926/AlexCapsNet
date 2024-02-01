import math
import shutil
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import DataLoader
import torchvision
import time
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Primary CapsNet Layer
class PrimaryCaps(nn.Module):
    '''
    传入为256*20*20，输出为 [batch_size,1152,8]
    '''
    def __init__(self,num_caps=32, in_channel=256, out_channel=8, kersel_size=9, stride=2, padding=0):      # nc: number of capsule;
        super(PrimaryCaps, self).__init__()                     # ic: in_channel;   oc: out_inchannels
        self.num_caps = num_caps                                            # ks: kernel_size;  stride: stride
        self.capsules = nn.ModuleList([                         # [in_chanel, 长，宽] -> [nc, ic, 处理过的长，宽] —>
            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=kersel_size,
                      stride=stride,
                      padding=padding)
            for i in range(num_caps)
        ])

    def forward(self,x):
        '''
        :param x:  input image
        :return:

        [,256,20,20]->[8,6,6]*32 = [32,8,6,6] -> [1152, 8]
        '''
        batch_size = x.size(0)

        u = []
        for i in range(self.num_caps):
            u_i = self.capsules[i](x)           # [batch_size,8,6,6]
            u_i = u_i.view(batch_size, 8, -1, 1)   # [batch_size, 8,36,1]* 32
            u.append(u_i)

        # 把u里面的合并一下
        u = torch.cat(u, dim=3)          # [batch_size, 8, 36, 32] 把胶囊放在了最后一层
        u = u.view(batch_size, 8, -1)   # [batch_size, 8, 36*32=1152]
        u = u.transpose(1, 2)            # [batch_size, 1152, 8]

        u_squashed = self.squash(u)

        return u_squashed

    def squash(self,u):
        '''
        Args:
            `u`: [batch_size, 1152, 8]

        Return:
            `u_squashed`: [batch_size, 1152, 8]
        v_j = (norm(s_j) ^ 2 / (1 + norm(s_j) ^ 2)) * (s_j / norm(s_j))
        '''
        batch_size = u.size(0)

        square = u ** 2                             # [batch_size, 1152, 8]
        square_sum = torch.sum(square,dim = 2)      # [batch_size, 1152]
        norm = torch.sqrt(square_sum)               # [batch_size, 1152]
        factor = norm**2 / (norm * (1 + norm**2))   # [batch_size, 1152]
        u_squashed = factor.unsqueeze(2)            # [batch_size, 1152, 1]
        u_squashed = u_squashed * u                 # [batch_size, 1152, 8]
        return u_squashed

# Dense CapsNet Layer
class DenseCapsule(nn.Module):
    """
    The dense capsule layer. It is similar to Dense (FC) layer. Dense layer has `in_num` inputs, each is a scalar, the
    output of the neuron from the former layer, and it has `out_num` output neurons. DenseCapsule just expands the
    output of the neuron from scalar to vector. So its input size = [None, in_num_caps, in_dim_caps] and output size = \
    [None, out_num_caps, out_dim_caps]. For Dense Layer, in_dim_caps = out_dim_caps = 1.

    :param in_num_caps: number of cpasules inputted to this layer 有多少个胶囊被扔进来，这里为32*6*6
    :param in_dim_caps: dimension of input capsules 一个胶囊的channel，这里为8
    :param out_num_caps: number of capsules outputted from this layer 输出多少个胶囊 ，有多少个class就是多少个胶囊，这里为10
    :param out_dim_caps: dimension of output capsules   输出的一个胶囊的维度，这里为16
    :param routings: number of iterations for the routing algorithm，routine多少次
    """
    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, device, routings=3):
        super(DenseCapsule, self).__init__()
        self.in_num_caps = in_num_caps # 32*6*6
        self.in_dim_caps = in_dim_caps # 8
        self.out_num_caps = out_num_caps # 10
        self.out_dim_caps = out_dim_caps # 16
        self.routings = routings # 3
        self.device = device
        self.weight = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))

    def forward(self, x):
        # x.size=[batch, in_num_caps, in_dim_caps] [batch, 32*6*6, 8]
        # expanded to    [batch, 1,  in_num_caps, in_dim_caps,  1]=[batch,1,32*6*6,8,1]
        # weight.size   =[out_num_caps, in_num_caps, out_dim_caps, in_dim_caps]=[10,32*6*6,16,8]
        # torch.matmul: [out_dim_caps, in_dim_caps] x [in_dim_caps, 1] -> [out_dim_caps, 1]
        # => x_hat.size =[batch, out_num_caps, in_num_caps, out_dim_caps]
        x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1)

        # In forward pass, `x_hat_detached` = `x_hat`;
        # In backward, no gradient can flow from `x_hat_detached` back to `x_hat`.
        # x_hat_detached = x_hat.detach()
        x_hat_detached = x_hat
        # The prior for coupling coefficient, initialized as zeros.
        # b.size = [batch, out_num_caps, in_num_caps]
        b = torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps).to(self.device)

        assert self.routings > 0, 'The \'routings\' should be > 0.'
        for i in range(self.routings):
            # c.size = [batch, out_num_caps, in_num_caps]
            c = F.softmax(b, dim=1)

            # At last iteration, use `x_hat` to compute `outputs` in order to backpropagate gradient
            if i == self.routings - 1:
                # c.size expanded to [batch, out_num_caps, in_num_caps, 1           ]
                # x_hat.size     =   [batch, out_num_caps, in_num_caps, out_dim_caps]
                # => outputs.size=   [batch, out_num_caps, 1,           out_dim_caps]
                outputs = self.squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat))  # alternative way
            else:  # Otherwise, use `x_hat_detached` to update `b`. No gradients flow on this path.
                outputs = self.squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat_detached))  # alternative way

                # outputs.size       =[batch, out_num_caps, 1,           out_dim_caps]
                # x_hat_detached.size=[batch, out_num_caps, in_num_caps, out_dim_caps]
                # => b.size          =[batch, out_num_caps, in_num_caps]
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)

        return torch.squeeze(outputs, dim=-2)

    def squash(self, inputs, axis=-1):
        """
        The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
        :param inputs: vectors to be squashed
        :param axis: the axis to squash
        :return: a Tensor with same size as inputs
        """
        norm = torch.norm(inputs, p=2, dim=axis, keepdim=True) # p=2表示计算L2范数即向量的长度
        scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
        return scale * inputs

# CapsNet Reconstruction layer
class ReconstructionNet(nn.Module):
    def __init__(self, num_dim, num_caps, img_size, original_chanel):
        super(ReconstructionNet, self).__init__()
        self.fc1 = nn.Linear(num_dim * num_caps, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, img_size*img_size*original_chanel)
        self.num_dim = num_dim
        self.num_caps = num_caps
        self.img_size = img_size

    def forward(self, x, targets):  # 输入的x是 [bs, 10, 16]
        # only use correct caps to reconstruct
        batch_size = x.size(0)
        one_hot_target = torch.nn.functional.one_hot(targets, num_classes=self.num_caps)    # one_hot [bs, 10]
        one_hot_target = one_hot_target.unsqueeze(2)
        mask = torch.cat([one_hot_target for _ in range(16)], dim=2).to(device)    # mask: [batch_size, 10, 16])
        x = x * mask                                                               # [bs, 10, 16]

        x = x.view(batch_size, -1)    # [bs, 160]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# change capsnet output v into norm
def vtoNorm(v):
    batch_size = v.size(0)
    # process caps v get norm(v)
    square = v ** 2
    square_sum = torch.sum(square, dim=2)  # [batch_size, 10]
    norm = torch.sqrt(square_sum)  # [batch_size,10]      target tensor([8, 0])
    return norm

# CapsNet marginal loss function
def caps_marginal_loss(norm, target, num_class,l=0.5):
    # process target into one-hot target
    one_hot_target = F.one_hot(target, num_class)

    # calculate Loss
    L_correct = one_hot_target * torch.clamp(0.9 - norm, min=0.) ** 2
    L_wrong = (1 - one_hot_target) * torch.clamp(norm - 0.1, min=0.) ** 2
    L = L_correct + l * L_wrong
    L_margin = L.sum(dim=1).mean()

    return L_margin

def caps_reconstruction_loss(reconstruction, img):
    batch_size = img.size(0)
    image = img.view(batch_size, -1)
    # Scalar Variable
    reconstruction_loss = torch.sum((reconstruction - image) ** 2)
    return reconstruction_loss

# return correct number
def caps_accuracy(norm, target, batch_size):
    index = torch.argmax(norm, dim=1)                            # index tensor([0, 0])
    correct_num = 0
    for i in range(norm.size(0)):
        correct_num = correct_num + 1 if index[i] == target[i] else correct_num
        # print(f'True or false {index[i] == target[i]}, the correct num is {correct_num}')
    return correct_num


# get confusion_matrix
def get_confusion_matrix(norm, targets, num_class, confusion_matrix):
    batch_size = norm.shape[0]
    index = torch.argmax(norm, dim=1)   # pre index

    norm = torch.zeros_like(norm)
    for i in range(batch_size):
        norm[i][index[i]] = 1
    tar = torch.zeros_like(norm)
    for i in range(batch_size):
        tar[i][targets[i]] = 1

    for i in range(batch_size):   # 遍历每一个batch_size
        target = tar[i]
        prediction = norm[i]
        # 遍历每个类别
        for class_label in range(num_class):
            if target[class_label] == 1 and prediction[class_label] == 1:
                confusion_matrix[class_label]['TP'] += 1
            elif target[class_label] == 0 and prediction[class_label] == 0:
                confusion_matrix[class_label]['TN'] += 1
            elif target[class_label] == 0 and prediction[class_label] == 1:
                confusion_matrix[class_label]['FN'] += 1
            elif target[class_label] == 1 and prediction[class_label] == 0:
                confusion_matrix[class_label]['FP'] += 1

    return confusion_matrix

# create file
def create_numbered_folder(base_path, mode='eval'):
    """
    Create a numbered folder in the given base path.
    If folders with names like 'eval 1', 'eval 2', etc. exist, it generates the next available number.
    """
    i = 1
    while True:
        folder_name = mode + str(i)
        folder_path = base_path + '/' + folder_name
        # folder_path = os.path.join(base_path, folder_name)

        if not os.path.exists(folder_path):
            # The folder with the current number does not exist, create it
            os.makedirs(folder_path)
            print(f"Created folder '{folder_name}' at {base_path}")
            break
        else:
            # Increment to the next number and check again
            i += 1
    return folder_path

# save best weight pth file when evaluting in trainning, macro value
def save_best_pth_file(model, folder_path, epoch, m, accuracy, best_accuracy, loss, best_loss,
                       precision, best_precision, recall, best_recall, F1, best_F1):

    if accuracy >= best_accuracy:
        torch.save(model.state_dict(), folder_path+'/best_weight/best_accuracy.pth')
        best_accuracy = accuracy
        m['best accuracy'] = epoch
    if loss <= best_loss:
        best_loss = loss
        torch.save(model.state_dict(), folder_path+'/best_weight/best_loss.pth')
        m['best loss'] = epoch
    if precision >= best_precision:
        best_precision = precision
        torch.save(model.state_dict(), folder_path+'/best_weight/best_precision.pth')
        m['best precision'] = epoch
    if recall >= best_recall:
        best_recall = recall
        torch.save(model.state_dict(), folder_path+'/best_weight/best_recall.pth')
        m['best recall'] = epoch
    if F1 >= best_F1:
        best_F1 = F1
        torch.save(model.state_dict(), folder_path+'/best_weight/best_F1.pth')
        m['best F1'] = epoch

    with open(folder_path+'/best_weight/best_weight.txt', 'a') as f:
        f.write(f'best accuracy: {best_accuracy}. \tThe epoch: {m["best accuracy"]}\n')
        f.write(f'best precision: {best_precision}. \tThe epoch: {m["best precision"]}\n')
        f.write(f'best recall: {best_recall}. \tThe epoch: {m["best recall"]}\n')
        f.write(f'best F1: {best_F1}. \tThe epoch: {m["best F1"]}\n')
        f.write(f'best loss: {best_loss}. \tThe epoch: {m["best loss"]}\n\n\n')

    return best_accuracy, best_loss, best_precision, best_recall, best_F1, m

# enter a summarywriter, epoch,accuracy, loss, precision, recall and F1 to draw the picture
def draw_in_tersorboard(writer, epoch, accuracy, loss, precision, recall, F1):
    writer.add_scalar("accuracy-epoch", accuracy,  epoch)
    writer.add_scalar('loss-epoch', loss, epoch)
    writer.add_scalar('precision-epoch', precision, epoch)
    writer.add_scalar('recall-epoch', recall, epoch)
    writer.add_scalar('F1 score-epoch', F1, epoch)



# training when capsnet as last layers
def train_capOutput(model, train_loader, test_loader, args):
    print('----------------------train--------------------------')
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)     # optimizer
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)   # learning rate decay

    # create file-----------------------------------------
    folder_path = create_numbered_folder(args.train_save_dir, 'train')
    # create a folder to save these best weights
    os.makedirs(folder_path + '/best_weight', exist_ok=True)
    train_file_path = folder_path + '/train_result.txt'
    eval_file_path = folder_path + '/eval_result.txt'
    best_weight_path = folder_path + '/best_weight/best_weight.txt'
    f = open(train_file_path, 'w')
    f.close()
    f = open(eval_file_path, 'w')
    f.close()
    f = open(best_weight_path, 'w')
    f.close()

    # variable to record evalute value---------------------------------------
    best_accuracy, best_loss, best_recall, best_precision, best_F1 = 0, 1000000, 0, 0, 0
    # record epoch
    m = {'best accuracy': None, 'best loss': None, 'best recall': None, 'best precision': None, 'best F1': None}
    # tensorboard writer
    tensorboard_log_path = create_numbered_folder(folder_path, mode='tensorboard_logs')
    writer = SummaryWriter(tensorboard_log_path)
    # start training------------------------------------------------------
    for epoch in range(args.epochs):   # epoch
        print(f"------this is epoch {epoch}")
        model.train()  # set to training mode

        training_loss = 0.0
        print(f"Epoch {epoch + 1}, Current Learning Rate: {optimizer.param_groups[0]['lr']}")

        if epoch != 0:
            lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`

        epoch_start_t = time.time()     # record trainning start time
        for batch_index, (imgs, targets) in enumerate(train_loader):
            t1 = time.time()
            imgs, targets = imgs.to(device), targets.to(device)     # GPU or CPU trainning
            if args.reconstruction:
                v, reconstruction = model(imgs, targets)
                norm = vtoNorm(v)
                # compute loss
                reconstruction_loss = caps_reconstruction_loss(reconstruction, imgs)
                margin_loss = caps_marginal_loss(norm, targets, args.num_class).to(device)
                loss = margin_loss + args.reconstruction_alpha*reconstruction_loss
            else:
                v = model(imgs)
                norm = vtoNorm(v)
                # compute loss
                margin_loss = caps_marginal_loss(norm, targets, args.num_class).to(device)
                loss = margin_loss
            training_loss = training_loss + loss


            optimizer.zero_grad()   # set gradients of optimizer to zero
            loss.backward()         # backward, compute all gradients of loss w.r.t all Variables
            optimizer.step()        # update the trainable parameters with computed gradients
            t2 = time.time()
            if batch_index % 50 == 0:
                print(f"this is epoch {epoch} batch {batch_index} This batch loss: {loss}, batch trainning time {t2-t1}")

        epoch_end_t = time.time()       # record training end time

        llearing_rate = optimizer.param_groups[0]['lr']
        print(f'The epoch {epoch} total loss is {training_loss}, learing rate is {llearing_rate}(next epoch lr is {llearing_rate*args.lr_decay})')                   # print loss
        print(f'The epoch {epoch} running time is {epoch_end_t - epoch_start_t}')   # print running time

        # record information into txt file in train_result.txt
        with open(train_file_path, 'a') as f:
            f.write(f'Epoch {epoch}:\nTotal Loss: {training_loss}\nRunning time: {epoch_end_t - epoch_start_t}\n')
            f.write(f'learning rate is {llearing_rate}\nnext epoch lr is {llearing_rate*args.lr_decay}, lr_dacay is {args.lr_decay}\n\n\n')

        # eval and save---------------------------------------------------------------------------------
        if args.train_eval:  # 是否要一个epoch预测一次
            accuracy, loss, precision, recall, F1 = test_capOutput(model, test_loader, args)

            # record eval result in eval_result.txt
            with open(eval_file_path, 'a') as f:
                f.write(f'the epoch: {epoch}\nThe accuracy: {accuracy}\nThe loss: {loss}\n')
                f.write(f'The precision: {precision}\nThe recall: {recall}\nThe F1: {F1}\n\n\n')

            # save the best metric and in best_weight.txt record the which epoch
            best_accuracy, best_loss, best_precision, best_recall, best_F1, m = \
                save_best_pth_file(model, folder_path, epoch, m, accuracy, best_accuracy,
                                          loss, best_loss,  precision, best_precision, recall, best_recall, F1, best_F1)

            if args.save_all:       # whether save all pth file
                if (epoch + 1) % args.num_save_epoch == 0:
                    torch.save(model.state_dict(), folder_path + f'/model_{str(epoch).zfill(3)}.pth')
                    # torch.save(model, f'./Alxnet/module_{epoch}.pth')
                    print(f'module_{str(epoch).zfill(3)}.pth has been saved')

            # draw picture in tensorboard---
            draw_in_tersorboard(writer, epoch, accuracy, loss, precision, recall, F1)

        else:  # 不eval，直接保存按照规定的model权重文件
            # save the model
            if (epoch + 1) % args.num_save_epoch == 0:
                torch.save(model.state_dict(), folder_path + f'/model_{str(epoch).zfill(3)}.pth')
                # torch.save(model, f'./Alxnet/module_{epoch}.pth')
                print(f'module_{str(epoch).zfill(3)}.pth has been saved')

        writer.close()

# test when capsnet as last layers
def test_capOutput(model, test_loader, args):
    print(f'----------------------eval--------------------------')
    # get gpu,or cpu
    start_time = time.time()
    model = model.to(device)

    total_image_number = len(test_loader.dataset)
    model.eval()
    total_loss = 0
    correct_num = 0
    with torch.no_grad():
        # create confusion first
        conmatrix = {}
        inner_matrix = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        for i in range(args.num_class):
            conmatrix[i] = copy.deepcopy(inner_matrix)

        for i, (image, targets) in enumerate(test_loader):
            image, targets = image.to(device), targets.to(device)
            if args.reconstruction:                     # with reconstruction
                output, reconstruction = model(image, targets)   # predict the output
                norm = vtoNorm(output)                  # [bs,num of cap, cap dim]->[bs, num of cap]
                # calculate loss
                marginal_loss = caps_marginal_loss(norm, targets, args.num_class).item()        # marginal loss
                reconstruction_loss = caps_reconstruction_loss(reconstruction, image) # reconstruction loss
                total_loss = total_loss + marginal_loss + args.reconstruction_alpha * reconstruction_loss
            else:
                output = model(image)   # predict output
                norm = vtoNorm(output)  # [bs,num of cap, cap dim]->[bs, num of cap]
                # calculate loss
                total_loss = total_loss + caps_marginal_loss(norm, targets, args.num_class).item()  # get total loss

            # get correct number
            correct_num += caps_accuracy(norm, targets, args.batch_size)
            # get confusion matrix
            conmatrix = get_confusion_matrix(norm, targets, args.num_class, conmatrix)
            # show the evaluting process
            if i % 50 == 0:
                print(f'{i} correct number is {correct_num}, accuracy {correct_num/total_image_number}')
                print(f'{i} confusion matrix is {conmatrix}')
    # calculate accuracy
    accuracy = correct_num/total_image_number
    # calculate macro recall, precision and F1 score
    temp_precision, temp_recall = 0.0, 0.0
    for i in range(args.num_class):
        TP, TN, FP, FN = conmatrix[i]['TP'], conmatrix[i]['TN'], conmatrix[i]['FP'], conmatrix[i]['FN']
        temp_precision += TP / (TP + FP + 1e-6)
        temp_recall += TP / (TP + FN + 1e-6)
    precision, recall = temp_precision/args.num_class, temp_recall/args.num_class
    F1 = 2*precision*recall/(precision + recall + 1e-6)
    end_time = time.time()

    print(f'total image num is {total_image_number}')
    print(f'final confusion matrix is {conmatrix} correct final num is {correct_num} ')
    print(f'accuracy is {accuracy}')
    print(f'precision: {precision}, recall {recall}, F1 score {F1}')
    print(f'evaluating time {end_time-start_time}')

    return accuracy, total_loss, precision, recall, F1

# fully connected output
def train(model, train_loader, test_loader, args):
    # get gpu,or cpu
    optimizer = Adam(model.parameters(), lr=args.lr)     # optimizer
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)   # learning rate decay
    loss_function = nn.CrossEntropyLoss().to(device)

    # create file
    folder_path = create_numbered_folder(args.train_save_dir, 'train')
    # create a folder to save these best weights
    os.makedirs(folder_path + '/best_weight', exist_ok=True)
    train_file_path = folder_path + '/train_result.txt'
    eval_file_path = folder_path + '/eval_result.txt'
    best_weight_path = folder_path + '/best_weight/best_weight.txt'
    f = open(train_file_path, 'w')
    f.close()
    f = open(eval_file_path, 'w')
    f.close()
    f = open(best_weight_path, 'w')
    f.close()

    # variable to record evalute value---------------------------------------
    best_accuracy, best_loss, best_recall, best_precision, best_F1 = 0, 1000000, 0, 0, 0
    # record epoch
    m = {'best accuracy': None, 'best loss': None, 'best recall': None, 'best precision': None, 'best F1': None}
    # tensorboard writer
    tensorboard_log_path = create_numbered_folder(folder_path, mode='tensorboard_logs')
    writer = SummaryWriter(tensorboard_log_path)
    # start training------------------------------------------------------
    for epoch in range(args.epochs):   # epoch
        print(f"------this is epoch {epoch}")
        model.train()  # set to training mode
        if epoch != 0:
            lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`
        training_loss = 0.0
        print(f"Epoch {epoch + 1}, Current Learning Rate: {optimizer.param_groups[0]['lr']}")
        epoch_start_t = time.time()     # record trainning start time
        for batch_index, (imgs, targets) in enumerate(train_loader):
            imgs, targets = imgs.to(device), targets.to(device)

            if args.reconstruction:
                outputs, reconstruction = model(imgs, targets)
                # compute loss
                reconstruction_loss = caps_reconstruction_loss(reconstruction, imgs)
                temp_loss = loss_function(outputs, targets)
                loss = temp_loss + args.reconstruction_alpha*reconstruction_loss
            else:
                outputs = model(imgs)
                # compute loss
                temp_loss = loss_function(outputs, targets)
                loss = temp_loss

            training_loss += loss
            optimizer.zero_grad()   # set gradients of optimizer to zero
            loss.backward()         # backward, compute all gradients of loss w.r.t all Variables
            optimizer.step()        # update the trainable parameters with computed gradients

            if batch_index % 50 == 0:
                print(f"this is epoch {epoch} batch {batch_index} This batch loss: {loss}")

        epoch_end_t = time.time()       # record training end time

        llearing_rate  = optimizer.param_groups[0]['lr']
        print(f'The epoch {epoch} total loss is {training_loss},learing rate is {llearing_rate}(next epoch lr is {llearing_rate * args.lr_decay})')
        print(f'The epoch {epoch} running time is {epoch_end_t - epoch_start_t}')   # print running time

        # record information into txt file
        with open(train_file_path, 'a') as f:
            f.write(f'\nEpoch {epoch}:\nTotal Loss: {training_loss}\nRunning time: {epoch_end_t - epoch_start_t}\n')
            f.write(f'learing rate is {llearing_rate}\nnext epoch lr is {llearing_rate*args.lr_decay},lr_dacay is {args.lr_decay}\n')

        # eval and save---------------------------------------------------------------------------------
        if args.train_eval:  # 是否要一个epoch预测一次
            accuracy, loss, precision, recall, F1 = test(model, test_loader, args)

            # record eval result in eval_result.txt
            with open(eval_file_path, 'a') as f:
                f.write(f'the epoch: {epoch}\nThe accuracy: {accuracy}\nThe loss: {best_loss}\n')
                f.write(f'The precision: {precision}\nThe recall: {recall}\nThe F1: {F1}\n\n\n')

            # save the best metric and in best_weight.txt record the which epoch
            best_accuracy, best_loss, best_precision, best_recall, best_F1, m \
                = save_best_pth_file(model, folder_path, epoch, m, accuracy, best_accuracy,
                                     loss, best_loss, precision, best_precision, recall, best_recall, F1, best_F1)

            if args.save_all:  # whether save all pth file
                if (epoch + 1) % args.num_save_epoch == 0:
                    torch.save(model.state_dict(), folder_path + f'/model_{str(epoch).zfill(3)}.pth')
                    # torch.save(model, f'./Alxnet/module_{epoch}.pth')
                    print(f'module_{str(epoch).zfill(3)}.pth has been saved')

            # draw picture in tensorboard---
            draw_in_tersorboard(writer, epoch, accuracy, loss, precision, recall, F1)

        else:  # 不eval，直接保存按照规定的model权重文件
            # save the model
            if (epoch + 1) % args.num_save_epoch == 0:
                torch.save(model.state_dict(), folder_path + f'/model_{str(epoch).zfill(3)}.pth')
                # torch.save(model, f'./Alxnet/module_{epoch}.pth')
                print(f'module_{str(epoch).zfill(3)}.pth has been saved')

        writer.close()

# fully connected output
def test(model, test_loader, args):
    # get gpu,or cpu
    print('-----------------------eval-----------------------')
    start_time = time.time()
    model = model.to(device)

    total_image_number = len(test_loader.dataset)
    loss_function = nn.CrossEntropyLoss().to(device)
    model.eval()
    total_loss = 0
    correct_num = 0
    with torch.no_grad():
        # create confusion first
        conmatrix = {}
        inner_matrix = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        for i in range(args.num_class):
            conmatrix[i] = copy.deepcopy(inner_matrix)


        for i, (image, targets) in enumerate(test_loader):
            image, targets = image.to(device), targets.to(device)

            if args.reconstruction:                     # with reconstruction
                outputs, reconstruction = model(image, targets)   # predict the output
                # calculate loss
                temp_loss = loss_function(outputs, targets).item()        # marginal loss
                reconstruction_loss = caps_reconstruction_loss(reconstruction, image) # reconstruction loss
                total_loss = total_loss + temp_loss + args.reconstruction_alpha * reconstruction_loss
            else:
                outputs = model(image)   # predict output
                # calculate loss
                total_loss = total_loss + loss_function(outputs, targets).item()  # get total loss
            # get accuracy
            index = torch.argmax(outputs, dim=1)
            for j in range(outputs.size(0)):
                correct_num = correct_num + 1 if index[j] == targets[j] else correct_num

            # get confusion matrix, TP,TN,FP,FN
            conmatrix = get_confusion_matrix(outputs, targets, args.num_class, conmatrix)  # get confusion matrix

            if i % 50 == 0:
                print(f'batch {i}: correct number is {correct_num}, and the accuracy is {correct_num/total_image_number}')
                print(f'batch {i}: confusion matrix is {conmatrix}')

    accuracy = correct_num/total_image_number   # calculate accuracy
    # calculate macro recall, precision and F1 score
    temp_precision, temp_recall = 0.0, 0.0
    for i in range(args.num_class):
        TP, TN, FP, FN = conmatrix[i]['TP'], conmatrix[i]['TN'], conmatrix[i]['FP'], conmatrix[i]['FN']
        temp_precision += TP / (TP + FP + 1e-6)
        temp_recall += TP / (TP + FN + 1e-6)
    precision, recall = temp_precision / args.num_class, temp_recall / args.num_class
    F1 = 2 * precision * recall / (precision + recall + 1e-6)

    end_time = time.time()

    print(f'total image num is {total_image_number}, time {end_time-start_time}')
    print(f'accuracy is {accuracy},correct final num is {correct_num} ')
    print(f'final confusion matrix is {conmatrix}')
    print(f'presicion: {precision}, recall {recall}, F1 score {F1}')

    return accuracy, total_loss, precision, recall, F1



# Dataloader
class dataloaders():
    # CIFAR10
    def CIFAR10(self, batch_size):
        transform = transforms.Compose([
            transforms.transforms.RandomRotation(0.5),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = torchvision.datasets.CIFAR10(root='./datasets/CIFAR10', train=True,
                                                     download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)

        test_dataset = torchvision.datasets.CIFAR10(root='./datasets/CIFAR10', train=False,
                                                    download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)
        return train_loader, test_loader
    # CIFAR100
    def CIFAR100(self, batch_size):
        transform = transforms.Compose([
            transforms.transforms.RandomRotation(0.5),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = torchvision.datasets.CIFAR100(root='./datasets/CIFAR100', train=True,
                                                     download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)

        test_dataset = torchvision.datasets.CIFAR100(root='./datasets/CIFAR100', train=False,
                                                    download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)
        return train_loader, test_loader
    # MNIST
    def MNIST(self, batch_size):
        transform = transforms.Compose([
            transforms.transforms.RandomRotation(0.5),
            transforms.RandomGrayscale(),
            transforms.ToTensor()
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = torchvision.datasets.MNIST(root='./datasets/MNIST', train=True,
                                                     download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)

        test_dataset = torchvision.datasets.MNIST(root='./datasets/MNIST', train=False,
                                                    download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)
        return train_loader, test_loader
    # Fshion-MNIST
    def FashionMNIST(self, batch_size):
        transform = transforms.Compose([
            transforms.transforms.RandomRotation(0.5),
            transforms.RandomGrayscale(),
            transforms.ToTensor()
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor()
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = torchvision.datasets.FashionMNIST(root='./datasets/FashionMNIST', train=True,
                                                     download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)

        test_dataset = torchvision.datasets.FashionMNIST(root='./datasets/FashionMNIST', train=False,
                                                    download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)
        return train_loader, test_loader
    # FOOD101 512*512->224
    def FOOD101(self, batch_size):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.transforms.RandomRotation(0.5),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = torchvision.datasets.Food101(root='./datasets/FOOD101', split='train', download=True,
                                                     transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)

        test_dataset = torchvision.datasets.Food101(root='./datasets/FOOD101', split='test', download=True,
                                                     transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)
        return train_loader, test_loader
    # Kuzushiji-MNIST
    def KMNIST(self, batch_size):
        transform = transforms.Compose([
            transforms.transforms.RandomRotation(0.5),
            transforms.RandomGrayscale(),
            transforms.ToTensor()
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.KMNIST(root='./datasets/K_MNIST', train=True,
                                                     download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)

        test_dataset = torchvision.datasets.MNIST(root='./datasets/K_MNIST', train=False,
                                                    download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)
        return train_loader, test_loader
    # Flower102
    def FLOWER102(self, batch_size):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.transforms.RandomRotation(0.5),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            transform_test = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            train_dataset = torchvision.datasets.Flowers102(root='./datasets/FLOWER102', split='test', download=True,
                                                         transform=transform)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                       shuffle=True, num_workers=4)

            val_dataset = torchvision.datasets.Flowers102(root='./datasets/FLOWER102', split='val', download=True,
                                                        transform=transform_test)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                      shuffle=False, num_workers=4)
            return train_loader, val_loader
    # SVHN     torchvision.datasets.SVHN
    def SVHN(self, batch_size):
        transform = transforms.Compose([
            transforms.transforms.RandomRotation(0.5),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = torchvision.datasets.SVHN(root='./datasets/SVHN', split='train',
                                                     download=True, transform=transform)

        test_dataset = torchvision.datasets.SVHN(root='./datasets/SVHN', split='test',
                                                    download=True, transform=transform_test)


        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)

        return train_loader, test_loader

class dataloaders_visual():
    # CIFAR10
    def CIFAR10(self, batch_size):
        transform = transforms.Compose([
            transforms.ToTensor(),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = torchvision.datasets.CIFAR10(root='./datasets/CIFAR10', train=True,
                                                     download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)

        test_dataset = torchvision.datasets.CIFAR10(root='./datasets/CIFAR10', train=False,
                                                    download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)
        return train_loader, test_loader
    # CIFAR100
    def CIFAR100(self, batch_size):
        transform = transforms.Compose([
            transforms.ToTensor(),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = torchvision.datasets.CIFAR100(root='./datasets/CIFAR100', train=True,
                                                     download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)

        test_dataset = torchvision.datasets.CIFAR100(root='./datasets/CIFAR100', train=False,
                                                    download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)
        return train_loader, test_loader
    # MNIST
    def MNIST(self, batch_size):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = torchvision.datasets.MNIST(root='./datasets/MNIST', train=True,
                                                     download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)

        test_dataset = torchvision.datasets.MNIST(root='./datasets/MNIST', train=False,
                                                    download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)
        return train_loader, test_loader
    # Fshion-MNIST
    def FashionMNIST(self, batch_size):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor()
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = torchvision.datasets.FashionMNIST(root='./datasets/FashionMNIST', train=True,
                                                     download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)

        test_dataset = torchvision.datasets.FashionMNIST(root='./datasets/FashionMNIST', train=False,
                                                    download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)
        return train_loader, test_loader
    # FOOD101 512*512->224
    def FOOD101(self, batch_size):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),

        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = torchvision.datasets.Food101(root='./datasets/FOOD101', split='train', download=True,
                                                     transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)

        test_dataset = torchvision.datasets.Food101(root='./datasets/FOOD101', split='test', download=True,
                                                     transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)
        return train_loader, test_loader
    # Flower102
    def FLOWER102(self, batch_size):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

            transform_test = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            train_dataset = torchvision.datasets.Flowers102(root='./datasets/FLOWER102', split='test', download=True,
                                                         transform=transform)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                       shuffle=True, num_workers=4)

            val_dataset = torchvision.datasets.Flowers102(root='./datasets/FLOWER102', split='val', download=True,
                                                        transform=transform_test)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                      shuffle=False, num_workers=4)
            return train_loader, val_loader
# Dataloader
class dataloaders_recon():
    # CIFAR10
    def CIFAR10(self, batch_size):
        transform = transforms.Compose([
            transforms.transforms.RandomRotation(0.5),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = torchvision.datasets.CIFAR10(root='./datasets/CIFAR10', train=True,
                                                     download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)

        test_dataset1 = torchvision.datasets.CIFAR10(root='./datasets/CIFAR10', train=False,
                                                    download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset1, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)
        test_dataset2 = torchvision.datasets.CIFAR10(root='./datasets/CIFAR10', train=False,
                                                    download=True)
        test_loader2 = torch.utils.data.DataLoader(test_dataset2, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)
        return train_loader, test_loader, test_loader2
    # CIFAR100
    def CIFAR100(self, batch_size):
        transform = transforms.Compose([
            transforms.transforms.RandomRotation(0.5),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = torchvision.datasets.CIFAR100(root='./datasets/CIFAR100', train=True,
                                                     download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)

        test_dataset1 = torchvision.datasets.CIFAR100(root='./datasets/CIFAR100', train=False,
                                                    download=True, transform=transform_test)
        test_loader1 = torch.utils.data.DataLoader(test_dataset1, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)

        test_dataset2 = torchvision.datasets.CIFAR100(root='./datasets/CIFAR100', train=False,
                                                    download=True)
        test_loader2 = torch.utils.data.DataLoader(test_dataset2, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)

        return train_loader, test_loader1, test_loader2
    # MNIST
    def MNIST(self, batch_size):
        transform = transforms.Compose([
            transforms.transforms.RandomRotation(0.5),
            transforms.RandomGrayscale(),
            transforms.ToTensor()
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = torchvision.datasets.MNIST(root='./datasets/MNIST', train=True,
                                                     download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)

        test_dataset1 = torchvision.datasets.MNIST(root='./datasets/MNIST', train=False,
                                                    download=True, transform=transform_test)
        test_loader1 = torch.utils.data.DataLoader(test_dataset1, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)
        test_dataset2 = torchvision.datasets.MNIST(root='./datasets/MNIST', train=False,
                                                   download=True)
        test_loader2 = torch.utils.data.DataLoader(test_dataset2, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)

        return train_loader, test_loader1, test_loader2
    # Fshion-MNIST
    def FashionMNIST(self, batch_size):
        transform = transforms.Compose([
            transforms.transforms.RandomRotation(0.5),
            transforms.RandomGrayscale(),
            transforms.ToTensor()
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor()
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = torchvision.datasets.FashionMNIST(root='./datasets/FashionMNIST', train=True,
                                                     download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)

        test_dataset1 = torchvision.datasets.FashionMNIST(root='./datasets/FashionMNIST', train=False,
                                                    download=True, transform=transform_test)
        test_loader1 = torch.utils.data.DataLoader(test_dataset1, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)

        test_dataset2 = torchvision.datasets.FashionMNIST(root='./datasets/FashionMNIST', train=False,
                                                    download=True, transform=transform_test)
        test_loader2 = torch.utils.data.DataLoader(test_dataset2, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)
        return train_loader, test_loader1, test_loader2
    # FOOD101 512*512->224
    def FOOD101(self, batch_size):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.transforms.RandomRotation(0.5),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.Resize((224, 225)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform_test2 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        train_dataset = torchvision.datasets.Food101(root='./datasets/FOOD101', split='train', download=True,
                                                     transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)

        test_dataset1 = torchvision.datasets.Food101(root='./datasets/FOOD101', split='test', download=True,
                                                     transform=transform_test)
        test_loader1 = torch.utils.data.DataLoader(test_dataset1, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)

        test_dataset2 = torchvision.datasets.Food101(root='./datasets/FOOD101', split='test', download=True,
                                                     transform=transform_test2)
        test_loader2 = torch.utils.data.DataLoader(test_dataset2, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)

        return train_loader, test_loader1, test_loader2
    # Flower102
    def FLOWER102(self, batch_size):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.transforms.RandomRotation(0.5),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            transform_test = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            train_dataset = torchvision.datasets.Flowers102(root='./datasets/FLOWER102', split='test', download=True,
                                                         transform=transform)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                       shuffle=True, num_workers=4)

            val_dataset1 = torchvision.datasets.Flowers102(root='./datasets/FLOWER102', split='val', download=True,
                                                        transform=transform_test)
            val_loader1 = torch.utils.data.DataLoader(val_dataset1, batch_size=batch_size,
                                                      shuffle=False, num_workers=4)

            val_dataset2 = torchvision.datasets.Flowers102(root='./datasets/FLOWER102', split='val', download=True
                                                        )
            val_loader2 = torch.utils.data.DataLoader(val_dataset2, batch_size=batch_size,
                                                      shuffle=False, num_workers=4)

            return train_loader, val_loader1, val_loader2
    # SVHN     torchvision.datasets.SVHN
    def SVHN(self, batch_size):
        transform = transforms.Compose([
            transforms.transforms.RandomRotation(0.5),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transform_test1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transform_test2 = transforms.Compose([
            transforms.ToTensor(),
            ])

        train_dataset = torchvision.datasets.SVHN(root='./datasets/SVHN', split='train',
                                                     download=True, transform=transform)

        test_dataset1 = torchvision.datasets.SVHN(root='./datasets/SVHN', split='test',
                                                    download=True, transform=transform_test1)
        test_dataset2 = torchvision.datasets.SVHN(root='./datasets/SVHN', split='test',
                                                    download=True, transform=transform_test2)


        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=4)
        test_loader1 = torch.utils.data.DataLoader(test_dataset1, batch_size=batch_size,
                                                   shuffle=False, num_workers=4)
        test_loader2 = torch.utils.data.DataLoader(test_dataset2, batch_size=batch_size,
                                                  shuffle=False, num_workers=4)

        return train_loader, test_loader1, test_loader2

