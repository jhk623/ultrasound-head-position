import cv2, torch
import sys, os, argparse, time

import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import datasets, hopenet
import torch.utils.model_zoo as model_zoo
import math

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=5, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=16, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.001, type=float)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='Pose_300W_LP', type=str)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
          default='', type=str)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshots.', default = '', type=str)
    parser.add_argument('--alpha', dest='alpha', help='Regression loss coefficient.',
          default=0.001, type=float)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)

    args = parser.parse_args()
    return args

def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 1], 66)

    if args.snapshot == '':
        load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    else:
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict)

    print ('Loading data.')

    transformations = transforms.Compose([transforms.Scale(240),
    transforms.RandomCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if args.dataset == 'Train_AUG':
        pose_dataset = datasets.Train_AUG(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'Test_AUG':
        pose_dataset = datasets.Test_AUG(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'NotAUG':
        pose_dataset = datasets.NotAUG(args.data_dir, args.filename_list, transformations)
    else:
        print( 'Error: not a valid dataset name')
        sys.exit()

    train_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=8)

    model.cuda(gpu)
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    reg_criterion = nn.MSELoss().cuda(gpu)
    # Regression loss coefficient
    alpha = args.alpha

    softmax = nn.Softmax().cuda(gpu)
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)

    optimizer = torch.optim.Adam([{'params': get_ignored_params(model), 'lr': 0},
                                  {'params': get_non_ignored_params(model), 'lr': args.lr},
                                  {'params': get_fc_params(model), 'lr': args.lr * 5}],
                                   lr = args.lr)

    print('Ready to train network.')
    for epoch in range(num_epochs):
        for i, (images, labels, cont_labels, name, xy) in enumerate(train_loader):
            images = Variable(images).cuda(gpu)

            # Binned labels
            label_roll = Variable(labels[:,2]).cuda(gpu)

            # Continuous labels
            label_roll_cont = Variable(cont_labels[:,2]).cuda(gpu)

            # Forward pass
            roll = model(images)

            # Cross entropy loss
            loss_roll = criterion(roll, label_roll)

            # MSE loss
            roll_predicted = softmax(roll)
#            temptensor = softmax(roll)
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99
#            temptensor = torch.sum(temptensor * idx_tensor, 1) 
#            aaa = Variable(torch.ByteTensor([1])).cuda()
#            bbb = Variable(torch.FloatTensor([math.pi]).cuda())
#            ind = 0
#            for tmp in temptensor:
#                while torch.equal(torch.ge(tmp, 2*math.pi), aaa):
#                    tmp =  torch.add(tmp, -2, bbb)
#                while torch.equal(torch.le(tmp, -2*math.pi), aaa):
#                    tmp = torch.add(tmp, 2, bbb)
#                if torch.equal(torch.gt(tmp, math.pi), aaa):
#                    tmp =  torch.add(tmp, -2, bbb)
#                if torch.equal(torch.le(tmp, -math.pi), aaa):
#                    tmp =  torch.add(tmp, 2, bbb)

#                roll_predicted[ind] = tmp
#                ind +=1

            loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)

            # Total loss
            loss_roll += alpha * loss_reg_roll

            loss_seq = [loss_roll]
            grad_seq = [torch.Tensor(1).cuda(gpu) for _ in range(len(loss_seq))]
            optimizer.zero_grad()
            torch.autograd.backward(loss_seq, grad_seq)
            optimizer.step()

            if (i+1) % 100 == 0:
                 print ('Epoch [%d/%d], Iter [%d/%d] Losses: Roll %.4f'
                       %(epoch+1, num_epochs, i+1, len(pose_dataset)//batch_size, loss_roll.data[0]))

        # Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs:
            print('Taking snapshot...')
            torch.save(model.state_dict(),
            'output/snapshots/' + args.output_string + '_epoch_'+ str(epoch+1) + '.pkl')
