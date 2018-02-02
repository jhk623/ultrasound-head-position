import torch,cv2
import sys, os, argparse

import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F

import datasets, hopenet, utils
import math

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
          default='', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Name of model snapshot.',
          default='', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=1, type=int)
    parser.add_argument('--save_viz', dest='save_viz', help='Save images with pose cube.',
          default=False, type=bool)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='NotAUG', type=str)
    parser.add_argument('--save_text', dest='save_text', help='Save as text', default=False, type=bool)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    gpu = args.gpu_id
    snapshot_path = args.snapshot

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 1], 66)

    print('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    print( 'Loading data.')

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if args.dataset == 'Train_AUG':
        pose_dataset = datasets.AUG(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'Test_AUG':
        pose_dataset = datasets.Test_AUG(args.data_dir, args.filename_list, transformations)
    elif args.dataset == 'NotAUG':
        pose_dataset = datasets.NotAUG(args.data_dir, args.filename_list, transformations)
    else:
        print ('Error: not a valid dataset name')
        sys.exit()
    test_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=8)

    model.cuda(gpu)

    print ('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

#    yaw_error = .0
#    pitch_error = .0
    roll_error = .0

    l1loss = torch.nn.L1Loss(size_average=False)
#    llii = []

    for i, (images, labels, cont_labels, name, xy) in enumerate(test_loader):
        data_path = args.data_dir
        ffff = open(data_path + name[0] + ".txt", 'r')
        imnamelist = ffff.readline().split(" ")
        imname = imnamelist[0] 

        images = Variable(images).cuda(gpu)
        total += cont_labels.size(0)

#        label_yaw = cont_labels[:,0].float()
#        label_pitch = cont_labels[:,1].float()
        label_roll = cont_labels[:,2].float()

#        yaw, pitch, roll = model(images)
        roll = model(images)

        # Binned predictions
#        _, yaw_bpred = torch.max(yaw.data, 1)
#        _, pitch_bpred = torch.max(pitch.data, 1)
        _, roll_bpred = torch.max(roll.data, 1)
        
        # Continuous predictions
#        yaw_predicted = utils.softmax_temperature(yaw.data, 1)
#        pitch_predicted = utils.softmax_temperature(pitch.data, 1)
        roll_predicted = utils.softmax_temperature(roll.data, 1)
        temptensor = utils.softmax_temperature(roll.data, 1)
#        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
#        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
        roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99
#        llii.append(math.degrees(roll_predicted[0]))
#        temptensor = torch.sum(temptensor * idx_tensor, 1).cpu() * 3 - 99
#        tmp = temptensor[0]  
#        while tmp >= 2 * math.pi:
#            tmp =  tmp - 2 * math.pi
#        while tmp <= -2 * math.pi:
#            tmp = tmp + 2 * math.pi
#        if tmp > math.pi:
#            tmp = tmp - 2 * math.pi
#        if tmp <= -math.pi:
#            tmp = tmp + 2 * math.pi       
#        while tmp  <= -math.pi/4:
#            tmp += math.pi/2
#        roll_predicted[0] = tmp

#        print(roll_bpred,tmp)
        # or = softmax(roll)ean absolute error
#        yaw_error += torch.sum(torch.abs(yaw_predicted - label_yaw))
#        pitch_error += torch.sum(torch.abs(pitch_predicted - label_pitch))
        roll_error += torch.sum(torch.abs(roll_predicted - label_roll))

        if args.save_text:
            name = name[0]
            f = open('output/text/' + name + ".txt", 'w')
            f.write("%s\n" %imname)
#            f.write("%s\n" %aaabbb)
            f.write("%s\n" %roll_predicted[0])
            f.write("%d %d %d %d" %(int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])))
            f.close()

    print('Test error in radian of the model on the ' + str(total) + ' test images. Roll: %.4f' % (roll_error / total))
#    llii.sort()
#    print(llii)



