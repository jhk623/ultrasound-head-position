import os
import numpy as np
import cv2
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from PIL import Image, ImageFilter
import math
import utils

def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

class Train_AUG(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_path = ("/home/garcons/deep-head-pose/aug_img/train/numfile.txt")
        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)
        annot = open(txt_path, 'r')
        line = annot.readline().split(' ')
        img_name = line[0]
        
        img = Image.open("/home/garcons/deep-head-pose/aug_img/train/" + img_name)
        img = img.convert(self.image_mode)
        
        # We get the pose in degrees
        annot = open(txt_path,'r')
        line = annot.readline().split(' ')
        yaw, pitch, roll = [float(line[1]), float(line[2]), float(line[3])]
#        roll = math.radians(roll)
#        while roll >= 2 * math.pi:
#            roll -= 2 * math.pi
#        while roll <= -2 * math.pi:
#            roll += 2 * math.pi
#        if roll > math.pi:
#            roll = roll - 2 * math.pi
#        if roll <= -math.pi:
#            roll = roll + 2 * math.pi

        # Crop the face loosely
        k = 0.32
        x1 = float(line[4])
        y1 = float(line[5])
        x2 = float(line[6])
        x2 += x1
        y2 = float(line[7])
        y2 += y1
        xy = (x1,y1,x2,y2)

        x1 -= 0.8 * k * abs(x2 - x1)
        y1 -= 2 * k * abs(y2 - y1)
        x2 += 0.8 * k * abs(x2 - x1)
        y2 += 1 * k * abs(y2 - y1)
       
        img = img.crop((int(x1), int(y1), int(x2), int(y2)))
        # Bin values
        bins = np.array(range(-99, 102, 3))
#        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index], xy

    def __len__(self):
        return self.length

class Test_AUG(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        
        filename_path = ("/home/garcons/deep-head-pose/aug_img/test/numfile.txt")
        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)
        annot = open(txt_path, 'r')
        line = annot.readline().split(' ')
        img_name = line[0]

        img = Image.open("/home/garcons/deep-head-pose/aug_img/test/" + img_name)
        img = img.convert(self.image_mode)

        # We get the pose in degrees
        annot = open(txt_path,'r')
        line = annot.readline().split(' ')
        yaw, pitch, roll = [float(line[1]), float(line[2]), float(line[3])]
#        roll = math.radians(roll)
#        while roll >= 2 * math.pi:
#            roll -= 2 * math.pi
#        while roll <= -2 * math.pi:
#            roll += 2 * math.pi
#        if roll > math.pi:
#            roll = roll - 2 * math.pi
#        if roll <= -math.pi:
#            roll = roll + 2 * math.pi

        # Crop the face loosely
        k = 0.32
        x1 = float(line[4])
        y1 = float(line[5])
        x2 = float(line[6])
        x2 += x1
        y2 = float(line[7])
        y2 += y1
        xy = (x1,y1,x2,y2)

        x1 -= 0.8 * k * abs(x2 - x1)
        y1 -= 2 * k * abs(y2 - y1)
        x2 += 0.8 * k * abs(x2 - x1)
        y2 += 1 * k * abs(y2 - y1)

        img = img.crop((int(x1), int(y1), int(x2), int(y2)))
        # Bin values
        bins = np.array(range(-99, 102, 3))
#        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index], xy

    def __len__(self):
        return self.length



class NotAUG(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_path = ("/home/garcons/deep-head-pose/test.txt")
        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)
        annot = open(txt_path, 'r')
        line = annot.readline().split(' ')
        img_name = line[0]

        img = Image.open("/home/garcons/deep-head-pose/JPEGImages/" + img_name)
        img = img.convert(self.image_mode)

        # We get the pose in degrees
        annot = open(txt_path,'r')
        line = annot.readline().split(' ')
        yaw, pitch, roll = [float(line[1]), float(line[2]), float(line[3])]
#        roll = math.radians(roll)
#        while roll >= 2 * math.pi:
#            roll -= 2 * math.pi
#        while roll <= -2 * math.pi:
#            roll += 2 * math.pi
#        if roll > math.pi:
#            roll = roll - 2 * math.pi
#        if roll <= -math.pi:
#            roll = roll + 2 * math.pi

        # Crop the face loosely
        k = 0.32
        x1 = float(line[4])
        y1 = float(line[5])
        x2 = float(line[6])
        x2 += x1
        y2 = float(line[7])
        y2 += y1
        xy = (x1,y1,x2,y2)

        x1 -= 0.8 * k * abs(x2 - x1)
        y1 -= 2 * k * abs(y2 - y1)
        x2 += 0.8 * k * abs(x2 - x1)
        y2 += 1 * k * abs(y2 - y1)

        img = img.crop((int(x1), int(y1), int(x2), int(y2)))
        # Bin values
        bins = np.array(range(-99, 102, 3))
#        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index], xy

    def __len__(self):
        return self.length

