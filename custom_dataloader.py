import torch
from PIL import Image
import os
import cv2
import time
import numpy as np
import torch
import random


from torch.utils.data import Dataset

class custom_dataloader(Dataset):
    def __init__(self, image_path, label_path, model):
        start_time = time.time()
        self.model_type = model
        label_file = open(label_path, 'r', encoding='utf-16')

        img_path = []
        labels = []
        while True:
            line = label_file.readline()
            if not line: break

            label = line.split(' ')[1].strip()
            img_path.append(line.split(' ')[0])
            labels.append(label)

        label_file.close()
        print("--- Data loaded. Loading time : %s seconds ---" % (time.time() - start_time))
        self.path = np.asarray(img_path)
        self.label = np.asarray(labels)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        img = cv2.imread(self.path[idx])
        if self.model_type.startswith('resnet'):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.asarray(img)
            img = np.reshape(img, (1, 48, 480))
        elif self.model_type == 'EfficientNet':
            img = np.asarray(img)
            img = np.reshape(img, (3, 48, 480))
        if len(self.label[idx]) == 0 :
            print("len = 1", [idx])
        return (torch.FloatTensor(img), self.label[idx])
