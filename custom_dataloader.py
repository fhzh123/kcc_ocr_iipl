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

        label_file = open(label_path, 'r', encoding='utf-16')

        imgs = []
        labels = []
        while True:
            line = label_file.readline()
            if not line: break

            if int(line.split(' ')[0]) % 10000 == 0:
                print(line.split(' ')[0])

            img = cv2.imread(image_path + line.split(' ')[0] + '.jpg')
            if model.startswith('resnet'):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.asarray(img)
                img = np.reshape(img, (1, 48, 480))
            elif model == 'EfficientNet':
                img = np.asarray(img)
                img = np.reshape(img, (3, 48, 480))

            label = int(line.split(' ')[1].strip())
            imgs.append(img)
            labels.append(label)

        label_file.close()
        print("--- Data loaded. Loading time : %s seconds ---" % (time.time() - start_time))
        self.img = np.asarray(imgs)
        self.label = np.asarray(labels)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.img[idx])
        y = self.label[idx]
        return x, y
