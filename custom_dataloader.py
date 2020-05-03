import torch
from PIL import Image
from torch.utils.data import sampler

import os
import cv2
import time
import numpy as np
import torch
import random


from torch.utils.data import Dataset

class custom_dataloader(Dataset):
    def __init__(self, label_path, model):
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
        img = np.reshape(img, (3, 480, 48))
        return (img, self.label[idx])


class alignCollate(object):
    def __init__(self, imgH=480, imgW=48, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)
        images = torch.cat([t.unsqueeze(0) for t in images], 0)
        print(labels)
        return images, labels

class randomSequentialSampler(sampler.Sampler):
    
    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples