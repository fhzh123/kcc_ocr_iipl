import torch 
import torch.nn as nn
import torch.nn.init as init
from basenet.resnet import ResNet18, ResNet50, ResNet34
from basenet.EfficientNet import EfficientNet

class recognition_architecture(nn.Module):
    def __init__(self, cnn_model_type):
        super(recognition_architecture, self).__init__()
        if cnn_model_type == 'resnet18':
            self.cnn = ResNet18()
        elif cnn_model_type == 'resnet34':
            self.cnn = ResNet34()
        elif cnn_model_type == 'resnet50':
            self.cnn = ResNet50()
        elif cnn_model_type == 'EfficientNet':
            self.cnn = EfficientNet(1, 1)

        
    def forward(self, x):
        ''' (Batch, 3, 48, 480) '''
        features = self.cnn(x)
        ''' (Batch, 512, 3, 30) ''' 

        
        return features
