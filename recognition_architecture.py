import torch 
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from basenet.resnet import ResNet18, ResNet50, ResNet34
from basenet.EfficientNet import EfficientNet
from basenet.rnn_bn import Attention, AttentionCell, BidirectionalLSTM

class recognition_architecture(nn.Module):
    def __init__(self, cnn_model_type, rnn_type, num_char = 0, num_hidden = 0, max_len = 10):
        super(recognition_architecture, self).__init__()
        self.max_len = max_len # Maximum Length

        ''' Feature Extraction '''
        if cnn_model_type == 'resnet18':
            self.cnn = ResNet18()
        elif cnn_model_type == 'resnet34':
            self.cnn = ResNet34()
        elif cnn_model_type == 'resnet50':
            self.cnn = ResNet50()
        elif cnn_model_type == 'EfficientNet':
            self.cnn = EfficientNet(1, 1)

        self.cnn_output = 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        ''' Sequence & Prediction'''
        self.rnn_type = rnn_type
        
        if rnn_type == 'rnn':
            self.rnn = nn.Sequential(
            BidirectionalLSTM(512, num_hidden, num_hidden),
            BidirectionalLSTM(num_hidden, num_hidden, num_char))
        elif rnn_type == 'attention':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.cnn_output, num_hidden, num_hidden),
                BidirectionalLSTM(num_hidden, num_hidden, num_hidden))
            self.SequenceModeling_output = num_hidden
            self.Prediction = Attention(self.SequenceModeling_output, num_hidden, num_char)
        elif rnn_type == 'transformer':
            # Transformer model setting
            # Need to fix Hyperparameter & vocab_num & etc
            self.transformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, 
                                            num_decoder_layers=6, dim_feedforward=2048, 
                                            dropout=0.1, activation='gelu')
            self.embedding = nn.Embedding(num_char, 512)
            self.output_linear = nn.Linear(512, num_char, bias=False)
            self.softmax = nn.LogSoftmax(2)

            # self.bos_idx = bos_idx # Start token of sentence
            # self.eos_idx = eos_idx # End token of sentece
            self.pad_idx = 0 # Padding token of sentece

        
    def forward(self, x, trg = None, is_train=True):
        ''' (Batch, 3, 480, 48) '''
        features = self.cnn(x)
        ''' (Batch, 512, 10, 1) ''' 

        # rnn features
        if self.rnn_type =='rnn':
            b, c, h, w = features.size()
            features = features.squeeze(3)
            features = features.permute(2, 0, 1)  # [h, b, c]
            output = self.rnn(features)
        elif self.rnn_type == 'attention':     
            visual_feature = self.AdaptiveAvgPool(features.permute(0, 2, 1, 3))  # [b, c, h, w] -> [b, h, c, w]
            visual_feature = visual_feature.squeeze(3)

            contextual_feature = self.SequenceModeling(visual_feature)
            output = self.Prediction(contextual_feature.contiguous(), trg, is_train, batch_max_length=self.max_len)
        elif self.rnn_type == 'transformer':
            #====================================#
            #======Transformer (By Kyohoon)======#
            #====================================#

            # 1) Feature reshape
            batch_size = features.size(0)
            features = features.squeeze(3)
            features = features.permute(2, 0, 1)  # [h, b, c]

            # 2) Target preprocessing
            trg_emb = self.embedding(trg)
            trg_emb = trg_emb.view(-1, batch_size, 512) # (Length, Batch, Feature)
            trg_key_padding_mask = (trg == self.pad_idx) # Need to fix pad_idx
            trg_key_padding_mask = trg_key_padding_mask.view(batch_size, -1)
            # 3) Transformer
            output = self.transformer(features, trg_emb, tgt_key_padding_mask=trg_key_padding_mask)
            output = torch.einsum('ijk->jik', output)
            output = self.output_linear(output)
            output = self.softmax(output)

        return output

    def predict(self, x):
        predicted = torch.LongTensor([[self.bos_idx]]).to(device)
        features = self.cnn(x)

        for _ in range(self.max_len):  # Need to fix input
            trg_emb = self.embedding(predicted)
            pred = self.transformer(features, trg_emb)
            pred = torch.einsum('ijk->jik', pred)
            pred = self.output_linear(pred)
            pred_id = pred.max(dim=2)[1][-1, 0] # Dimension confusing...

            if pred_id == self.eos_idx:
                break
            
            predicted = torch.cat([predicted, y_pred_id.view(1, 1)], dim=0) # Dimension confusing...

        return predicted