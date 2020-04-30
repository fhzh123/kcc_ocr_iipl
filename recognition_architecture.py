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

        # Transformer model setting
        # Need to fix Hyperparameter & vocab_num & etc
        self.transformer = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, 
                                          num_decoder_layers=6, dim_feedforward=2048, 
                                          dropout=0.1, activation='gelu')
        self.embedding = nn.Embedding(vocab_num, 512)
        self.output_linear = nn.Linear(d_model=512, vocab_num, bias=False)

        self.max_len = max_len # Maximum Length
        self.bos_idx = bos_idx # Start token of sentence
        self.eos_idx = eos_idx # End token of sentece
        self.pad_idx = pad_idx # Padding token of sentece
        
    def forward(self, x): # Need to fix input
        ''' (Batch, 3, 48, 480) '''
        features = self.cnn(x)
        ''' (Batch, 512, 3, 30) ''' 

#################################################################################

        #====================================#
        #======Transformer (By Kyohoon)======#
        #====================================#

        # 1) Feature reshape
        batch_size = features.size(1)
        features = features.view(-1, batch_size, 512) # (W*H, Batch, Feature)

        # 2) Target preprocessing
        trg_emb = self.embedding(trg)
        trg_emb = trg_emb.view(-1, batch_size, 512) # (Length, Batch, Feature)
        trg_key_padding_mask = (trg == args.pad_idx) # Need to fix pad_idx

        # 3) Transformer
        outputs = self.transformer(features, trg_emb, tgt_key_padding_mask=trg_key_padding_mask)
        outputs = torch.einsum('ijk->jik', outputs) # (Batch, Length, Feature)
        outputs = self.output_linear(outputs)

        return outputs

    def predict(self, x): # Need to fix input
        predicted = torch.LongTensor([[self.bos_idx]]).to(device)
        features = self.cnn(x)

        for _ in range(self.max_len):
            trg_emb = self.embedding(predicted)
            pred = self.transformer(features, trg_emb)
            pred = torch.einsum('ijk->jik', pred)
            pred = self.output_linear(pred)
            pred_id = pred.max(dim=2)[1][-1, 0] # Dimension confusing...

            if pred_id == self.eos_idx:
                break
            
            predicted = torch.cat([predicted, y_pred_id.view(1, 1)], dim=0) # Dimension confusing...

        return predicted