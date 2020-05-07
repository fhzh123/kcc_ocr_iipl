import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SequenceTransformer(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SequenceTransformer, self).__init__()
        self.embedding = nn.Embedding(num_classes, input_size)
        self.transformer = nn.Transformer(d_model=input_size, nhead=8, num_encoder_layers=6, 
                                            num_decoder_layers=6, dim_feedforward=2048, 
                                            dropout=0.1, activation='gelu')
        self.generator = nn.Linear(input_size, num_classes, bias=False)
        self.pad_idx = 2
        self.num_classes = num_classes
    
    def forward(self, batch_H, text, is_train=True, batch_max_length=10):
        # Transformer
        # src : (S,N,E) N : Batch Size, E : Feature number
        # tgt = (T,N,E) T : Target sequence Length, E: Feature number
        # tgt_key_padding_mask : (N, T)

        # src : (Batch, 1, Channel)
        batch_H = batch_H.permute(1, 0, 2) # (Batch, 1, Channel) -> (1, Batch, 512)

        # text = (Bacth, Length)
        text = text.permute(1, 0) # (Batch, Length) -> (Length, Batch)

        if is_train:
            #trg_emb = (Batch, Length, Channel)
            trg_emb = self.embedding(text)
            
            #trg_key_padding_mask = (Length, Batch)
            trg_key_padding_mask = (text == self.pad_idx) # Need to fix pad_idx
            trg_key_padding_mask = trg_key_padding_mask.permute(1, 0)

            # pred = (Length, Batch, Channel)
            pred = self.transformer(batch_H, trg_emb, tgt_key_padding_mask=trg_key_padding_mask)
            pred = pred.permute(1, 0, 2) # (Length, Batch, Channel) -> (Batch, Length, Channel)
            
            # probs = (Batch, Length, Vocab_size)
            probs = self.generator(pred)
        else:
            batch_size = batch_H.size(1)
            num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.

            targets = torch.LongTensor(batch_size).fill_(0).to(device)  # [GO] token
            targets = targets.unsqueeze(0)
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)

            for i in range(num_steps):
                #trg_emb = (1, Batch)
                trg_emb = self.embedding(targets)
                
                #trg_key_padding_mask = (Length, Batch)
                trg_key_padding_mask = (targets == self.pad_idx)
                trg_key_padding_mask = trg_key_padding_mask.permute(1, 0)

                pred = self.transformer(batch_H, trg_emb, tgt_key_padding_mask=trg_key_padding_mask)
                pred = pred.permute(1, 0, 2) # (1, Batch, Channel) -> (Batch, 1, Channel)
                pred = self.generator(pred) # (Batch, 1, Vocab_size)
                pred_id = pred.max(2)[1] # (Batch, 1)
                pred_id = pred_id[:, -1] # (Batch)
                pred_id = pred_id.unsqueeze(0)
                targets = torch.cat([targets, pred_id], dim=0)
            probs = pred

        return probs  # batch_size x num_steps x num_classes

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output
