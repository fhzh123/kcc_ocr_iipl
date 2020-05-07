import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TestTransformer(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TestTransformer, self).__init__()

        self.embedding = nn.Embedding(num_classes, input_size)
        self.transformer = nn.Transformer(d_model=input_size, nhead=8, num_encoder_layers=6, 
                                            num_decoder_layers=6, dim_feedforward=2048, 
                                            dropout=0.1, activation='gelu')
        self.generator = nn.Linear(input_size, num_classes, bias=False)
        self.pad_idx = 2
        self.num_classes = num_classes
        self.input_size = input_size

    def forward(self, batch_H, text, is_train=True, batch_max_length=10):
        # Transformer
        # src : (S,N,E) N : Batch Size, E : Feature number
        # tgt = (T,N,E) T : Target sequence Length, E: Feature number
        # tgt_key_padding_mask : (N, T)

        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.

        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.input_size).fill_(0).to(device)

        # src : (Batch, 1, Channel)
        batch_H = batch_H.permute(1, 0, 2) # (Batch, 1, Channel) -> (1, Batch, 256)

        # text = (Bacth, Length)
        text = text.permute(1, 0) # (Batch, Length) -> (Length, Batch)

        if is_train:
            #trg_emb = (Batch, Length, Channel)
            trg_emb = self.embedding(text)
            
            #trg_key_padding_mask = (Length, Batch)
            trg_key_padding_mask = (text == self.pad_idx)
            trg_key_padding_mask = trg_key_padding_mask.permute(1, 0)

            # pred = (Length, Batch, Channel)
            pred = self.transformer(batch_H, trg_emb, tgt_key_padding_mask=trg_key_padding_mask)
            pred = pred.permute(1, 0, 2) # (Length, Batch, Channel) -> (Batch, Length, Channel)
            
            # probs = (Batch, Length, Vocab_size)
            probs = self.generator(pred)

        else:
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

class Transformer(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Transformer, self).__init__()

        self.embedding = nn.Embedding(num_classes, input_size)
        self.transformer = nn.Transformer(d_model=input_size, nhead=8, num_encoder_layers=6, 
                                            num_decoder_layers=6, dim_feedforward=2048, 
                                            dropout=0.1, activation='gelu')
        self.generator = nn.Linear(input_size, num_classes, bias=False)
        self.pad_idx = 2
        self.num_classes = num_classes
        self.input_size = input_size

    def forward(self, batch_H, text, is_train=True, batch_max_length=10):
        # Transformer
        # src : (S,N,E) N : Batch Size, E : Feature number
        # tgt = (T,N,E) T : Target sequence Length, E: Feature number
        # tgt_key_padding_mask : (N, T)

        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.

        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.input_size).fill_(0).to(device)

        # src : (Batch, 1, Channel)
        batch_H = batch_H.permute(1, 0, 2) # (Batch, 1, Channel) -> (1, Batch, 256)

        # text = (Bacth, Length)
        text = text.permute(1, 0) # (Batch, Length) -> (Length, Batch)

        if is_train:
            for i in range(num_steps):
                #trg_emb = (Batch, Channel)
                trg_emb = self.embedding(text[i, :])
                trg_emb = trg_emb.unsqueeze(0) #(Batch, Channel) -> (1, Batch, Channel)
                
                #trg_key_padding_mask = (Batch)
                trg_key_padding_mask = (text[i, :] == self.pad_idx) # Need to fix pad_idx
                trg_key_padding_mask = trg_key_padding_mask.unsqueeze(1) # (Batch) -> (Batch, 1)

                # pred = (1, Batch, Channel)
                pred = self.transformer(batch_H, trg_emb, tgt_key_padding_mask=trg_key_padding_mask)
                pred = pred.permute(1, 0, 2) # (1, Batch, Channel) -> (Batch, 1, Channel)
                pred = pred.squeeze(1) # (Batch, 1, Channel) -> (Batch, Channel)

                output_hiddens[:, i, :] = pred
            probs = self.generator(output_hiddens)

        else:
            targets = torch.LongTensor(batch_size).fill_(0).to(device)  # [GO] token
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)

            for i in range(num_steps):
                #trg_emb = (Batch, Channel)
                trg_emb = self.embedding(targets)
                trg_emb = trg_emb.unsqueeze(0) #(Batch) -> (1, Batch)

                pred = self.transformer(batch_H, trg_emb)
                pred = pred.permute(1, 0, 2) # (1, Batch, Channel) -> (Batch, 1, Channel)
                pred = pred.squeeze(1) # (Batch, 1, Channel) -> (Batch, Channel)
                pred = self.generator(pred) # (Batch, Vocab_size)

                probs[:, i, :] = pred

                _, next_input = pred.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes





class Attention(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)

    def _char_to_onehot(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, batch_H, text, is_train=True, batch_max_length=10):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x num_classes]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1  # +1 for [s] at end of sentence.

        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)
        hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device))

        if is_train:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
            probs = self.generator(output_hiddens)

        else:
            targets = torch.LongTensor(batch_size).fill_(0).to(device)  # [GO] token
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
                probs_step = self.generator(hidden[0]) # (Batch, Vocab.size)
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes


class AttentionCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # batch_size x num_channel
        concat_context = torch.cat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha