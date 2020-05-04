import os
import argparse
import csv
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm import tqdm
from torch.utils.data import DataLoader

from recognition_architecture import recognition_architecture
from custom_dataloader import custom_dataloader
from custom_dataloader import alignCollate
from custom_dataloader import randomSequentialSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_options = ['resnet18', 'resnet34', 'resnet50', 'EfficientNet']
rnn_options = ['rnn', 'attention', 'transformer']
optimizer_options = ['adam', 'adadelta', 'rms', 'sgd']
dataset_options = ['ours']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='ours',
                    choices=dataset_options)
parser.add_argument('--model_type', '-a', default='EfficientNet',
                    choices=model_options)
parser.add_argument('--rnn_type', '-r', default='transformer',
                    choices=rnn_options)
parser.add_argument('--optimizer', '-o', default='adadelta',
                    choices=optimizer_options)
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1,
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,  metavar='M',
                    help='momentum')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--ngpu', type=int, default=1,
                    help='gpu number (default: 1)')
parser.add_argument('--resume', type=bool, default=False,
                    help='restart')
parser.add_argument('--lr_decay', type=float, default=1e-4,
                     help='learning decay for lr scheduler')


def main():
    global args, char_set, converter
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    test_id = '20_05_04' + str(args.model_type) + '_' + str(args.rnn_type) + '_' + str(args.batch_size) + '_2'

    csv_path = 'logs/'
    model_path = 'checkpoints'

    print(test_id, csv_path)

    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    #char_file = open('dataset/new_charset.txt', 'r', encoding='utf-16')
    char_file = open('dataset/charset_0503.txt', 'r', encoding='utf-16')
    char_set = char_file.read()
    char_file.close()

    print("class Num : " + str(len(char_set)))

    if args.dataset == "ours":
        train_dataset = custom_dataloader('dataset/label_train_0503.txt', model = args.model_type)
        #train_dataset = custom_dataloader('dataset/label_final.txt', model = args.model_type)
        test_dataset = custom_dataloader( 'dataset/label_test_0503.txt', model = args.model_type)
        #test_dataset = custom_dataloader( 'dataset/label_test.txt', model = args.model_type)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = recognition_architecture(args.model_type, args.rnn_type, len(char_set) + 2, 256, 10)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).cuda()
    #criterion = nn.NLLLoss().cuda()

    if device.type == 'cuda':
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(args.ngpu))

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))

    # setup optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.999))
    elif args.optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr,)
    elif args.optimizer == 'rms':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr,)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.lr_decay, nesterov=True)

    filename = csv_path + '/' + test_id + '.csv'
    csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_loss', 'test_loss', 'test_acc'], filename=filename)

    converter = AttnLabelConverter(char_set)

    best_loss = 100
    for epoch in range(args.epochs):
        progress_bar = tqdm(train_loader)

        #adjust_learning_rate(optimizer, epoch)
        ''' train for one epoch'''
        model.train()
        train_loss = train(progress_bar, model, criterion, optimizer, epoch)
    
        model.eval()
        with torch.no_grad():
            test_acc, test_loss = test(test_loader, model, criterion)

        tqdm.write('train_loss: {0:.3f} / test_loss: {1:.3f} test_acc: {2:.3f}'.format(train_loss, test_loss, test_acc))

        row = {'epoch': str(epoch), 'train_loss': str(train_loss), 'test_loss': str(test_loss), 'test_acc': str(test_acc)}
        csv_logger.writerow(row)

        if test_loss < best_loss:
            best_loss = test_loss
            save_checkpoint({
                'epoch': epoch,
                'arch': args.model_type,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}, model_path, test_id)
            tqdm.write('save_checkpoint')
    csv_logger.close()


def train(progress_bar, model, criterion, optimizer, epoch):
    xentropy_loss_avg = 0.
    total = 0.
    losses = 0

    for i, (img, target) in enumerate(progress_bar):
        img = torch.FloatTensor(np.asarray(img))
        img = img.cuda()

        text, length = converter.encode(target, batch_max_length=10)

        output = model(img, trg = text[:, :-1])
        labels = text[:, 1:]  # without [GO] Symbol
        loss = criterion(output.view(-1, output.shape[-1]), labels.contiguous().view(-1))

        model.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_description('Epoch = {0} / loss =  {1:.4f}'.format(str(epoch), loss.item()))
        
        xentropy_loss_avg += loss.item()

   
    ''' Calculate running average of accuracy '''
    train_loss = xentropy_loss_avg / (i + 1)

    progress_bar.set_postfix(xentropy='%.3f' % (train_loss))

    return train_loss

def test(loader, model, criterion):
    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.
    losses = 0
    last_c = None
    output_pred = None
    for i, (img, target) in enumerate(loader):
        batch_size = len(target)

        # For max length prediction
        length_for_pred = torch.IntTensor([10] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, 10 + 1).fill_(0).to(device)

        text_for_loss, length_for_loss = converter.encode(target, batch_max_length=10)

        # measure data loading time
        img = torch.FloatTensor(np.asarray(img))
        img = img.cuda()

        preds = model(img, trg = text_for_pred, is_train=False)

        preds = preds[:, :text_for_loss.shape[1] - 1, :]
        label = text_for_loss[:, 1:]  # without [GO] Symbol
        loss = criterion(preds.contiguous().view(-1, preds.shape[-1]), label.contiguous().view(-1))

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)
        labels = converter.decode(text_for_loss[:, 1:], length_for_loss)
        r_index = random.randint(0, len(target) - 1)
        output_pred = preds_str[r_index]
        last_c = labels[r_index]

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            gt = gt[:gt.find('[s]')]
            pred_EOS = pred.find('[s]')
            pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[:pred_EOS]
            
            if pred == gt:
                correct += 1

        xentropy_loss_avg += loss.item()

    tqdm.write('target : ' + output_pred + ' test output :' + last_c)
    ''' Calculate running average of accuracy '''
    accuracy = correct / (i+1)
    test_loss = xentropy_loss_avg / (i+1)

    return accuracy, test_loss

class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default
        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts

class CSVLogger():
    def __init__(self, args, fieldnames, filename='log.csv'):
        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()

def save_checkpoint(state, model_path, test_id):
    filename = model_path + '/' + test_id +'.pth.tar'
    torch.save(state, filename)

if __name__ == '__main__':
    main()