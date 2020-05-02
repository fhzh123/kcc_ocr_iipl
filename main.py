import os
import argparse
import csv
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import Label_Encoder
from recognition_architecture import recognition_architecture
from custom_dataloader import custom_dataloader
from custom_dataloader import alignCollate
from custom_dataloader import randomSequentialSampler

import warnings

warnings.filterwarnings("ignore")

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
parser.add_argument('--optimizer', '-o', default='adam',
                    choices=optimizer_options)
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1e-3,
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
    global args, char_set, vocab
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    test_id = '20_05_02' + str(args.model_type) + '_' + str(args.rnn_type) + '_' + str(args.batch_size)

    csv_path = 'logs/'
    model_path = 'checkpoints'

    print(test_id, csv_path)

    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    char_file = open('dataset/new_charset.txt', 'r', encoding='utf-16')
    char_set = char_file.read()
    char_file.close()

    print("class Num : " + str(len(char_set)))

    if args.dataset == "ours":
        train_dataset = custom_dataloader('dataset/word_image/', 'dataset/label_final.txt', model = args.model_type)
        test_dataset = custom_dataloader('dataset/test_word_image/', 'dataset/label_test.txt', model = args.model_type)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True, pin_memory = True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    if args.rnn_type == 'transformer':
        model = recognition_architecture(args.model_type, args.rnn_type, 1, len(char_set) + 1, 256)
    else:
        model = recognition_architecture(args.model_type, args.rnn_type, 1, len(char_set), 256)

    criterion = nn.CrossEntropyLoss().cuda()

    if device.type == 'cuda':
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(args.ngpu))
        criterion = criterion.cuda()

    encoder = Label_Encoder()    

    # setup optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                            betas=(0.5, 0.999))
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

    best_acc = 0
    for epoch in range(args.epochs):
        progress_bar = tqdm(train_loader)

        #adjust_learning_rate(optimizer, epoch)
        ''' train for one epoch'''
        train_loss = train(progress_bar, model, criterion, optimizer, epoch)
    
        test_acc, test_loss = test(test_loader, model, criterion)

        tqdm.write('train_loss: {0:.3f} / test_loss: {1:.3f} test_acc: {2:.3f}'.format(train_loss, test_loss, test_acc))

        row = {'epoch': str(epoch), 'train_loss': str(train_loss), 'test_loss': str(test_loss), 'test_acc': str(test_acc)}
        csv_logger.writerow(row)

        if test_acc > best_acc:
            best_acc = test_acc
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
        for p in model.parameters():
            p.requires_grad = True

        model.train()

        # img : 한자 묶음 image
        img = torch.FloatTensor(np.asarray(img))
        img = img.cuda()
        img_var = torch.autograd.Variable(img)

        # target은 한자 단어
        # ex(target : 夙駕)

        char, length = encode_word(target)
        # char은 배치의 단어를 각 한자의 index로 변환하여 하나의 tensor로 뱉음
        # ex target : 夙駕 -> char : 0,1
        # 여기서 index는 charset 기준 (Charset은 데이터셋에 사용된 한자들의 집합)
        # length는 각 char의 길이

        if args.rnn_type == 'transformer':
            c = torch.LongTensor(args.batch_size, 10)
            temp = 0
            # target : [夙駕, 夙]
            # char : [[0, 1,4952,4952,4952,....,4952],[0,4952,4952...,4952]]
            for index_length, leng in enumerate(length):
                c[index_length][:leng] = torch.LongTensor(char[temp : temp + leng])
                c[index_length][leng:] = 4952 # 4952 = pad indx
                temp = temp + leng
        else:
            # target : [夙駕, 夙]
            # char : [0, 1, 0] , length = [2, 1]
            c = torch.LongTensor(char)
        c = c.cuda()
        c_var = torch.autograd.Variable(c)

        l = torch.LongTensor(length)
        l = length.cuda()
        l_var = torch.autograd.Variable(l)

        if args.rnn_type == 'transformer':
            output = model(img_var, l_var, c_var)
        else:
            output = model(img_var, l_var)

        loss = criterion(output, c_var)
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

    for i, (img, target) in enumerate(loader):
        for p in model.parameters():
            p.requires_grad = False

        model.eval()

        # measure data loading time
        img = torch.FloatTensor(np.asarray(img))
        img = img.cuda()
        img_var = torch.autograd.Variable(img)

        char, length = encode_word(target)

        if args.rnn_type == 'transformer':
            c = torch.LongTensor(args.batch_size, 10)
            temp = 0
            for index_length, leng in enumerate(length):
                c[index_length][:leng] = torch.LongTensor(char[temp : temp + leng])
                c[index_length][leng:] = 4952
                temp = temp + leng
        else:
            c = torch.LongTensor(char)
        c = c.cuda()
        c_var = torch.autograd.Variable(c)

        l = torch.LongTensor(length)
        l = length.cuda()
        l_var = torch.autograd.Variable(l)

        with torch.no_grad():
            if args.rnn_type == 'transformer':
                output = model(img_var, l_var, c_var)
            else:
                output = model(img_var, l_var)

        loss = criterion(output, c_var)

        if args.rnn_type == 'transformer':
            _, output = output.max(1)

            correct += (output == c.data).sum().item()
        else:
            _, output = output.max(1)
            output = output.view(-1)

            output = output.cpu()

            temp = 0
            for l_t in length:
                acc = 0
                for l_p in range(l_t):
                    if output[temp + l_p] == char[temp + l_p]:
                        acc = acc + 1
                temp = temp + l_t
                correct = correct + (acc / l_t)

        total += len(target)
        xentropy_loss_avg += loss.item()

    ''' Calculate running average of accuracy '''
    accuracy = correct / total
    test_loss = xentropy_loss_avg / total

    return accuracy, test_loss

def encode_word(target):
    char_l = []
    length = []
    for word in target:
        l = 0
        for char in word:
            char_l.append(char_set.index(char))
            l = l + 1
        length.append(l)
    return torch.LongTensor(np.asarray(char_l)), torch.LongTensor(np.asarray(length))

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 200))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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