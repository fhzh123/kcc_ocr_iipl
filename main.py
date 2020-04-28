import argparse

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from recognition_architecture import recognition_architecture
from custom_dataloader import custom_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_options = ['resnet18', 'resnet34', 'resnet50', 'EfficientNet']
dataset_options = ['ours']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='ours',
                    choices=dataset_options)
parser.add_argument('--model_type', '-a', default='EfficientNet',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.25,
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,  metavar='M',
                    help='momentum')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--resume', type=bool, default=False,
                    help='restart')
parser.add_argument('--lr_decay', type=float, default=1e-4,
                     help='learning decay for lr scheduler')


def main():
    global args
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    if args.dataset == "ours":
        dataset = custom_dataloader('dataset/word_image/', 'dataset/label_new.txt', model = args.model_type)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = recognition_architecture(args.model_type)
    model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.lr_decay, nesterov=True)

    # filename = csv_path + '/' + test_id + '.csv'
    # csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'], filename=filename)

    #print(model)

    criterion = nn.CrossEntropyLoss().cuda()

    best_acc = 0
    for epoch in range(args.epochs):
        progress_bar = tqdm(train_loader)

        #adjust_learning_rate(optimizer, epoch)

        ''' train for one epoch'''
        train_acc, train_loss = train(progress_bar, model, criterion, optimizer, epoch)
    
        # test_acc, test_loss = test(test_loader, model, criterion)

        tqdm.write('train_loss: {0:.3f} train_acc: {1:.3f} / test_loss: {2:.3f} test_acc: {3:.3f}'.format(train_loss, train_acc, test_loss, test_acc))
        # tqdm.write('test_loss: %.3f' % (test_loss))
        # tqdm.write('test_acc: %.3f' % (test_acc))

        # row = {'epoch': str(epoch), 'train_loss': str(train_loss), 'train_acc': str(train_acc), 'test_loss': str(test_loss), 'test_acc': str(test_acc)}
        # csv_logger.writerow(row)

        # if test_acc > best_acc:
        #     best_acc = test_acc
        #     save_checkpoint({
        #         'epoch': epoch,
        #         'arch': args.model_type,
        #         'state_dict': model.state_dict(),
        #         'optimizer': optimizer.state_dict()}, model_path, test_id)
        #     tqdm.write('save_checkpoint')
    csv_logger.close()


def train(progress_bar, model, criterion, optimizer, epoch):
    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.
    losses = 0

    for i, (img, target) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        # measure data loading time
        img = img.cuda()
        target = target.cuda()

        img_var = torch.autograd.Variable(img, requires_grad=True)
        target_var = torch.autograd.Variable(target)
        target_var = target_var.type(torch.long)
        target_var = target_var.squeeze()

        output = model(img_var)

    #     loss = criterion(output, target_var)
    #     output = torch.max(output.data, 1)[1]

    #     total += target.size(0)
    #     correct += (output == target.data).sum().item()
    #     xentropy_loss_avg += loss.item()

    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    ''' Calculate running average of accuracy '''
    # accuracy = correct / total
    # train_loss = xentropy_loss_avg / total

    progress_bar.set_postfix(xentropy='%.3f' % (train_loss), acc='%.3f' % accuracy)

    return accuracy, train_loss

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