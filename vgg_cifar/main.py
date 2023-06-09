import argparse
import os
import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import wandb

import vgg

#################### Random Seed 고정 ####################
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
##########################################################


def main(args):
    if args.dataset == "cifar10" :
        num_classes = 10
    elif args.dataset == "cifar100" :
        num_classes = 100
    print("dataset :", args.dataset)
    print("weight folder :", args.save_dir)

    # Check the save_dir exists or not
    save_path = os.path.join(args.save_dir, args.dataset, args.block)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = vgg.__dict__[args.arch](num_classes, args.block)

    model.features = torch.nn.DataParallel(model.features)
    if args.cpu:
        model.cpu()
    else:
        model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = False

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.dataset == "cifar10":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    elif args.dataset == "cifar100" :
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss()
    if args.cpu:
        criterion = criterion.cpu()
    else:
        criterion = criterion.cuda()

    if args.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.dataset == "cifar10" :
        wandb.init(
            project='VGG_CIFAR10',
            entity="miv_yubin",
            config={
                'archi': args.arch,
                'learning_rate': args.lr,
                'weight_decay': args.weight_decay,
                "epochs": args.epochs,
                "dataset": args.dataset
            }
        )
    elif args.dataset == "cifar100" :
        wandb.init(
            project='VGG_CIFAR100',
            entity="miv_yubin",
            config={
                'archi': args.arch,
                'learning_rate': args.lr,
                'weight_decay': args.weight_decay,
                "epochs": args.epochs,
                "dataset": args.dataset
            }
        )
    wandb.run.name = args.block

    if args.evaluate:
        validate(val_loader, model, criterion, args.cpu, args.half, args.print_freq)
        return

    best_prec1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args.lr, optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args.cpu, args.half, args.print_freq)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, args.cpu, args.half, args.print_freq)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(save_path, 'checkpoint_{}.tar'.format(epoch)))


def train(train_loader, model, criterion, optimizer, epoch, is_cpu, is_half, print_freq):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    if print_freq < 0:
        print_freq = len(train_loader) - 1
    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if is_cpu == False:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        if is_half:
            input = input.half()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader) - 1, batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    wandb.log({
            'train_prec1': top1.avg,
            'train_loss': losses.avg
        })


def validate(val_loader, model, criterion, is_cpu, is_half, print_freq):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    if print_freq < 0:
        print_freq = len(val_loader) - 1
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if is_cpu == False:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        if is_half:
            input = input.half()

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader) - 1, batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    wandb.log({
            'test_prec1': top1.avg,
            'test_loss': losses.avg
        })

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


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


def adjust_learning_rate(lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    new_lr = lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    model_names = sorted(name for name in vgg.__dict__
                         if name.islower() and not name.startswith("__")
                         and name.startswith("vgg")
                         and callable(vgg.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch VGG Trainer')
    parser.add_argument('--arch', metavar='ARCH', default='vgg19_bn', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: vgg19)')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=0.05, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--print-freq', default=20, type=int, metavar='N', help='print frequency (default: 20)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--half', dest='half', action='store_true', help='use half-precision(16-bit) ')
    parser.add_argument('--cpu', dest='cpu', action='store_true', help='use cpu')
    parser.add_argument('--save-dir', dest='save_dir', help='The directory used to save the trained models',
                        default='weights', type=str)
    parser.add_argument('--dataset', help='choose one of dataset : cifar10 or cifar100', default='cifar10', type=str)
    parser.add_argument('--block', help='block_type', default='VGG19', type=str)

    args = parser.parse_args()

    main(args)
