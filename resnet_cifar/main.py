'''Train CIFAR10 with PyTorch.'''
import argparse
import os
import numpy as np
import random

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import wandb

from resnet import *

#################### Random Seed 고정 ####################
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
##########################################################


# Training
def train(epoch, trainloader, net, criterion, optimizer, device):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    last_idx = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        last_idx = batch_idx

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(' - Train : Loss: %.3f | Acc: %.3f%% (=%d/%d)'
          % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    wandb.log({
        'train_acc': 100.*correct/total,
        'train_loss': train_loss/(last_idx+1)
    })


def test(epoch, testloader, net, criterion, best_acc, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(' - Test : Loss: %.3f | Acc: %.3f%% (=%d/%d)'
          % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    wandb.log({
        'test_acc': 100.*correct/total,
        'test_loss': test_loss/(batch_idx+1)
    })
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        save_path = os.path.join(args.save_dir, args.dataset, args.block)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(state, os.path.join(save_path, 'checkpoint_{}.pth'.format(epoch)))
        best_acc = acc

    return best_acc


def main(args):
    print("dataset :", args.dataset)
    print("weight folder :", args.save_dir)

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #            'dog', 'frog', 'horse', 'ship', 'truck')

    if args.dataset == 'cifar10':
        trainset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

        testset = datasets.CIFAR10(
            root='./data', train=False, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        wandb.init(
            project='ResNet_CIFAR10',
            entity="miv_yubin",
            config={
                'archi': 'ResNet18',
                'learning_rate': args.lr,
                'weight_decay': 5e-4,
                "epochs": args.epochs,
                "dataset": args.dataset
            }
        )
        wandb.run.name = args.block
    else:
        trainset = datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

        testset = datasets.CIFAR100(
            root='./data', train=False, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        wandb.init(
            project='ResNet_CIFAR100',
            entity="miv_yubin",
            config={
                'archi': 'ResNet18',
                'learning_rate': args.lr,
                'weight_decay': 5e-4,
                "epochs": args.epochs,
                "dataset": args.dataset
            }
        )
        wandb.run.name = args.block

    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'

    # Model
    print('==> Building model..')
    net = ResNet18(block=args.block, num_classes=100 if args.dataset == 'cifar100' else 10)

    # print("model : ", net)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        # cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for idx in range(start_epoch, start_epoch+args.epochs):
        train(idx, trainloader, net, criterion, optimizer, device)
        best_acc = test(idx, testloader, net, criterion, best_acc, device)
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ResNet Training')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--cpu', dest='cpu', action='store_true', help='use cpu')
    parser.add_argument('--save-dir', dest='save_dir', help='The directory used to save the trained models',
                        default='weights', type=str)
    parser.add_argument('--dataset', help='choose one of dataset : cifar10 or cifar100', default='cifar100', type=str)
    parser.add_argument('--block', help='block type', default='RESNET', type=str)

    args = parser.parse_args()

    main(args)
