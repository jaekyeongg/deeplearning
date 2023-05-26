'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import numpy as np
import random

import wandb
#################### Random Seed 고정 ####################
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
##########################################################


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--dataset', help='dataset', default='cifar10', type=str)
parser.add_argument('--block', help='block type', default='RESNET', type=str)
parser.add_argument('--save_dir', default='save_temp', type=str)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

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

cfg = {
    'RESNET': ['None','None','None','None'],
    'SA_123': ['SA','SA', 'SA','None'],
    'SA_1': ['SA','None', 'None','None'],
    'SA_12': ['SA','SA', 'None','None'],
    'SA_23': ['None','SA', 'SA','None'],
    'NEW_1': ['NEW','None', 'None','None'],
    'NEW_12': ['NEW','NEW', 'None','None'],
    'NEW_123': ['NEW','NEW', 'NEW','None'],
    'NEW_23': ['None','NEW', 'NEW','None'],
    'SE_12': ['SE','SE', 'None','None'],
    'SE_23': ['None','SE', 'SE','None'],
    'SE_34': ['None','None','SE', 'SE'],
    'AA_12': ['AA','AA', 'None','None'],
    'AA_23': ['None','AA', 'AA','None'],
    'AA_34': ['None','None','AA', 'AA'],
    'SE_SA_1': ['SE_SA','None','None', 'None'],
    'CBAM_1': ['CBAM','None','None', 'None'],
}

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18(cfg=cfg[args.block])
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
#net = SimpleDLA()
print("model : ", net)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    #cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
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

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    wandb.log({
        'train_acc': 100.*correct/total,
        'train_loss': train_loss/(batch_idx+1)
    })


def test(epoch):
    global best_acc
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

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
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
        save_path = os.path.join(args.save_dir, args.block)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(state, os.path.join(save_path, 'checkpoint_{}.pth'.format(epoch)))
        best_acc = acc


for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    test(epoch)
    scheduler.step()
