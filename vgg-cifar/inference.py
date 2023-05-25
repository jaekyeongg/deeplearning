import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import vgg

import numpy as np
import random
import cv2

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

model_names = sorted(name for name in vgg.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("vgg")
                     and callable(vgg.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: vgg19)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='use cpu')
parser.add_argument('--checkpoint', dest='checkpoint',
                    help='The directory used to save the trained models',
                    default='./save_temp/checkpoint_0.tar', type=str)
parser.add_argument('--dataset', help='dataset', default='cifar10', type=str)



classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def main():
    global args, best_prec1
    args = parser.parse_args()
    if args.dataset == "cifar10" :
        num_classes = 10
    elif args.dataset == "cifar100" :
        num_classes = 100
    print("dataset : ", args.dataset)
    print("num classes : ", num_classes)
    print("args.checkpoint : ", args.checkpoint)

    model = vgg.__dict__[args.arch](num_classes)
    model.features = torch.nn.DataParallel(model.features)
    if args.cpu:
        model.cpu()
    else:
        model.cuda()
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.dataset == "cifar10" :
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    elif args.dataset == "cifar100" :
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    dataiter = iter(val_loader)
    images, labels = next(dataiter)
    print("images shape : ", images.shape)
    #img = torchvision.utils.make_grid(images)
    #images = images / 2 + 0.5     # unnormalize
    #npimg = images.numpy()
    #print("npimg shape : ", npimg.shape)
    torchvision.utils.save_image(images, "./input.png", nrow=4, normalize = True, range=(-1,1))
    print("input gt labels : ")
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(args.batch_size)))
    output = model(images)
    maxk = 1
    pred = output.topk(maxk, 1, True, True)
    print("pred : ", pred)
    print("pred labels : ")
    for i in range(args.batch_size) :
        print(classes[pred.indices[i]])

if __name__ == '__main__':
    main()
