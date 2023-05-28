import argparse
import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import vgg

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, \
    EigenGradCAM, LayerCAM, FullGrad, GradCAMElementWise
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image

import numpy as np
import random

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

# Label and its index for CIFAR10
# https://www.cs.toronto.edu/~kriz/cifar.html
class_cifar10 = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
                 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

# Label and its index for CIFAR100
# https://huggingface.co/datasets/cifar100
class_cifar100 = {0: 'apple', 1: 'aquarium_fish', 2: 'baby', 3: 'bear', 4: 'beaver', 5: 'bed', 6: 'bee', 7: 'beetle',
                  8: 'bicycle', 9: 'bottle', 10: 'bowl', 11: 'boy', 12: 'bridge', 13: 'bus', 14: 'butterfly',
                  15: 'camel', 16: 'can', 17: 'castle', 18: 'caterpillar', 19: 'cattle', 20: 'chair', 21: 'chimpanzee',
                  22: 'clock', 23: 'cloud', 24: 'cockroach', 25: 'couch', 26: 'cra', 27: 'crocodile', 28: 'cup',
                  29: 'dinosaur', 30: 'dolphin', 31: 'elephant', 32: 'flatfish', 33: 'forest', 34: 'fox', 35: 'girl',
                  36: 'hamster', 37: 'house', 38: 'kangaroo', 39: 'keyboard', 40: 'lamp', 41: 'lawn_mower',
                  42: 'leopard', 43: 'lion', 44: 'lizard', 45: 'lobster', 46: 'man', 47: 'maple_tree', 48: 'motorcycle',
                  49: 'mountain', 50: 'mouse', 51: 'mushroom', 52: 'oak_tree', 53: 'orange', 54: 'orchid', 55: 'otter',
                  56: 'palm_tree', 57: 'pear', 58: 'pickup_truck', 59: 'pine_tree', 60: 'plain', 61: 'plate',
                  62: 'poppy', 63: 'porcupine', 64: 'possum', 65: 'rabbit', 66: 'raccoon', 67: 'ray', 68: 'road',
                  69: 'rocket', 70: 'rose', 71: 'sea', 72: 'seal', 73: 'shark', 74: 'shrew', 75: 'skunk',
                  76: 'skyscraper', 77: 'snail', 78: 'snake', 79: 'spider', 80: 'squirrel', 81: 'streetcar',
                  82: 'sunflower', 83: 'sweet_pepper', 84: 'table', 85: 'tank', 86: 'telephone', 87: 'television',
                  88: 'tiger', 89: 'tractor', 90: 'train', 91: 'trout', 92: 'tulip', 93: 'turtle', 94: 'wardrobe',
                  95: 'whale', 96: 'willow_tree', 97: 'wolf', 98: 'woman', 99: 'worm'}


def main(args):
    #############################################
    # Load dataset
    #############################################
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.dataset == "cifar100":
        num_classes = 100
        classes = class_cifar100
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:  # default dataset is CIFAR10
        num_classes = 10
        classes = class_cifar10
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    print("dataset : ", args.dataset)
    print("num classes : ", num_classes)
    print("args.checkpoint : ", args.checkpoint)

    #############################################
    # Load model
    #############################################
    model = vgg.__dict__[args.arch](num_classes, args.block)

    cam_layers = [model.features[52]]

    model.features = torch.nn.DataParallel(model.features)
    if args.cpu:
        model.cpu()
    else:
        model.cuda()
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    #############################################
    # Evaluate model
    #############################################
    dataiter = iter(val_loader)
    images, labels = next(dataiter)
    print("images shape : ", images.shape)
    #img = torchvision.utils.make_grid(images)
    #images = images / 2 + 0.5     # unnormalize
    #npimg = images.numpy()
    #print("npimg shape : ", npimg.shape)
    torchvision.utils.save_image(images, "./input.png", nrow=4, normalize=True, range=(-1, 1))
    print("input gt labels : ")
    np_labels = labels.detach().cpu()
    print([classes[int(np_labels[j])] for j in range(args.batch_size)])
    output = model(images)
    maxk = 1
    pred = output.topk(maxk, 1, True, True)
    # print("pred : ", pred)
    print("pred labels : ")
    np_indices = pred.indices.detach().cpu()
    print([classes[int(np_indices[j][0])] for j in range(args.batch_size)])

    #############################################
    # Create CAM
    #############################################
    cam = GradCAM(model=model, target_layers=cam_layers, use_cuda=False if args.cpu else True)
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=False if args.cpu else True)

    grayscale_cams = cam(input_tensor=images)

    final_cam = None
    final_gb = None
    final_cam_gb = None
    for idx, grayscale_cam in enumerate(grayscale_cams):
        tensor_img = images[idx]

        rgb_img = deprocess_image(tensor_img.permute(1, 2, 0).numpy()) / 255.0
        # print(rgb_img)
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        gb = gb_model(tensor_img[None, :], target_category=None)

        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)

        if final_cam is None:
            final_cam = cam_image
            final_gb = gb
            final_cam_gb = cam_gb
        else:
            final_cam = cv2.hconcat([final_cam, cam_image])
            final_gb = cv2.hconcat([final_gb, gb])
            final_cam_gb = cv2.hconcat([final_cam_gb, cam_gb])

    cv2.imwrite('gradCAM_cam.jpg', final_cam)
    cv2.imwrite('gradCAM_gb.jpg', final_gb)
    cv2.imwrite('gradCAM_cam_gb.jpg', final_cam_gb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch VGG Evaluation')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg19',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                             ' (default: vgg19)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--cpu', dest='cpu', action='store_true', help='use cpu')
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='The directory used to save the trained models',
                        default='./save_temp/checkpoint_0.tar', type=str)
    parser.add_argument('--dataset', help='choose one of dataset : cifar10 or cifar100', default='cifar10', type=str)
    parser.add_argument('--block', help='block_type', default='VGG19', type=str)

    main(parser.parse_args())
