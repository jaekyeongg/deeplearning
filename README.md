# Exploiting Channel Attention and Spatial Attention Blocks
We investigated the effects of applying well-known attention blocks like __Squeeze & Excitation Block__, 
__Channel Attention Module__, and __Spatial Attention Module__ at different positions in ResNet18 and VGG19.
And then, we propose new image model block that combines channel and spatial attention blocks and
emphasizes them together.

So, we use the CIFAR dataset to train ResNet18 and VGG19 with several attention blocks added, 
and analyze the inference results.

## Prerequisites

|  Package  |  Link   |
|:---------:|:-------|
|  PyTorch  |   https://pytorch.org/get-started/locally/   |
| Grad-Cam  |  https://github.com/jacobgil/pytorch-grad-cam   |

## Training

|       Tool       |                Model                 | File                 |
|:----------------:|:------------------------------------:|:---------------------|
| Jupyter notebook |           ResNet18, VGG19            | demo_training.ipynb  |
|   Command line   | ResNet18 | resnet_cifar/main.py |
|   Command line   | ResNet18 | vgg_cifar/main.py    |

## Inference

|       Tool       |                Model                 | File                      |
|:----------------:|:------------------------------------:|:--------------------------|
| Jupyter notebook |           ResNet18, VGG19            | demo_inference.ipynb      |
|   Command line   | ResNet18 | resnet_cifar/inference.py |
|   Command line   | ResNet18 | vgg_cifar/inference.py         |

## Attention blocks

|    Name     |   Type   | Description                                   |
|:-----------:|:--------:|:----------------------------------------------|
|     CA      |  Single  | Channel Attention module                      |
| SE (or SER) |  Single  | Squeeze-and-Excitation block + Residual block |
|     SEC     |  Single  | Squeeze-and-Excitation block with 1x1 conv.   |
|     SA      |  Single  | Spatial attention module                      |
|     AA      |  Single  | Attention Augmented Convolutional Networks    |
|    SE_SA    | Combine  | linear combination of SE and SA               |
|   SEC_SA    | Combine  | linear combination of SEC and SA              |
|    CBAM     | Combine  | linear combination of CA and SA               |
|     NEW     | Combine  | __our model__                                 |

### CA
![CA](https://github.com/jaekyeongg/deeplearning/blob/main/fig/CAM.png)

### SEC
![SEC](https://github.com/jaekyeongg/deeplearning/blob/main/fig/SEblock.png)

### SA
![SEC](https://github.com/jaekyeongg/deeplearning/blob/main/fig/SAM.png)

### Combination of channel attention and spatial attention
![SEC](https://github.com/jaekyeongg/deeplearning/blob/main/fig/our_block.png)

## Performance
Please, read __'demo.pdf'__ file or run 'demo_inference.ipynb' file.

### Test Accuracy with Cifar100 in ResNet18
![experiment](https://github.com/jaekyeongg/deeplearning/blob/main/fig/experiment.png)

#### Comparison of CBAM and our model by using GradCAM
![GradCam](https://github.com/jaekyeongg/deeplearning/blob/main/fig/GradCam.png)

### Test Accuracy with Cifar10 in VGG19
![experiment2](https://github.com/jaekyeongg/deeplearning/blob/main/fig/experiment2.png)


