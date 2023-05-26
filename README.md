# Image Model Blocks for VGG and ResNet  
VGG와 ResNet를 대상으로 간단하고 쉽게 적용이 가능한 image model block의 성능 테스트를 수행

### 파일 구조 
 - resnet-cifar : CIFAR10과 CIFAR100으로 학습이 가능한 ResNet 폴더
 - vgg-cifar : CIFAR10과 CIFAR100으로 학습이 가능한 VGG 폴더
 - block.py  : 테스트를 위한 image model block를 모아놓은 파일 (resnet-cifar와 vgg-cifar에서 함께 사용)

#### Image model block 종류
 - SA : Spatial attention module
 - SE : Squeeze-and-Excitation block + Residual block
 - SEC : Squeeze-and-Excitation block with 1x1 conv.
 - AA : Attention Augmented Convolutional Networks
 - SE_SA : linear pair of SE and SA
 - SEC_SA : linear pair of SEC and SA
 - CBAM : linear pair of channel attention and spatial attention
 - NEW : our model
