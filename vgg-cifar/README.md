# VGG Model for Testing Image Model Blocks  
This is the PyTorch implementation of VGG network trained on CIFAR10 or CIFAR100 dataset

### Requirements. 
[PyTorch] (https://pytorch.org/)

[wandb] (https://wandb.ai/site)

### Training 
	# CIFAR10
	CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10  --block [block_type]
    # CIFAR100
	CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar100 --block [block_type]

#### block_type = "block_name"_"target layer"
 - SA_123 = Spaital Attention module is applied to layer 1, 2, and 3.
 
#### block name
 - SA : Spatial attention module
 - SE : Squeeze-and-Excitation block + Residual block
 - SEC : Squeeze-and-Excitation block with 1x1 conv.
 - AA : Attention Augmented Convolutional Networks
 - SE_SA : linear pair of SE and SA
 - SEC_SA : linear pair of SEC and SA
 - CBAM : linear pair of channel attention and spatial attention
 - NEW : our model

### Evaluation 	
	# CUDA version
	python main.py --resume=[trained weights file (.tar)] -e
	# CPU version	
	python main.py --resume=[trained weights file (.tar)] -e --cpu
