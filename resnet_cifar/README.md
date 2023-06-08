# ResNet Model for Testing Attention Blocks    
This is the PyTorch implementation of ResNet network trained on CIFAR10 or CIFAR100 dataset

## Training 
	# CIFAR10
	python main.py --dataset cifar10  --block [block type]
    # CIFAR100
	python main.py --dataset cifar100 --block [block type]

### How to set 'block' argument
It consist of block name and layer position to be added.

    [block type] = [block name]_[layer position]

Ex) SA_123 = Spaital Attention module is applied to layer 1, 2, and 3.

##### List of block_type

| Block name | Layer position |
|:----------:|:---------------|
|   RESNET   | -              |
|     CA     | 1, 23, 34      |
|     SA     | 1, 12, 123, 23 |
|     SE     | 1, 23, 34      |
|    SEC     | 1, 23, 34      |
|     AA     | 1, 12, 123, 23 |
|   SE_SA    | 1, 12, 123 |
|   SEC_SA   | 1, 12, 123 |
|    CBAM    | 1, 12, 123 |
|    NEW     | 1, 12, 123 |

### Inference
	python inference.py --batch-size [# of images to be tested] --checkpoint [weight file] --dataset [cifar10 or cifar100] --block [block type]

It should be the same 'dataset' and 'block' as when training the weight file (checkpoint).