# ResNet for Testing Image Model Blocks  
This is the PyTorch implementation of ResNet network trained on CIFAR10 or CIFAR100 dataset

### Requirements. 
- Python 3.6+
- PyTorch 1.0+

### Training 
    # Start training with: 
    python main.py --block [block_type]
    # You can manually resume the training with: 
    python main.py --resume --lr=0.01 --block [block_type]

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
