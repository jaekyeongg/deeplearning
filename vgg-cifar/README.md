### run example
CUDA_VISIBLE_DEVICES=1 python main.py --dataset cifar10  --block [block_type]

CUDA_VISIBLE_DEVICES=2 python main.py --dataset cifar100 --block [block_type]

### block_type = "block_name"_"target layer"
SA_123 = Spaital Attention module is applied to layer 1, 2, and 3.
 
### block name
 - SA : Spatial attention module
 - SE : Squeeze-and-Excitation Block
 - SE_SA : linear pair of SE and SA
 - AA : Attention Augmented Convolutional Networks
 - CBAM : linear pair of channel attention and spaital attention
 - NEW : ours
