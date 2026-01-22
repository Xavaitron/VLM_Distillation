# IGDM
This is the official code for "Indirect Gradient Matching for Adversarial Robust Distillation". 


## Setup
- Install pytorch
- Install Torchattacks following the [official site](https://github.com/Harry24k/adversarial-attacks-pytorch)
- Install RobustBench following the [official site](https://github.com/RobustBench/robustbench)
- Install wandb to use [Wandb.ai](https://wandb.ai)

### Code Implementation
- CIFAR-100 : `sh cifar100.sh`
- SVHN: `sh svhn.sh`
- Tiny-ImageNet : `sh tinyimg.sh`
- Changing 'method' in each bash file to run other methods.