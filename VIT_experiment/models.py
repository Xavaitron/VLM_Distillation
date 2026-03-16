import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import timm

class CIFAR100ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(512, 100)
        
    def forward(self, x):
        return self.model(x)

def get_cnn_student():
    return CIFAR100ResNet18()

def get_cnn_teacher(teacher_name='Wang2023Better_WRN-28-10'):
    from robustbench.utils import load_model
    # The BDM model from the paper
    print(f"Loading robustbench CNN teacher: {teacher_name}...")
    model = load_model(model_name=teacher_name, dataset='cifar100', threat_model='Linf')
    return model

def get_vit_student():
    # A small ViT suitable for 32x32 inputs from timm
    # Alternatively, use a standard ViT with resized inputs (e.g. 224x224)
    # The user implied using ViT "instead of CNNs". We'll use a standard small ViT structure.
    # Note: For strict 32x32, we might want vit_tiny_patch4_32x32, but let's see what's available
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=100)
    return model

def get_vit_teacher():
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=100)
    return model
