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

import torch.nn.functional as F

class ViTStudentWrapper(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model
        
    def forward(self, x):
        # Upsample 32x32 from dataloader to 224x224 expected by the Vision Transformer
        # This allows fast PGD generation on the 32x32 space!
        x_up = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        return self.vit(x_up)

def get_vit_student():
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=100)
    return ViTStudentWrapper(model)

def get_vit_teacher(teacher_name='Debenedetti2022Light_XCiT-S12'):
    from robustbench.utils import load_model
    # Robust XCiT-S12 from RobustBench (~26M params, 62.80% clean, 27.16% robust on CIFAR-100 Linf)
    print(f"Loading robustbench ViT teacher: {teacher_name}...")
    model = load_model(model_name=teacher_name, dataset='cifar100', threat_model='Linf')
    return model
