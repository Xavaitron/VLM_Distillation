# Knowledge Distillation Experiment: ResNet-152 → ViT-Tiny (CIFAR-100)

## Overview
This experiment demonstrates **Cross-Architecture Knowledge Distillation** by compressing a large **ResNet-152** (CNN) teacher model into a compact **ViT-Tiny** (Transformer) student model. This explores whether a Transformer can learn from CSV's inductive biases (locality, translation equivariance).

## Model Stats

| Metric | Teacher (ResNet-152) | Student (ViT-Tiny) | Reduction |
| :--- | :--- | :--- | :--- |
| **Parameters** | ~58.3 M | ~5.5 M | **~10.5x Smaller** |
| **Architecture** | CNN (Deep ResNet) | Transformer (ViT) | Cross-Architecture |
| **Input Size** | 224x224 | 224x224 | Same |

## Experimental Results (CIFAR-100)

We ran this experiment for 10 epochs to compare a Baseline ViT-Tiny (trained normally) against a Distilled ViT-Tiny (trained with ResNet-152 teacher).

| Model Configuration | Best Test Accuracy | Performance Gain |
| :--- | :--- | :--- |
| **Baseline ViT-Tiny** | 79.38% | - |
| **Distilled ViT-Tiny** | **80.72%** | **+1.34%** (vs Baseline) |

### 🔍 Analysis
- **Cross-Architecture Success**: Successfully transferred knowledge from a CNN to a Transformer, improving the ViT's performance by **+1.34%**.
- **CNN → Transformer**: The ViT student benefits from the CNN teacher's local feature extraction patterns and hierarchical representations.
- **Compression**: Achieved **10.5x compression** while improving accuracy over the baseline.
- **Different Paradigms**: This proves knowledge can flow between fundamentally different architectures (convolutions vs attention).

## Training Details
- **Dataset**: CIFAR-100 (Images resized to 224x224)
- **Distillation Params**:
  - **Temperature ($T$)**: 4.0 (Softens probability distributions)
  - **Alpha ($\alpha$)**: 0.5 (Equal weight to Hard Label Loss and Soft Teacher Loss)
- **Optimizer**: AdamW, LR=3e-4
- **Epochs**: 10

## Key Takeaways

| Distillation Type | Teacher → Student | Improvement |
| :--- | :--- | :--- |
| Same Architecture | ViT-Base → ViT-Tiny | +6.47% |
| Cross-Architecture (T→C) | ViT-Base → ResNet-18 | +2.42% |
| **Cross-Architecture (C→T)** | **ResNet-152 → ViT-Tiny** | **+1.34%** |

Cross-architecture distillation works in both directions, though same-architecture distillation tends to yield larger gains.

---

## Scripts Reference

| Script | Description |
| :--- | :--- |
| `resnet152_cifar100_teacher.py` | Train ResNet-152 teacher |
| `baseline_vit_tiny.py` | Train ViT-Tiny baseline |
| `distill_vit_tiny.py` | Distill ResNet-152 → ViT-Tiny |
