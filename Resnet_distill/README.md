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

---

## Experiment 2: ResNet-152 → ResNet-18 (Same-Architecture CNN)

We also explored same-architecture distillation within the CNN family.

### Model Stats

| Metric | Teacher (ResNet-152) | Student (ResNet-18) | Reduction |
| :--- | :--- | :--- | :--- |
| **Parameters** | ~58.3 M | ~11.2 M | **~5.2x Smaller** |
| **Architecture** | CNN (Deep) | CNN (Shallow) | Same Architecture |

### Results

| Model Configuration | Best Test Accuracy | Performance Gain |
| :--- | :--- | :--- |
| **Baseline ResNet-18** | 81.02% | - |
| **Distilled ResNet-18** | **81.96%** | **+0.94%** (vs Baseline) |

### 🔍 Analysis
- **Same-Architecture Success**: Knowledge transfers efficiently between CNNs of different depths.
- **Strong Baseline**: ResNet-18 already achieves strong performance (81.02%), leaving less room for improvement.
- **Compression**: Achieved **5.2x compression** while improving accuracy.

---

## Key Takeaways

| Distillation Type | Teacher → Student | Improvement | Compression |
| :--- | :--- | :--- | :--- |
| Same Architecture (ViT) | ViT-Base → ViT-Tiny | +6.47% | 15.5x |
| **Same Architecture (CNN)** | **ResNet-152 → ResNet-18** | **+0.94%** | **5.2x** |
| Cross-Architecture (C→T) | ResNet-152 → ViT-Tiny | +1.34% | 10.5x |

Same-architecture distillation works well for both Transformers and CNNs.

---

## Scripts Reference

| Script | Description |
| :--- | :--- |
| `resnet152_cifar100_teacher.py` | Train ResNet-152 teacher |
| `baseline_vit_tiny.py` | Train ViT-Tiny baseline |
| `distill_vit_tiny.py` | Distill ResNet-152 → ViT-Tiny |
| `baseline_resnet18.py` | Train ResNet-18 baseline |
| `distill_resnet18.py` | Distill ResNet-152 → ResNet-18 |
