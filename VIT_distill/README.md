# Knowledge Distillation Experiment: ViT Base → ViT Tiny (CIFAR-100)

## Overview
This experiment demonstrates the power of **Knowledge Distillation (KD)** by compressing a large **ViT Base** teacher model into a compact **ViT Tiny** student model. The goal is to retain as much accuracy as possible while reducing the model size significantly.

## Model Stats

| Metric | Teacher (ViT Base) | Student (ViT Tiny) | Reduction |
| :--- | :--- | :--- | :--- |
| **Parameters** | ~85.8 M | ~5.5 M | **~15.5x Smaller** |
| **Architecture** | Patch16, 224x224 | Patch16, 224x224 | Same input, fewer layers/heads |

## Experimental Results (CIFAR-100)

We ran this experiment for 10 epochs to compare a Baseline Student (trained normally) against a Distilled Student (trained with a Teacher).

| Model Configuration | Best Test Accuracy | Performance Gain |
| :--- | :--- | :--- |
| **Teacher Model** | **91.12%** | (Reference) |
| **Baseline Student** | 76.91% | - |
| **Distilled Student** | **83.38%** | **+6.47%** (vs Baseline) |

### 🔍 Analysis
- **Significant Boost**: Distillation provided a massive **~6.5% accuracy improvement** over training the student model alone.
- **High Retention**: The tiny student achieved **~91.5% of the teacher's performance** despite having only **6% of the parameters**.
- **Efficiency**: The student model is practical for deployment on edge devices where the massive ViT Base would be too slow or heavy.

## Training Details
- **Dataset**: CIFAR-100 (Images resized to 224x224)
- **Distillation Params**:
  - **Temperature ($T$)**: 4.0 (Softens probability distributions)
  - **Alpha ($\alpha$)**: 0.5 (Equal weight to Hard Label Loss and Soft Teacher Loss)
- **Optimizer**: AdamW, LR=3e-4
- **Epochs**: 10
