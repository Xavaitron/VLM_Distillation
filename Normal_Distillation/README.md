# VLM Distillation: Knowledge Distillation Experiments

A comprehensive exploration of **Knowledge Distillation** techniques on CIFAR-100, comparing same-architecture and cross-architecture approaches across Vision Transformers (ViT) and CNNs (ResNet).

---

## 📊 Results Summary

### ViT-Based Distillation (VIT_distill/)

| Experiment | Teacher | Student | Baseline | Distilled | Gain | Compression |
|:-----------|:--------|:--------|:---------|:----------|:-----|:------------|
| **ViT → ViT** | ViT-Base (85.8M) | ViT-Tiny (5.5M) | 76.91% | **83.38%** | **+6.47%** | 15.5x |
| **ViT → ResNet** | ViT-Base (85.8M) | ResNet-18 (11.2M) | 78.44% | **80.86%** | **+2.42%** | 7.6x |
| **ViT → MobileViT** | ViT-Base (85.8M) | MobileViT-S (5.0M) | 83.90% | **84.45%** | **+0.55%** | 17.2x |

### ResNet-Based Distillation (Resnet_distill/)

| Experiment | Teacher | Student | Baseline | Distilled | Gain | Compression |
|:-----------|:--------|:--------|:---------|:----------|:-----|:------------|
| **ResNet → ViT** | ResNet-152 (58.3M) | ViT-Tiny (5.5M) | 79.38% | **80.72%** | **+1.34%** | 10.5x |
| **ResNet → ResNet** | ResNet-152 (58.3M) | ResNet-18 (11.2M) | 81.02% | **81.96%** | **+0.94%** | 5.2x |

---

## 🔑 Key Findings

1. **Same-Architecture Distillation Works Best**: ViT-Base → ViT-Tiny achieved the highest improvement (+6.47%)
2. **Cross-Architecture Works**: Knowledge transfers between CNNs ↔ Transformers in both directions
3. **Strong Baselines Limit Gains**: MobileViT & ResNet-18 have strong pretrained weights, leaving less room for improvement
4. **Compression Champion**: MobileViT achieves 17.2x compression while maintaining 84.45% accuracy

---

## 📁 Repository Structure

```
VLM_Distillation/
├── VIT_distill/                    # ViT-Base as Teacher
│   ├── vit_base_patch16_224_cifar100.py   # Train ViT-Base teacher
│   ├── baseline_vit_tiny.py               # ViT-Tiny baseline
│   ├── distill_vit_tiny.py                # ViT-Base → ViT-Tiny
│   ├── baseline_resnet.py                 # ResNet-18 baseline
│   ├── distillation_resnet.py             # ViT-Base → ResNet-18
│   ├── baseline_mobilevit.py              # MobileViT-S baseline
│   └── distill_mobilevit.py               # ViT-Base → MobileViT-S
│
└── Resnet_distill/                 # ResNet-152 as Teacher
    ├── resnet152_cifar100_teacher.py      # Train ResNet-152 teacher
    ├── baseline_vit_tiny.py               # ViT-Tiny baseline
    ├── distill_vit_tiny.py                # ResNet-152 → ViT-Tiny
    ├── baseline_resnet18.py               # ResNet-18 baseline
    └── distill_resnet18.py                # ResNet-152 → ResNet-18
```

---

## 🚀 Quick Start

```bash
# Clone and navigate
cd VLM_Distillation

# Train a teacher model
python VIT_distill/vit_base_patch16_224_cifar100.py

# Train baseline student
python VIT_distill/baseline_vit_tiny.py

# Distill teacher → student
python VIT_distill/distill_vit_tiny.py
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|:----------|:------|
| Dataset | CIFAR-100 (224×224) |
| Temperature (T) | 4.0 |
| Alpha (α) | 0.5 |
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Epochs | 10 |
| Batch Size | 64-128 |

---

## 📚 References

- [Hinton et al., 2015 - Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Touvron et al., 2021 - Training data-efficient image transformers](https://arxiv.org/abs/2012.12877)
