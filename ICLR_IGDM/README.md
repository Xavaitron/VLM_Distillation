# IGDM - Indirect Gradient Matching for Adversarial Robust Distillation

This is the official code for "Indirect Gradient Matching for Adversarial Robust Distillation" (ICLR 2024).

---

## 📖 Overview

**Goal**: Train a small, efficient neural network (student) that is robust to adversarial attacks by learning from a large, pre-trained robust model (teacher).

### The Problem
- Large adversarially robust models (e.g., WideResNet-28-10) are accurate but slow and memory-heavy
- Small models (e.g., ResNet-18) are fast but lack robustness when trained normally
- Standard knowledge distillation doesn't transfer adversarial robustness well

### The Solution: IGDM
**Indirect Gradient Matching** improves distillation by matching not just outputs, but how the student and teacher *respond to perturbations*.

---

## 🧠 Method

### Traditional Distillation
```
Student tries to match: teacher(x)
```

### IGDM Distillation
```
Student tries to match: teacher(x + δ) - teacher(x - δ)
```

This captures the **gradient behavior** of the teacher — how its predictions change when inputs are perturbed. By matching this, the student learns the teacher's robust decision boundaries.

### Loss Function

```python
# Standard KL loss on adversarial examples
kl_loss = KL(student(x_adv), teacher(x_adv))

# IGDM loss: match the gradient-like behavior
igdm_loss = KL(
    student(x + β*δ) - student(x - γ*δ),
    teacher(x + β*δ) - teacher(x - γ*δ)
)

# Total loss
loss = kl_loss + α * igdm_loss
```

Where:
- `δ` = adversarial perturbation direction
- `α` = IGDM loss weight (default: 20)
- `β, γ` = perturbation scaling factors (default: 1)

---

## 🏫 Teacher Models

Teachers are **pre-trained adversarially robust models** from [RobustBench](https://robustbench.github.io/). They are automatically downloaded — no training needed!

| Model | Architecture | Clean Acc | Robust Acc (AutoAttack) |
|-------|--------------|-----------|-------------------------|
| Wang2023Better (BDM) | WideResNet-28-10 | 72.58% | 38.83% |
| Cui2023Decoupled (DEC) | WideResNet-28-10 | 73.85% | 35.54% |
| Chen2021LTD (LTD) | WideResNet-34-10 | 72.02% | 34.66% |

These models were trained with adversarial training techniques and millions of parameters.

---

## 🎓 Student Models

Students are **smaller models trained from scratch** to inherit the teacher's robustness.

| Model | Architecture | Parameters |
|-------|--------------|------------|
| RES-18 | ResNet-18 | 11.2M |
| MN-V2 | MobileNetV2 | 3.4M |

---

## 🔄 Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING LOOP                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Load batch (x, y) from CIFAR-100                                │
│                                                                      │
│  2. Generate adversarial example x_adv using PGD attack             │
│     └── 10 steps, ε = 8/255, step_size = 2/255                      │
│                                                                      │
│  3. Compute perturbation direction: δ = x_adv - x                   │
│                                                                      │
│  4. Get teacher predictions (frozen, no grad):                      │
│     └── teacher(x), teacher(x+δ), teacher(x-δ), teacher(x_adv)      │
│                                                                      │
│  5. Get student predictions:                                        │
│     └── student(x), student(x+δ), student(x-δ), student(x_adv)      │
│                                                                      │
│  6. Compute losses:                                                 │
│     └── KL loss: student(x_adv) vs teacher(x_adv)                  │
│     └── IGDM loss: gradient matching                                │
│                                                                      │
│  7. Backprop and update student                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Evaluation

### During Training (after epoch 190)
- **PGD-20 Attack**: 20-step PGD with ε=8/255

### After Training
- **AutoAttack**: Ensemble of 4 attacks (APGD-CE, APGD-DLR, FAB, Square)
- This is the gold standard for measuring adversarial robustness

---

## 🔬 Baseline Methods

| Method | Description |
|--------|-------------|
| **ARD** | Adversarial Robust Distillation - distill on adversarial examples |
| **RSLAD** | Robust Soft Label Adversarial Distillation - uses soft labels |
| **AdaAD** | Adaptive Adversarial Distillation - adapts to student capacity |
| **+ IGDM** | Any method enhanced with Indirect Gradient Matching |

---

## 🚀 Quick Start

```bash
# Setup
conda create -n igdm python=3.10 -y && conda activate igdm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install robustbench torchattacks
pip install git+https://github.com/fra31/auto-attack.git
mv autoattack autoattack_local

# Run
python adaad_IGDM_cifar100.py --epochs 200 --teacher BDM --alpha 20 --beta 1 --gamma 1 --nowand 1
```

---

## 📁 File Structure

| File | Purpose |
|------|---------|
| `adaad_IGDM_cifar100.py` | AdaAD + IGDM (recommended) |
| `ard_IGDM_cifar100.py` | ARD + IGDM |
| `rslad_IGDM_cifar100.py` | RSLAD + IGDM |
| `*_cifar100.py` (no IGDM) | Baseline methods |
| `attacks.py` | PGD, FGSM, IGDM inner loop |
| `rslad_loss.py` | Loss functions |
| `cifar100_models/` | Student architectures |

---

## 📚 References

- [RobustBench: A Standardized Adversarial Robustness Benchmark](https://robustbench.github.io/)
- [AutoAttack: Reliable Evaluation of Adversarial Robustness](https://github.com/fra31/auto-attack)