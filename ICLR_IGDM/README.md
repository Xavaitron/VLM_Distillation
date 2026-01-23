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

| Flag | Architecture | Parameters | Use Case |
|------|--------------|------------|----------|
| `RES-18` | ResNet-18 | 11.2M | Default, good balance |
| `MN-V2` | MobileNetV2 | 3.4M | Mobile/edge deployment |

### ResNet-18 Architecture (Default Student)
```
Input (3x32x32)
    ↓
Conv1 (64 filters) → BN → ReLU
    ↓
Layer1: 2x BasicBlock (64 filters)
Layer2: 2x BasicBlock (128 filters, stride=2)
Layer3: 2x BasicBlock (256 filters, stride=2)
Layer4: 2x BasicBlock (512 filters, stride=2)
    ↓
Global Average Pooling
    ↓
FC (512 → 100 classes)
```

### Compression Ratio
| | Teacher (WRN-28-10) | Student (ResNet-18) | Reduction |
|---|---------------------|---------------------|-----------|
| Parameters | ~36.5M | ~11.2M | **3.2x smaller** |
| FLOPs | ~10.5G | ~0.56G | **18x faster** |

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

### During Training (after epoch 90)
- **PGD-20 Attack**: 20-step PGD with ε=8/255

### After Training
- **AutoAttack**: Ensemble of 4 attacks (APGD-CE, APGD-DLR, FAB, Square)
- This is the gold standard for measuring adversarial robustness

---

## 📈 Results

### CIFAR-100 (100 epochs, Teacher: BDM, Student: ResNet-18)

| Method | Robust Accuracy (AutoAttack) | Improvement |
|--------|------------------------------|-------------|
| AdaAD (Baseline) | 26.54% | - |
| **AdaAD + IGDM** | **27.88%** | **+1.34%** |

### Key Takeaways
- **IGDM improves robust accuracy by ~1.3%** over baseline distillation
- Student (ResNet-18) achieves ~72% of teacher's robustness with **3.2x fewer parameters**
- Training: 100 epochs on single A30 GPU

---

## 🔬 Distillation Methods

### 1. ARD (Adversarial Robust Distillation)

**Core Idea**: Distill on adversarial examples instead of clean examples.

**Loss Function**:
```
L_ARD = KL(S(x_adv) || T(x_adv)) + λ · CE(S(x_adv), y)
```

Where:
- `S(x_adv)` = student prediction on adversarial example
- `T(x_adv)` = teacher prediction on adversarial example  
- `CE` = cross-entropy with true label
- `λ` = balance coefficient

**Key Insight**: Standard distillation uses `KL(S(x) || T(x))` on clean inputs. ARD generates adversarial `x_adv` and distills there, forcing the student to learn robust representations.

---

### 2. RSLAD (Robust Soft Label Adversarial Distillation)

**Core Idea**: Use teacher's soft labels (instead of hard labels) for generating adversarial examples.

**Inner Loop** (generating x_adv):
```
x_adv = argmax_δ KL(S(x + δ) || T(x))     subject to ||δ||_∞ ≤ ε
```

**Outer Loss**:
```
L_RSLAD = KL(S(x_adv) || T(x_adv))
```

**Key Insight**: Traditional adversarial training uses hard labels `y` to generate attacks. RSLAD uses soft labels from teacher `T(x)`, which provides richer gradient information and better aligns student-teacher decision boundaries.

---

### 3. AdaAD (Adaptive Adversarial Distillation)

**Core Idea**: Adapt the adversarial attack to maximize divergence between student and teacher.

**Inner Loop** (adaptive attack):
```
x_adv = argmax_δ KL(S(x + δ) || T(x + δ))     subject to ||δ||_∞ ≤ ε
```

**Outer Loss**:
```
L_AdaAD = KL(S(x_adv) || T(x_adv))
```

**Key Insight**: Instead of attacking based on ground truth labels, AdaAD finds perturbations where student and teacher disagree most. This "adaptive" attack focuses training on the student's weakest regions.

---

### 4. IGDM Enhancement (This Paper)

**Core Idea**: Match the *gradient behavior* of teacher, not just outputs.

**IGDM Loss**:
```
L_IGDM = KL( [S(x+βδ) - S(x-γδ)] || [T(x+βδ) - T(x-γδ)] )
```

**Combined Loss** (e.g., AdaAD + IGDM):
```
L_total = L_AdaAD + α · (epoch/200) · L_IGDM
```

**Key Insight**: Matching `T(x+δ) - T(x-δ)` approximates matching the teacher's input gradient `∂T/∂x`. This transfers not just "what" the teacher predicts, but "how sensitively" it responds to perturbations — crucial for robustness.

---

### Method Comparison

| Method | Attack Target | Distillation Target | IGDM Compatible |
|--------|---------------|---------------------|-----------------|
| ARD | Hard labels `y` | `T(x_adv)` | ✅ |
| RSLAD | Soft labels `T(x)` | `T(x_adv)` | ✅ |
| AdaAD | Student-Teacher gap | `T(x_adv)` | ✅ |

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