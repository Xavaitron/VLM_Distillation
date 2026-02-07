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

## 🔄 Training Pipeline (Detailed)

### Hyperparameters
```
ε (epsilon) = 8/255 ≈ 0.0314    # Maximum perturbation magnitude
step_size   = 2/255 ≈ 0.0078    # PGD step size
α (alpha)   = 20                 # IGDM loss weight
β (beta)    = 1                  # Forward perturbation scale
γ (gamma)   = 1                  # Backward perturbation scale
```

---

### Step 1: Generate Adversarial Example (x_adv)

**Method**: AdaAD Inner Loss (10-step PGD)

```python
def adaad_inner_loss(student, teacher, X, steps=10):
    delta = uniform_random(-ε, ε)  # Start with random noise
    
    for _ in range(steps):
        # AdaAD objective: maximize KL divergence between student and teacher
        loss = KL(student(X + delta) || teacher(X + delta))
        
        # Gradient ascent (maximize divergence)
        delta = delta + step_size * sign(∇loss)
        delta = clamp(delta, -ε, ε)  # Project into ε-ball
    
    return X + delta  # x_adv
```

**Key Insight**: Unlike standard PGD (which attacks using ground truth labels), AdaAD finds perturbations where **student and teacher disagree most**.

---

### Step 2: Compute Perturbation Direction

```python
δ = x_adv - X  # Captures "where the student is weak"
```

---

### Step 3: Forward Passes

```python
# Teacher (frozen, no gradients)
teacher(X), teacher(X + βδ), teacher(X - γδ), teacher(x_adv)

# Student (gradients flow through these)
student(X), student(X + βδ), student(X - γδ), student(x_adv)
```

---

### Step 4: Loss Function

**Main KL Loss** (distill on adversarial examples):
```
L_KL = KL(student(x_adv) || teacher(x_adv))
```

**IGDM Loss** (match gradient behavior):
```
L_IGDM = KL(student(X+βδ) - student(X-γδ) || teacher(X+βδ) - teacher(X-γδ))
```

**Why `f(x+δ) - f(x-δ)`?** This approximates the gradient: `∂f/∂x ≈ (f(x+δ) - f(x-δ)) / 2δ`

**Combined Loss with Epoch Scaling**:
```
L_total = L_KL + α × (epoch/200) × L_IGDM
```

| Epoch | IGDM Weight |
|-------|-------------|
| 1 | 20 × (1/200) = 0.1 |
| 50 | 20 × (50/200) = 5.0 |
| 100 | 20 × (100/200) = 10.0 |

This lets the student first learn basic distillation, then gradually incorporate gradient matching.

---

### Step 5: Evaluation

**PGD-20 (epochs 91-100)**:
```python
# Standard PGD attack using TRUE labels (not teacher)
for _ in range(20):
    loss = CrossEntropy(student(X + delta), y_true)
    delta = delta + step_size * sign(∇loss)
```

**AutoAttack (after training)**: Ensemble of 4 attacks:
1. APGD-CE: Auto-PGD with Cross-Entropy
2. APGD-DLR: Auto-PGD with Difference of Logit Ratio
3. FAB: Fast Adaptive Boundary
4. Square: Black-box query attack

---

### Training vs Evaluation Attacks

| | Training (AdaAD) | Evaluation (PGD-20) |
|---|------------------|---------------------|
| Objective | Max student-teacher divergence | Minimize accuracy (true labels) |
| Steps | 10 | 20 |
| Purpose | Find hard examples for learning | Test robustness reliably |

---

## 📊 Evaluation

### During Training (after epoch 90)
- **PGD-20 Attack**: 20-step PGD with ε=8/255

### After Training
- **AutoAttack**: Ensemble of 4 attacks (APGD-CE, APGD-DLR, FAB, Square)
- This is the gold standard for measuring adversarial robustness

---

## 📈 Results

### CIFAR-100 (100 epochs, Student: ResNet-18)

| Method | Teacher | Robust Accuracy (AutoAttack) | IGDM Gain |
|--------|---------|------------------------------|-----------|
| AdaAD | BDM | 26.54% | - |
| **AdaAD + IGDM** | BDM | **27.88%** | **+1.34%** |
| ARD | DEC | 21.52% | - |
| **ARD + IGDM** | DEC | **24.81%** | **+3.29%** |

### Key Takeaways
- **IGDM consistently improves robust accuracy** across different methods and teachers
- AdaAD + IGDM: **+1.34%** improvement
- ARD + IGDM: **+3.29%** improvement
- Student (ResNet-18) achieves strong robustness with **3.2x fewer parameters** than teacher
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

### Option 1: With Conda (Python 3.10+)
```bash
conda create -n igdm python=3.10 -y && conda activate igdm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/fra31/auto-attack.git
pip install torchattacks
pip install robustbench --no-deps
pip install timm==1.0.9 geotorch torchdiffeq gdown==5.1.0 tqdm numpy Jinja2 pandas wandb
mv autoattack autoattack_local
```

### Option 2: With venv (Python 3.8 - No Conda)
```bash
# Create and activate virtual environment
python3.8 -m venv ~/igdm_env
source ~/igdm_env/bin/activate
pip install --upgrade pip

# Install PyTorch (CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install AutoAttack from GitHub (MUST be before robustbench)
pip install git+https://github.com/fra31/auto-attack.git

# Install TorchAttacks
pip install torchattacks

# Install RobustBench (no-deps to avoid pyautoattack issue)
pip install robustbench --no-deps

# Install RobustBench dependencies
pip install timm==1.0.9 geotorch torchdiffeq gdown==5.1.0 tqdm numpy Jinja2 pandas wandb

# Fix Python 3.8 compatibility (type hints issue)
python << 'EOF'
import os, glob
arch_dir = os.path.expanduser('~/igdm_env/lib/python3.8/site-packages/robustbench/model_zoo/architectures/')
for filepath in glob.glob(arch_dir + '*.py'):
    with open(filepath, 'r') as f:
        content = f.read()
    if 'from __future__ import annotations' not in content:
        content = 'from __future__ import annotations\n' + content
        with open(filepath, 'w') as f:
            f.write(content)
print('✅ RobustBench Python 3.8 compatibility fixed!')
EOF

# Rename local autoattack folder to avoid import conflicts
mv autoattack autoattack_local
```

### Verify Installation
```bash
python -c "import torch; print(f'✅ PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from robustbench.utils import load_model; print('✅ RobustBench OK')"
python -c "import autoattack; print('✅ AutoAttack OK')"
python -c "import torchattacks; print('✅ TorchAttacks OK')"
```

### Run Training
```bash
# AdaAD + IGDM (recommended)
python adaad_IGDM_cifar100.py --epochs 200 --teacher BDM --alpha 20 --beta 1 --gamma 1 --nowand 1

# ARD + IGDM
python ard_IGDM_cifar100.py --epochs 200 --teacher DEC --alpha 20 --beta 1 --gamma 1 --nowand 1

# RSLAD + IGDM
python rslad_IGDM_cifar100.py --epochs 200 --teacher BDM --alpha 20 --beta 1 --gamma 1 --nowand 1
```

### Activate Environment (Future Sessions)
```bash
source ~/igdm_env/bin/activate
cd ~/ICLR_IGDM
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