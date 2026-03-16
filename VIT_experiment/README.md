# VIT Experiment: Adversarial Robust Distillation

This repository contains an independent, clean-room implementation of the distillation algorithms from the ICLR 2025 paper: **Indirect Gradient Matching for Adversarial Robust Distillation (IGDM)**.

The codebase is optimized for running on high-end GPUs like an **A30 (24GB VRAM)** and evaluates methods like Normal KD, AdaAD, and AdaAD with IGDM on both ResNet and Vision Transformer (ViT) architectures.

## 🚀 Setup Instructions for SSH Server

### 1. Environment Setup

It is highly recommended to use a virtual environment or conda environment.

```bash
# Create and activate a conda environment
conda create -n vit_distill python=3.10 -y
conda activate vit_distill

# Or using Python venv
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

Install PyTorch according to your server's CUDA version. For example, for CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Install the remaining dependencies from the provided requirements file:
```bash
pip install -r requirements.txt
```

*Note: The script automatically relies on `robustbench` to download the specific teacher models (BDM, DEC, LTD) mentioned in the paper, and `autoattack` to run the rigorous benchmark evaluations.*

---

## 🏃‍♂️ Running the Experiments

The `run_experiments.sh` script automates the entire pipeline. It sequentially runs Normal KD, AdaAD, and AdaAD+IGDM across the three RobustBench ResNet teachers (BDM, DEC, and LTD) and your provided ViT teacher models.

### Modify Configuration
Before running, open `run_experiments.sh` and set your preferred parameters:
- `GPU_ID`: Set to `"0"` or `"1"` depending on which GPU you want to utilize on the SSH server.
- `EPOCHS`: Default is 150.
- `BATCH_SIZE`: Default is 256 (tailored for 24GB VRAM).
- `VIT_TEACHER_PATH`: Provide the absolute or relative path to your pre-trained ViT weights if running Transformer experiments.

### Execute
```bash
bash run_experiments.sh
```

---

## 📊 Evaluation & Metrics

The pipeline evaluates student models against multiple rigorous attacks at the end of every epoch. The following metrics are automatically recorded into `result_models/training_results.csv`:
- **Clean Accuracy**
- **FGSM Accuracy** ($\epsilon = 8/255$)
- **PGD-20 Accuracy** (20 steps, $\epsilon = 8/255$)
- **C&W Accuracy** ($L_\infty$ margin loss based PGD attack)
- **AutoAttack Accuracy** (Standard evaluation)

Check the terminal window or inspect `training_results.csv` to view the comprehensive evaluation of each model architecture and distillation method.
