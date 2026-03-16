#!/bin/bash

# Configuration for 24GB A30 GPUs
EPOCHS=200
CNN_BATCH_SIZE=1024 # ResNet-18 is much smaller, can handle large batches (~16GB)
VIT_BATCH_SIZE=256  # ViT is large, 256 uses ~18GB
CNN_GPU="1" # GPU ID for ResNet experiments
VIT_GPU="2" # GPU ID for ViT experiments

# For CNNs, we use the 3 RobustBench models specified in the paper:
# BDM (Wang), DEC (Cui), and LTD (Chen)
declare -a CNN_TEACHERS=("Wang2023Better_WRN-28-10" "Cui2023Decoupled_WRN-28-10" "Chen2021LTD_WRN34_10")

# For ViT, specify the path to your downloaded/trained ViT teacher weights.
VIT_TEACHER_PATH=""

echo "Starting Distillation Experiments..."

# --- Helper Functions for Parallel Execution ---

run_cnn_experiments() {
    for t in "${CNN_TEACHERS[@]}"; do
        echo "[CNN GPU $CNN_GPU] =================================="
        echo "[CNN GPU $CNN_GPU] Running Teacher: $t"
        echo "[CNN GPU $CNN_GPU] =================================="
        
        echo "[CNN GPU $CNN_GPU] Running Normal KD"
        python train_distill.py --arch cnn --method kd --epochs $EPOCHS --batch-size $CNN_BATCH_SIZE --teacher-name "$t" --gpu $CNN_GPU
        sleep 5

        echo "[CNN GPU $CNN_GPU] Running AdaAD"
        python train_distill.py --arch cnn --method adaad --epochs $EPOCHS --batch-size $CNN_BATCH_SIZE --teacher-name "$t" --gpu $CNN_GPU
        sleep 5

        echo "[CNN GPU $CNN_GPU] Running AdaAD + IGDM"
        python train_distill.py --arch cnn --method adaad_igdm --epochs $EPOCHS --batch-size $CNN_BATCH_SIZE --teacher-name "$t" --gpu $CNN_GPU
        sleep 5
    done
    echo "[CNN GPU $CNN_GPU] All CNN experiments finished!"
}

run_vit_experiments() {
    echo "[ViT GPU $VIT_GPU] =================================="
    echo "[ViT GPU $VIT_GPU] Running ViT Experiments"
    echo "[ViT GPU $VIT_GPU] =================================="

    echo "[ViT GPU $VIT_GPU] Running Normal KD"
    python train_distill.py --arch vit --method kd --epochs $EPOCHS --batch-size $VIT_BATCH_SIZE --teacher-name "$VIT_TEACHER_PATH" --gpu $VIT_GPU
    sleep 5

    echo "[ViT GPU $VIT_GPU] Running AdaAD"
    python train_distill.py --arch vit --method adaad --epochs $EPOCHS --batch-size $VIT_BATCH_SIZE --teacher-name "$VIT_TEACHER_PATH" --gpu $VIT_GPU
    sleep 5

    echo "[ViT GPU $VIT_GPU] Running AdaAD + IGDM"
    python train_distill.py --arch vit --method adaad_igdm --epochs $EPOCHS --batch-size $VIT_BATCH_SIZE --teacher-name "$VIT_TEACHER_PATH" --gpu $VIT_GPU
    echo "[ViT GPU $VIT_GPU] All ViT experiments finished!"
}

# --- Launch Experiments in Parallel ---

# Start ViT side in the background
run_vit_experiments &
VIT_PID=$!

# Start CNN side in the foreground (or background)
run_cnn_experiments &
CNN_PID=$!

# Wait for both chains to finish
echo "Both pipelines have been dispatched to the GPUs. Waiting for completion..."
wait $CNN_PID
wait $VIT_PID

echo "All parallel distillation experiments have successfully finished!"
