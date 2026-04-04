import os
import argparse
import time
import csv
import glob
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from dataset import get_cifar100_dataloaders
from models import get_cnn_student, get_cnn_teacher, get_vit_student, get_vit_teacher
from losses import kd_loss, adaad_inner_loss, igdm_inner_loss
from attacks import eval_robustness

def find_latest_checkpoint(result_dir, arch, teacher_arch, method):
    """Find the latest checkpoint matching the current training config."""
    pattern = os.path.join(result_dir, f"{arch}_{teacher_arch}teacher_{method}_checkpoint_epoch_*.pt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None, 0
    # Extract epoch numbers and find the max
    def extract_epoch(path):
        match = re.search(r'_epoch_(\d+)\.pt$', path)
        return int(match.group(1)) if match else 0
    latest = max(checkpoints, key=extract_epoch)
    epoch = extract_epoch(latest)
    return latest, epoch

def main():
    parser = argparse.ArgumentParser(description='Distillation Training')
    parser.add_argument('--arch', type=str, default='cnn', choices=['cnn', 'vit'], help='Model architecture family')
    parser.add_argument('--method', type=str, default='kd', choices=['kd', 'adaad', 'adaad_igdm'], help='Distillation method')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (256-512 good for 24GB VRAM)')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=20.0, help='IGDM loss weight')
    parser.add_argument('--beta', type=float, default=1.0, help='Forward perturbation scale for IGDM')
    parser.add_argument('--gamma', type=float, default=1.0, help='Backward perturbation scale for IGDM')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--teacher-name', type=str, default='Wang2023Better_WRN-28-10', help='RobustBench teacher name or custom path')
    parser.add_argument('--gpu', type=str, default='0', help='Comma separated list of GPU(s) to use.')
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint')
    args = parser.parse_args()

    # Set GPU explicitly
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Setup seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cross-architecture: Both CNN and ViT will take 32x32 from dataloader to save VRAM and keep PGD fast.
    # The ViT student will internally upsample to 224x224 in its forward pass.
    img_size = 32
    trainloader, testloader = get_cifar100_dataloaders(batch_size=args.batch_size, img_size=img_size)

    # Initialize models
    if args.arch == 'cnn':
        student = get_cnn_student()
        teacher = get_cnn_teacher(teacher_name=args.teacher_name)
        teacher_arch = 'cnn'
    else:
        student = get_vit_student()
        teacher = get_vit_teacher(teacher_name=args.teacher_name)
        teacher_arch = 'vit'

    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()  # Teacher is always in eval mode

    optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # Mixed precision scaler
    scaler = GradScaler()

    # Scheduling from ICLR_IGDM codebase
    def adjust_learning_rate(optimizer, epoch, args):
        lr = args.lr
        if args.epochs > 150:
            if epoch >= 100: lr *= 0.1
            if epoch >= 150: lr *= 0.1
        else:
            if epoch >= 70: lr *= 0.1
            if epoch >= 90: lr *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    os.makedirs('result_models', exist_ok=True)

    # Resume from checkpoint if requested
    start_epoch = 1
    if args.resume:
        ckpt_path, ckpt_epoch = find_latest_checkpoint('result_models', args.arch, teacher_arch, args.method)
        if ckpt_path is not None:
            print(f"🔄 Resuming from checkpoint: {ckpt_path} (epoch {ckpt_epoch})")
            checkpoint = torch.load(ckpt_path, map_location=device)
            student.load_state_dict(checkpoint['student_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            del checkpoint
            torch.cuda.empty_cache()
            print(f"   Resuming training from epoch {start_epoch}")
        else:
            print("⚠️  --resume was set but no checkpoint found. Starting from scratch.")

    print(f"Starting training: Arch={args.arch}, Method={args.method}, Epochs={args.epochs}, AMP=enabled")
    print(f"  Training epochs {start_epoch} to {args.epochs}")

    for epoch in range(start_epoch, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch, args)
        student.train()
        
        train_loss = 0.0
        
        for step, (X, y) in enumerate(trainloader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            if args.method == 'kd':
                # Normal KD: Only matches clean outputs
                with autocast('cuda'):
                    student_logits = student(X)
                    with torch.no_grad():
                        teacher_logits = teacher(X)
                    loss = kd_loss(student_logits, teacher_logits)
                
            elif args.method == 'adaad':
                # AdaAD: Generate adv examples (inner loop already uses AMP via losses.py)
                X_adv = adaad_inner_loss(student, teacher, X, step_size=2/255.0, steps=10, epsilon=8/255.0)
                
                # Forward passes with AMP
                with autocast('cuda'):
                    student_clean = student(X)
                    student_adv = student(X_adv)
                    with torch.no_grad():
                        teacher_clean = teacher(X)
                        teacher_adv = teacher(X_adv)
                    
                    # AdaAD loss consists of clean KD + adv KD
                    kl_clean = kd_loss(student_clean, teacher_clean)
                    kl_adv = kd_loss(student_adv, teacher_adv)
                    loss = kl_clean + kl_adv

                # Free intermediates
                del X_adv, student_clean, student_adv, teacher_clean, teacher_adv
                
            elif args.method == 'adaad_igdm':
                # Generate adversarial examples (inner loop already uses AMP via losses.py)
                X_adv = adaad_inner_loss(student, teacher, X, step_size=2/255.0, steps=10, epsilon=8/255.0)
                
                # Compute IGDM perturbation
                delta = X_adv - X
                
                with autocast('cuda'):
                    with torch.no_grad():
                        teacher_clean = teacher(X)
                        teacher_adv = teacher(X_adv)
                        teacher_plus = teacher(X + args.beta * delta)
                        teacher_minus = teacher(X - args.gamma * delta)
                    
                    # Forward passes
                    student_clean = student(X)
                    student_adv = student(X_adv)
                    student_plus = student(X + args.beta * delta)
                    student_minus = student(X - args.gamma * delta)
                    
                    # KD Losses (Clean + Adv)
                    kl_clean = kd_loss(student_clean, teacher_clean)
                    kl_adv = kd_loss(student_adv, teacher_adv)
                    
                    # IGDM Gradient Matching Loss
                    criterion_kl = nn.KLDivLoss(reduction="batchmean")
                    ours_loss = criterion_kl(F.log_softmax(student_plus - student_minus, dim=1), 
                                             F.softmax((teacher_plus - teacher_minus).detach(), dim=1))
                    
                    # Combined Loss with Epoch Scaling
                    loss = kl_clean + kl_adv + args.alpha * (epoch / args.epochs) * ours_loss

                # Free intermediates
                del X_adv, delta, teacher_clean, teacher_adv, teacher_plus, teacher_minus
                del student_clean, student_adv, student_plus, student_minus

            # AMP backward + step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

            if step % 50 == 0:
                print(f"Epoch {epoch}/{args.epochs} Step {step}/{len(trainloader)} Loss: {loss.item():.4f}")

        # Save checkpoint every 25 epochs
        if epoch % 25 == 0:
            ckpt_name = f"{args.arch}_{teacher_arch}teacher_{args.method}_checkpoint_epoch_{epoch}.pt"
            ckpt_path = os.path.join('result_models', ckpt_name)
            torch.save({
                'epoch': epoch,
                'student_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'args': vars(args),
            }, ckpt_path)
            print(f"💾 Checkpoint saved: {ckpt_name}")

        # Robust evaluation after saving epoch 200 checkpoint (or final epoch if > 200)
        if epoch == 200 or epoch == args.epochs:
            torch.cuda.empty_cache()
            clean_acc, pgd_acc, fgsm_acc, cw_acc, aa_acc = eval_robustness(student, testloader, device)
            print(f"Epoch {epoch} Results: Clean Acc: {clean_acc*100:.2f}%, PGD20 Acc: {pgd_acc*100:.2f}%, FGSM Acc: {fgsm_acc*100:.2f}%, C&W Acc: {cw_acc*100:.2f}%, AA Acc: {aa_acc*100:.2f}%")
        else:
            student.eval()
            correct = 0
            with torch.no_grad():
                with autocast('cuda'):
                    for X_test, y_test in testloader:
                        X_test, y_test = X_test.to(device), y_test.to(device)
                        correct += (student(X_test).argmax(1) == y_test).sum().item()
            clean_acc = correct / len(testloader.dataset)
            pgd_acc, fgsm_acc, cw_acc, aa_acc = 0.0, 0.0, 0.0, 0.0
            print(f"Epoch {epoch} Results: Clean Acc: {clean_acc*100:.2f}% (Robust eval at epoch 200 & final)")

        torch.cuda.empty_cache()

    # Save final model
    save_name = f"{args.arch}_{teacher_arch}teacher_{args.method}_epochs_{args.epochs}.pt"
    torch.save(student.state_dict(), os.path.join('result_models', save_name))
    print(f"Saved final robust model to {save_name}")

    # Log to CSV
    csv_file = os.path.join('result_models', 'training_results.csv')
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Date', 'Architecture', 'Method', 'Teacher', 'Batch Size', 'Epochs', 'Clean Acc (%)', 'FGSM Acc (%)', 'PGD20 Acc (%)', 'C&W Acc (%)', 'AA Acc (%)'])
        writer.writerow([
            time.strftime('%Y-%m-%d %H:%M:%S'),
            args.arch.upper(),
            args.method.upper(),
            args.teacher_name,
            args.batch_size,
            args.epochs,
            f"{clean_acc*100:.2f}",
            f"{fgsm_acc*100:.2f}",
            f"{pgd_acc*100:.2f}",
            f"{cw_acc*100:.2f}",
            f"{aa_acc*100:.2f}"
        ])

    # Print final formatted summary
    print("\n" + "="*60)
    print("📊 TRAINING RESULTS SUMMARY")
    print("="*60)
    print(f"Architecture:    {args.arch.upper()}")
    print(f"Method:          {args.method.upper()}")
    print(f"Teacher:         {args.teacher_name}")
    print(f"Epochs:          {args.epochs}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"AMP:             Enabled")
    print("-" * 60)
    print(f"Clean Accuracy:  {clean_acc*100:.2f}%")
    print(f"FGSM Accuracy:   {fgsm_acc*100:.2f}%")
    print(f"PGD-20 Accuracy: {pgd_acc*100:.2f}%")
    print(f"C&W Accuracy:    {cw_acc*100:.2f}%")
    print(f"AA Accuracy:     {aa_acc*100:.2f}%")
    print("-" * 60)
    print(f"Results appended to: {csv_file}")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
