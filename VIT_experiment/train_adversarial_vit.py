import os
import argparse
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_cifar100_dataloaders
from models import get_vit_small_student
from attacks import eval_robustness

def pgd_attack(model, X, y, epsilon=8/255.0, step_size=2/255.0, steps=10):
    """PGD-10 attack for adversarial training (generates adv examples on-the-fly)."""
    model.eval()
    ce_loss = nn.CrossEntropyLoss()

    X_adv = X.detach() + torch.zeros_like(X).uniform_(-epsilon, epsilon)
    X_adv = torch.clamp(X_adv, 0, 1)

    for _ in range(steps):
        X_adv.requires_grad_(True)
        with torch.enable_grad():
            loss = ce_loss(model(X_adv), y)
        grad = torch.autograd.grad(loss, X_adv)[0]
        X_adv = X_adv.detach() + step_size * torch.sign(grad.detach())
        X_adv = torch.min(torch.max(X_adv, X - epsilon), X + epsilon)
        X_adv = torch.clamp(X_adv, 0, 1)

    model.train()
    return X_adv.detach()


def evaluate_and_log(model, testloader, device, label, results_list):
    """Run full robustness evaluation and append results to the list."""
    print(f"\n{'='*60}")
    print(f"📊 Evaluating: {label}")
    print(f"{'='*60}")

    clean_acc, pgd_acc, fgsm_acc, cw_acc, aa_acc = eval_robustness(model, testloader, device)

    print(f"  Clean Acc:  {clean_acc*100:.2f}%")
    print(f"  FGSM Acc:   {fgsm_acc*100:.2f}%")
    print(f"  PGD-20 Acc: {pgd_acc*100:.2f}%")
    print(f"  C&W Acc:    {cw_acc*100:.2f}%")
    print(f"  AA Acc:     {aa_acc*100:.2f}%")
    print(f"{'='*60}\n")

    results_list.append({
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'arch': 'VIT_SMALL',
        'method': label,
        'batch_size': '-',
        'epochs': '-',
        'clean': f"{clean_acc*100:.2f}",
        'fgsm': f"{fgsm_acc*100:.2f}",
        'pgd20': f"{pgd_acc*100:.2f}",
        'cw': f"{cw_acc*100:.2f}",
        'aa': f"{aa_acc*100:.2f}",
    })
    return results_list


def main():
    parser = argparse.ArgumentParser(description='Adversarial Training for ViT-Small on CIFAR-100')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate (lower for pretrained)')
    parser.add_argument('--pgd-steps', type=int, default=10, help='PGD steps for adversarial training')
    parser.add_argument('--epsilon', type=float, default=8/255.0, help='Perturbation budget')
    parser.add_argument('--step-size', type=float, default=2/255.0, help='PGD step size')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    args = parser.parse_args()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use 32x32 images — the ViTStudentWrapper internally upsamples to 224x224
    trainloader, testloader = get_cifar100_dataloaders(batch_size=args.batch_size, img_size=32)

    # Load pretrained ViT-Small
    model = get_vit_small_student(pretrained=True).to(device)
    print(f"Loaded pretrained ViT-Small (vit_small_patch16_224) with {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    results = []

    # ─── Phase 1: Evaluate BEFORE adversarial training ───
    results = evaluate_and_log(model, testloader, device, "Before AT (Pretrained)", results)

    # ─── Phase 2: Adversarial Training with PGD-10 ───
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    ce_loss = nn.CrossEntropyLoss()

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

    print(f"\nStarting Adversarial Training: PGD-{args.pgd_steps}, ε={args.epsilon:.4f}, epochs={args.epochs}")
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch, args)
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for step, (X, y) in enumerate(trainloader):
            X, y = X.to(device), y.to(device)

            # Generate PGD adversarial examples
            X_adv = pgd_attack(model, X, y, epsilon=args.epsilon,
                               step_size=args.step_size, steps=args.pgd_steps)

            # Train on adversarial examples
            model.train()
            optimizer.zero_grad()
            logits = model(X_adv)
            loss = ce_loss(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

            if step % 50 == 0:
                print(f"Epoch {epoch}/{args.epochs} Step {step}/{len(trainloader)} "
                      f"Loss: {loss.item():.4f} Acc: {100*correct/total:.1f}%")

        # Quick clean accuracy check (skip full robust eval until last epoch)
        if epoch == args.epochs:
            pass  # Full eval done below
        else:
            model.eval()
            eval_correct = 0
            with torch.no_grad():
                for X_test, y_test in testloader:
                    X_test, y_test = X_test.to(device), y_test.to(device)
                    eval_correct += (model(X_test).argmax(1) == y_test).sum().item()
            print(f"Epoch {epoch} — Clean Acc: {100*eval_correct/len(testloader.dataset):.2f}% "
                  f"(full robust eval at final epoch)")

    # Save model
    save_name = f"vit_small_adversarial_pgd{args.pgd_steps}_epochs_{args.epochs}.pt"
    torch.save(model.state_dict(), os.path.join('result_models', save_name))
    print(f"\nSaved adversarially trained model to result_models/{save_name}")

    # ─── Phase 3: Evaluate AFTER adversarial training ───
    # Update the batch_size/epochs in the "After" row
    results = evaluate_and_log(model, testloader, device, f"After AT (PGD-{args.pgd_steps})", results)
    results[-1]['batch_size'] = args.batch_size
    results[-1]['epochs'] = args.epochs

    # ─── Write CSV ───
    csv_file = os.path.join('result_models', 'adversarial_training_results.csv')
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'Date', 'Architecture', 'Method', 'Batch Size', 'Epochs',
                'Clean Acc (%)', 'FGSM Acc (%)', 'PGD20 Acc (%)', 'C&W Acc (%)', 'AA Acc (%)'
            ])
        for row in results:
            writer.writerow([
                row['date'], row['arch'], row['method'],
                row['batch_size'], row['epochs'],
                row['clean'], row['fgsm'], row['pgd20'], row['cw'], row['aa']
            ])

    # Print final summary
    print("\n" + "="*60)
    print("📊 ADVERSARIAL TRAINING RESULTS SUMMARY")
    print("="*60)
    for row in results:
        print(f"\n  [{row['method']}]")
        print(f"    Clean: {row['clean']}%  |  FGSM: {row['fgsm']}%  |  PGD-20: {row['pgd20']}%  |  C&W: {row['cw']}%  |  AA: {row['aa']}%")
    print(f"\nResults saved to: {csv_file}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
