import torch
import torch.nn as nn
import numpy as np
try:
    from autoattack import AutoAttack
except ImportError:
    AutoAttack = None

def attack_pgd(model, X, y, attack_iters=20, step_size=2.0/255.0, epsilon=8.0/255.0):
    ce_loss = nn.CrossEntropyLoss()
    model.eval()
    
    # Random start
    X_pgd = X.detach() + torch.zeros_like(X).uniform_(-epsilon, epsilon)
    X_pgd = torch.clamp(X_pgd, 0, 1)
    
    for _ in range(attack_iters):
        X_pgd.requires_grad_()
        with torch.enable_grad():
            with torch.amp.autocast('cuda'):
                logits = model(X_pgd)
                loss = ce_loss(logits, y)
        
        grad = torch.autograd.grad(loss.float(), [X_pgd])[0]
        X_pgd = X_pgd.detach() + step_size * torch.sign(grad.detach())
        X_pgd = torch.min(torch.max(X_pgd, X - epsilon), X + epsilon)
        X_pgd = torch.clamp(X_pgd, 0, 1)
        
    return X_pgd

def attack_fgsm(model, X, y, epsilon=8.0/255.0):
    ce_loss = nn.CrossEntropyLoss()
    model.eval()
    
    X_fgsm = X.detach().clone()
    X_fgsm.requires_grad_()
    
    with torch.enable_grad():
        with torch.amp.autocast('cuda'):
            logits = model(X_fgsm)
            loss = ce_loss(logits, y)
        
    grad = torch.autograd.grad(loss.float(), [X_fgsm])[0]
    X_fgsm = X_fgsm.detach() + epsilon * torch.sign(grad.detach())
    X_fgsm = torch.clamp(X_fgsm, 0, 1)
        
    return X_fgsm

def attack_cw_linf(model, X, y, attack_iters=20, step_size=2.0/255.0, epsilon=8.0/255.0):
    model.eval()
    X_cw = X.detach() + torch.zeros_like(X).uniform_(-epsilon, epsilon)
    X_cw = torch.clamp(X_cw, 0, 1)
    
    for _ in range(attack_iters):
        X_cw.requires_grad_()
        with torch.enable_grad():
            with torch.amp.autocast('cuda'):
                logits = model(X_cw)
                target_logits = logits[torch.arange(X.shape[0]), y]
                logits_without_target = logits.clone()
                logits_without_target[torch.arange(X.shape[0]), y] = -1e4
                max_other_logits = logits_without_target.max(dim=1)[0]
                # C&W Margin Loss
                loss = torch.clamp(max_other_logits - target_logits, min=-50.0).sum()
            
        grad = torch.autograd.grad(loss.float(), [X_cw])[0]
        X_cw = X_cw.detach() + step_size * torch.sign(grad.detach())
        X_cw = torch.min(torch.max(X_cw, X - epsilon), X + epsilon)
        X_cw = torch.clamp(X_cw, 0, 1)
        
    return X_cw

def eval_robustness(model, testloader, device):
    model.eval()
    test_accs = []
    test_accs_adv = []
    test_accs_fgsm = []
    test_accs_cw = []
    
    for X, y in testloader:
        X, y = X.to(device), y.to(device)
        
        # --- Clean accuracy ---
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                preds = model(X).argmax(1)
        test_accs.append((preds == y).cpu().numpy())
        del preds
        
        # --- PGD-20 (sequential: generate → infer → free) ---
        X_pgd = attack_pgd(model, X, y, attack_iters=20, step_size=2.0/255.0, epsilon=8.0/255.0)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                preds_pgd = model(X_pgd).argmax(1)
        test_accs_adv.append((preds_pgd == y).cpu().numpy())
        del X_pgd, preds_pgd
        torch.cuda.empty_cache()
        
        # --- FGSM (sequential) ---
        X_fgsm = attack_fgsm(model, X, y, epsilon=8.0/255.0)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                preds_fgsm = model(X_fgsm).argmax(1)
        test_accs_fgsm.append((preds_fgsm == y).cpu().numpy())
        del X_fgsm, preds_fgsm
        torch.cuda.empty_cache()
        
        # --- C&W (sequential) ---
        X_cw = attack_cw_linf(model, X, y, attack_iters=20, step_size=2.0/255.0, epsilon=8.0/255.0)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                preds_cw = model(X_cw).argmax(1)
        test_accs_cw.append((preds_cw == y).cpu().numpy())
        del X_cw, preds_cw
        torch.cuda.empty_cache()
        
    clean_acc = np.mean(np.concatenate(test_accs))
    robust_acc_pgd = np.mean(np.concatenate(test_accs_adv))
    robust_acc_fgsm = np.mean(np.concatenate(test_accs_fgsm))
    robust_acc_cw = np.mean(np.concatenate(test_accs_cw))
    
    # AutoAttack
    if AutoAttack is not None:
        autoattack = AutoAttack(model, norm='Linf', eps=8/255.0, version='standard', device=device)
        autoattack.seed = 0
        
        # Collect all batches on CPU to avoid VRAM spike
        x_total = []
        y_total = []
        for X, y in testloader:
            x_total.append(X)
            y_total.append(y)
        x_total = torch.cat(x_total, 0)
        y_total = torch.cat(y_total, 0)
        
        # Run AA with small batch size to limit VRAM
        x_adv_aa = autoattack.run_standard_evaluation(x_total, y_total, bs=250)
        
        with torch.no_grad():
            from torch.utils.data import TensorDataset
            aa_loader = torch.utils.data.DataLoader(TensorDataset(x_adv_aa, y_total), batch_size=250, shuffle=False)
            aa_correct = 0
            for X_aa, y_aa in aa_loader:
                X_aa, y_aa = X_aa.to(device), y_aa.to(device)
                with torch.amp.autocast('cuda'):
                    logits_aa = model(X_aa)
                aa_correct += (logits_aa.argmax(1) == y_aa).sum().item()
                del X_aa, y_aa, logits_aa
                torch.cuda.empty_cache()
            robust_acc_aa = aa_correct / len(y_total)
    else:
        print("AutoAttack library not found. Returning 0.0 for AA accuracy.")
        robust_acc_aa = 0.0
    
    return clean_acc, robust_acc_pgd, robust_acc_fgsm, robust_acc_cw, robust_acc_aa
