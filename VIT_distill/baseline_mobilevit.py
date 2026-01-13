import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import timm
from tqdm import tqdm

# --- 1. CONFIGURATION ---
BATCH_SIZE = 64          # MobileViT uses more memory per sample due to architecture
LR = 3e-4                # Same LR as other experiments for fair comparison
EPOCHS = 10              # Same Epochs as distillation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training MobileViT Baseline on {DEVICE}...")

# --- 2. DATA LOADERS (CIFAR-100) ---
# MobileViT expects 256x256 input by default, but 224x224 works well too
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(256, padding=32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
])

transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- 3. MODEL SETUP (MobileViT) ---
print("Loading MobileViT-S (Small) as Baseline...")
# Available MobileViT variants in timm:
# - mobilevit_xxs: ~1.3M params (extra extra small)
# - mobilevit_xs:  ~2.3M params (extra small)
# - mobilevit_s:   ~5.6M params (small)
model = timm.create_model('mobilevit_s', pretrained=True, num_classes=100)
model = model.to(DEVICE)

# Print model info
total_params = sum(p.numel() for p in model.parameters())
print(f"MobileViT-S Parameters: {total_params:,}")

# --- 4. OPTIMIZER & LOSS ---
# Standard Cross Entropy (Hard Targets Only)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

# Learning rate scheduler for better convergence
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# --- 5. TRAINING LOOP ---
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    loop = tqdm(trainloader, desc=f"MobileViT Baseline Epoch {epoch+1}/{EPOCHS}")
    
    train_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in loop:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward pass (No Teacher involved)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        # Stats
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        loop.set_postfix(acc=100.*correct/total, loss=loss.item())

    # Step the scheduler
    scheduler.step()
    
    # --- VALIDATION LOOP ---
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    acc = 100.*val_correct/val_total
    print(f"Epoch {epoch+1} Validation Accuracy: {acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "mobilevit_s_cifar100_baseline.pth")

print(f"Best MobileViT Baseline Accuracy: {best_acc:.2f}%")
print("Baseline Model Saved!")
