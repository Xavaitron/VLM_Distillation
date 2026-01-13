import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import timm
from tqdm import tqdm

# --- 1. CONFIGURATION ---
BATCH_SIZE = 128
LR = 3e-4               # Same LR as distillation for fair comparison
EPOCHS = 10             # Same Epochs as distillation
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

print(f"Training ResNet-18 Baseline on {DEVICE}...")

# --- 2. DATA LOADERS (CIFAR-100) ---
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=28),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- 3. MODEL SETUP (ResNet-18 Student) ---
print("Loading ResNet-18 (Baseline)...")
model = timm.create_model('resnet18', pretrained=True, num_classes=100)
model = model.to(DEVICE)

# Print model info
total_params = sum(p.numel() for p in model.parameters())
print(f"ResNet-18 Parameters: {total_params:,}")

# --- 4. OPTIMIZER & LOSS ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

# --- 5. TRAINING LOOP ---
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    loop = tqdm(trainloader, desc=f"Baseline Epoch {epoch+1}/{EPOCHS}")
    
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
    print(f"Epoch {epoch+1} Validation Accuracy: {acc:.2f}%")
    
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "resnet18_cifar100_baseline.pth")

print(f"Best Baseline Accuracy: {best_acc:.2f}%")
print("Baseline Model Saved!")
