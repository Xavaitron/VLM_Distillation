import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import timm
from tqdm import tqdm

# --- CONFIGURATION ---
BATCH_SIZE = 128
LR = 0.001              # Standard starting LR for ResNet
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training Baseline CNN (ResNet18) on {DEVICE}...")

# --- DATA LOADERS ---
# We keep 224x224 so we can compare apples-to-apples with the distillation run later
# (The Teacher requires 224, so the Student must see 224 to learn from it)
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- MODEL (Student CNN) ---
# We use ResNet18. 
# pretrained=True gives it a head start (ImageNet features).
print("Loading Student (ResNet18)...")
student = timm.create_model('resnet18', pretrained=True, num_classes=100)
student = student.to(DEVICE)

# --- TRAINING LOOP (Standard) ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(student.parameters(), lr=LR)

for epoch in range(EPOCHS):
    student.train()
    loop = tqdm(trainloader, desc=f"Baseline Epoch {epoch+1}/{EPOCHS}")
    
    correct = 0
    total = 0
    
    for inputs, labels in loop:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = student(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        loop.set_postfix(acc=100.*correct/total, loss=loss.item())

    # Validation
    student.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = student(inputs)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    print(f"Validation Accuracy: {100.*val_correct/val_total:.2f}%")

# Save Baseline
torch.save(student.state_dict(), "resnet18_cifar100_baseline.pth")
print("Baseline CNN Saved.")