import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import timm
from tqdm import tqdm

# --- 1. CONFIGURATION ---
BATCH_SIZE = 64          # Reduced batch size since running both teacher and student
LR = 3e-4                # Lower learning rate for the student
EPOCHS = 10              # For demo (real training might need 50-100)
TEMPERATURE = 4.0        # Softmax temperature (T)
ALPHA = 0.5              # Weight: 0.5 for Soft Loss, 0.5 for Hard Loss
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Distilling ResNet-152 -> ViT-Tiny on {DEVICE}")
print(f"Temperature={TEMPERATURE}, Alpha={ALPHA}")

# --- 2. DATA LOADERS (CIFAR-100) ---
# Using 224x224 for both models
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
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

# --- 3. LOAD MODELS ---

# A. Load TEACHER (ResNet-152, Frozen)
print("Loading Teacher (ResNet-152)...")
teacher = timm.create_model('resnet152', pretrained=False, num_classes=100)
teacher.load_state_dict(torch.load("resnet152_cifar100_teacher.pth"))
teacher = teacher.to(DEVICE)
teacher.eval()  # CRITICAL: Set to eval mode

# Freeze Teacher parameters
for param in teacher.parameters():
    param.requires_grad = False

# B. Load STUDENT (ViT-Tiny, Trainable)
print("Loading Student (ViT-Tiny)...")
student = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=100)
student = student.to(DEVICE)
student.train()

# Compare sizes
t_params = sum(p.numel() for p in teacher.parameters())
s_params = sum(p.numel() for p in student.parameters())
print(f"Teacher (ResNet-152) Parameters: {t_params:,}")
print(f"Student (ViT-Tiny) Parameters: {s_params:,}")
print(f"Compression Ratio: {t_params/s_params:.1f}x")
print("Cross-Architecture Distillation: CNN -> Transformer")

# --- 4. LOSS FUNCTION ---
class DistillationLoss(nn.Module):
    def __init__(self, temperature, alpha):
        super(DistillationLoss, self).__init__()
        self.T = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # Soft Targets (Knowledge Transfer)
        student_soft = F.log_softmax(student_logits / self.T, dim=1)
        teacher_soft = F.log_softmax(teacher_logits / self.T, dim=1)
        
        # Calculate KL Divergence and scale gradients by T^2
        soft_loss = self.kl_div(student_soft, teacher_soft) * (self.T ** 2)

        # Hard Targets (Standard Classification)
        hard_loss = self.ce_loss(student_logits, labels)

        # Weighted Sum
        return (self.alpha * soft_loss) + ((1 - self.alpha) * hard_loss)

criterion = DistillationLoss(TEMPERATURE, ALPHA)
optimizer = optim.AdamW(student.parameters(), lr=LR)

# --- 5. TRAINING LOOP ---
best_acc = 0.0

for epoch in range(EPOCHS):
    student.train()
    loop = tqdm(trainloader, desc=f"Distilling Epoch {epoch+1}/{EPOCHS}")
    
    train_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in loop:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        # 1. Teacher Prediction (No Grad)
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        
        # 2. Student Prediction
        optimizer.zero_grad()
        student_logits = student(inputs)
        
        # 3. Calculate Distillation Loss
        loss = criterion(student_logits, teacher_logits, labels)
        
        # 4. Backprop
        loss.backward()
        optimizer.step()
        
        # Stats
        train_loss += loss.item()
        _, predicted = student_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        loop.set_postfix(acc=100.*correct/total, loss=loss.item())

    # --- VALIDATION LOOP ---
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
    
    acc = 100.*val_correct/val_total
    print(f"Epoch {epoch+1} Validation Accuracy: {acc:.2f}%")
    
    if acc > best_acc:
        best_acc = acc
        torch.save(student.state_dict(), "vit_tiny_cifar100_distilled_from_resnet.pth")

print(f"\nBest Distilled ViT-Tiny Accuracy: {best_acc:.2f}%")
print("Distilled Student Saved as 'vit_tiny_cifar100_distilled_from_resnet.pth'!")
