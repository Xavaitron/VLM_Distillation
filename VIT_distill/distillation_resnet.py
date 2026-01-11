import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import timm
from tqdm import tqdm
import os

# --- 1. CONFIGURATION ---
BATCH_SIZE = 64         # Batch size 
LR = 0.001              # Learning Rate for Student
EPOCHS = 10             # Number of epochs
TEMPERATURE = 4.0       # Distillation Temperature (T) 
ALPHA = 0.5             # Weighting: 50% Soft Targets, 50% Hard Targets 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running Cross-Architecture Distillation (ViT -> ResNet)")
print(f"Device: {DEVICE} | T={TEMPERATURE} | Alpha={ALPHA}")

# --- 2. DATA TRANSFORMS ---
# We resize to 224x224 so the Student sees exactly what the Teacher sees.
# This ensures the features align perfectly.

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# FIX: Define transform_test explicitly before using it
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 3. DATASET & LOADERS ---
# We use num_workers=2 to speed up data loading
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, 
                                         download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, 
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, 
                                        download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, 
                                         shuffle=False, num_workers=2)

# --- 4. MODEL SETUP ---

# A. LOAD TEACHER (ViT Base)
# We must load the architecture first, then the weights
print("\nLoading Teacher (ViT-Base)...")
teacher = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=100)

# CHECK: Ensure the teacher weights file exists
teacher_weights = "vit_base_cifar100_teacher.pth"
if not os.path.exists(teacher_weights):
    raise FileNotFoundError(f"Could not find {teacher_weights}. Please run the teacher training script first!")

teacher.load_state_dict(torch.load(teacher_weights))
teacher = teacher.to(DEVICE)
teacher.eval() # Set to evaluation mode

# Freeze Teacher (We don't want to update the teacher, only learn from it)
for param in teacher.parameters():
    param.requires_grad = False

# B. CREATE STUDENT (ResNet18)
print("Loading Student (ResNet18)...")
student = timm.create_model('resnet18', pretrained=True, num_classes=100)
student = student.to(DEVICE)
student.train() # Set to training mode

# --- 5. DISTILLATION LOSS FUNCTION ---
class DistillationLoss(nn.Module):
    def __init__(self, temperature, alpha):
        super(DistillationLoss, self).__init__()
        self.T = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # 1. Soft Targets (Knowledge Transfer)
        # Scale by Temperature T to soften the distribution 
        student_soft = F.log_softmax(student_logits / self.T, dim=1)
        teacher_soft = F.log_softmax(teacher_logits / self.T, dim=1)
        
        # Calculate KL Divergence
        # We multiply by T^2 to scale gradients properly 
        soft_loss = self.kl_div(student_soft, teacher_soft) * (self.T ** 2)

        # 2. Hard Targets (Standard Classification)
        hard_loss = self.ce_loss(student_logits, labels)

        # 3. Weighted Combination
        return (self.alpha * soft_loss) + ((1 - self.alpha) * hard_loss)

criterion = DistillationLoss(TEMPERATURE, ALPHA)
optimizer = optim.Adam(student.parameters(), lr=LR)

# --- 6. TRAINING LOOP ---
print(f"\nStarting Distillation for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    student.train()
    loop = tqdm(trainloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    train_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in loop:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        # 1. Get Teacher Predictions (No Gradients needed)
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        
        # 2. Get Student Predictions
        optimizer.zero_grad()
        student_logits = student(inputs)
        
        # 3. Calculate Loss (Distillation)
        loss = criterion(student_logits, teacher_logits, labels)
        
        # 4. Backpropagation
        loss.backward()
        optimizer.step()
        
        # Stats update
        train_loss += loss.item()
        _, predicted = student_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        loop.set_postfix(acc=f"{100.*correct/total:.2f}%", loss=f"{loss.item():.4f}")

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
    
    val_acc = 100.*val_correct/val_total
    print(f"Epoch {epoch+1} Validation Accuracy: {val_acc:.2f}%")

# --- 7. SAVE MODEL ---
save_path = "resnet18_cifar100_distilled.pth"
torch.save(student.state_dict(), save_path)
print(f"\nDistillation Complete! Model saved to {save_path}")