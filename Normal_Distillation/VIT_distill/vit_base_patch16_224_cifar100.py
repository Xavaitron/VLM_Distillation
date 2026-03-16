import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import timm
from tqdm import tqdm

# 1. SETUP & HYPERPARAMETERS
BATCH_SIZE = 64
LR_HEAD = 0.003      # High LR for the head
LR_BODY = 1e-5       # Low LR for fine-tuning the body later
EPOCHS_HEAD = 1      # Warmup epochs (Head only)
EPOCHS_FINE = 3      # Fine-tuning epochs (Whole body)
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# 2. DATA PREPARATION (CIFAR-100)
# Still resizing to 224x224 for the ViT
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # Added simple augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]) # CIFAR-100 stats
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
])

# Change dataset to CIFAR100
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                         download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

# 3. MODEL SETUP
# Change num_classes to 100
print("Loading ViT Base (CIFAR-100)...")
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=100)
model = model.to(DEVICE)

# 4. PHASE 1: LINEAR PROBE (Train Head Only)
print("--- Phase 1: Warming up the Head ---")
for param in model.parameters():
    param.requires_grad = False
for param in model.head.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.head.parameters(), lr=LR_HEAD)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS_HEAD):
    model.train()
    loop = tqdm(trainloader, desc=f"Head Epoch {epoch+1}/{EPOCHS_HEAD}")
    for inputs, labels in loop:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

# 5. PHASE 2: FINE-TUNING (Train Everything)
# For CIFAR-100, you really need this step.
print("\n--- Phase 2: Fine-Tuning the Whole Model ---")
for param in model.parameters():
    param.requires_grad = True # Unfreeze everything

# Lower learning rate significantly for the backbone
optimizer = optim.Adam(model.parameters(), lr=LR_BODY)

for epoch in range(EPOCHS_FINE):
    model.train()
    loop = tqdm(trainloader, desc=f"Fine-Tune Epoch {epoch+1}/{EPOCHS_FINE}")
    correct = 0
    total = 0
    
    for inputs, labels in loop:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        loop.set_postfix(acc=100.*correct/total, loss=loss.item())

# Save the Teacher
torch.save(model.state_dict(), "vit_base_cifar100_teacher.pth")
print("Teacher saved!")