import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import timm
from tqdm import tqdm
import os

# 1. SETUP & HYPERPARAMETERS
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# 2. DATA PREPARATION (CIFAR-100)
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
])

print("Preparing Data...")
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

# 3. MODEL SETUP
print("Loading ViT Base (CIFAR-100)...")
model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=100)

# Load the trained weights
model_path = "vit_base_cifar100_teacher.pth"
if not os.path.exists(model_path):
    print(f"Error: {model_path} not found. Please make sure the model is trained and saved.")
    exit(1)

model.load_state_dict(torch.load(model_path, map_location=DEVICE))
print(f"Loaded weights from {model_path}")

model = model.to(DEVICE)
model.eval()

# 4. EVALUATION
correct = 0
total = 0
criterion = nn.CrossEntropyLoss()
total_loss = 0.0

print("\n--- Starting Evaluation ---")
with torch.no_grad():
    loop = tqdm(testloader, desc="Testing")
    for inputs, labels in loop:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        loop.set_postfix(acc=100.*correct/total, loss=loss.item())

accuracy = 100. * correct / total
avg_loss = total_loss / len(testloader)

print(f"\nTest Accuracy: {accuracy:.2f}%")
print(f"Test Loss: {avg_loss:.4f}")
