import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -------------------
# CONFIG
# -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 30          # train properly
LR = 1e-3
NUM_CLASSES = 10

# -------------------
# DATA
# -------------------
transform = transforms.Compose([
    transforms.Resize(96),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

trainset = datasets.CIFAR10(root="./data", train=True,
                            transform=transform, download=True)
testset  = datasets.CIFAR10(root="./data", train=False,
                            transform=transform, download=True)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader  = DataLoader(testset, batch_size=BATCH_SIZE)

# -------------------
# MODEL (TEACHER)
# -------------------
teacher = timm.create_model("resnet18", pretrained=True, num_classes=NUM_CLASSES)
teacher = teacher.to(DEVICE)

# -------------------
# LOSS + OPTIM
# -------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(teacher.parameters(), lr=LR)

# -------------------
# TRAIN
# -------------------
def train_teacher():
    teacher.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        print("Starting training...")

        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = teacher(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[Teacher] Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(trainloader):.4f}")

# -------------------
# TEST
# -------------------
def test_teacher():
    teacher.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = teacher(images)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100 * correct / total
    print(f"[Teacher] Test Accuracy: {acc:.2f}%")

# -------------------
# RUN
# -------------------
train_teacher()
test_teacher()

# -------------------
# SAVE TEACHER
# -------------------
torch.save(teacher.state_dict(), "teacher_cifar10.pth")
print("Teacher model saved as teacher_cifar10.pth")
