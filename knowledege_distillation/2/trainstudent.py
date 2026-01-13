import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -------------------
# CONFIG
# -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
NUM_CLASSES = 10

T =3.0  # temperature
ALPHA = 0.7   # weight for KD loss

# -------------------
# DATA
# -------------------
transform = transforms.Compose([
    transforms.Resize(224),
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
# MODELS
# -------------------

# ---- Teacher (load trained weights)
teacher = timm.create_model("resnet18", pretrained=False, num_classes=NUM_CLASSES)
teacher.load_state_dict(torch.load("teacher_cifar10.pth"))
teacher = teacher.to(DEVICE)

# freeze teacher
for p in teacher.parameters():
    p.requires_grad = False
teacher.eval()

# ---- Student
student = timm.create_model("mobilenetv2_100", pretrained=True, num_classes=NUM_CLASSES)
student = student.to(DEVICE)

# -------------------
# LOSSES
# -------------------
ce_loss = nn.CrossEntropyLoss()
kl_loss = nn.KLDivLoss(reduction="batchmean")

# -------------------
# OPTIMIZER
# -------------------
optimizer = optim.Adam(student.parameters(), lr=LR)

# -------------------
# DISTILLATION LOSS
# -------------------
def distillation_loss(student_logits, teacher_logits, labels):
    # Hard loss (true labels)
    loss_ce = ce_loss(student_logits, labels)

    # Soft loss (teacher guidance)
    s_log_probs = F.log_softmax(student_logits / T, dim=1)
    t_probs     = F.softmax(teacher_logits / T, dim=1)

    loss_kd = kl_loss(s_log_probs, t_probs) * (T * T)

    return ALPHA * loss_kd + (1 - ALPHA) * loss_ce

# -------------------
# TRAIN
# -------------------
def train_student():
    student.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0

        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            with torch.no_grad():
                teacher_logits = teacher(images)

            student_logits = student(images)

            loss = distillation_loss(student_logits, teacher_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[Student KD] Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(trainloader):.4f}")

# -------------------
# TEST
# -------------------
def test_student():
    student.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = student(images)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = 100 * correct / total
    print(f"[Student KD] Test Accuracy: {acc:.2f}%")
    return acc

# -------------------
# RUN
# -------------------
train_student()
test_student()
