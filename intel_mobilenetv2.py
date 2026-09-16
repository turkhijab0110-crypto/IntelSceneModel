import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split


# ============================================================
# 1. SETTINGS
# ============================================================

TRAIN_DIR = r"C:\Users\turkh\ComputerVision\intel_dataset\seg_train\seg_train"
TEST_DIR = r"C:\Users\turkh\ComputerVision\intel_dataset\seg_test\seg_test"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ============================================================
# 2. IMAGE TRANSFORMS
# ============================================================

# Training images:
# Resize + augmentation + convert to tensor + normalization
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Validation/test images:
# Only resize + tensor + normalization
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 3. LOAD DATASET
# ============================================================

# Load the training dataset
full_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

# Load the separate test dataset
test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

print("Classes:", full_dataset.classes)
print("Total training images:", len(full_dataset))
print("Total test images:", len(test_dataset))


# ============================================================
# 4. SPLIT TRAINING DATA INTO TRAIN + VALIDATION
# ============================================================

# 80% training, 20% validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=generator
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# ============================================================
# 5. CREATE DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 6. LOAD PRETRAINED MOBILENETV2
# ============================================================

print("\nLoading pretrained MobileNetV2...")

weights = models.MobileNet_V2_Weights.DEFAULT

model = models.mobilenet_v2(weights=weights)


# ============================================================
# 7. FREEZE PRETRAINED LAYERS
# ============================================================


for parameter in model.features.parameters():
    parameter.requires_grad = False


# ============================================================
# 8. REPLACE THE CLASSIFIER
# ============================================================

number_of_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    number_of_features,
    6
)

# Add dropout of 0.30, as used in the project methodology
model.classifier = nn.Sequential(
    nn.Dropout(0.30),
    nn.Linear(number_of_features, 6)
)

model = model.to(device)

print("\nModel:")
print(model.classifier)


# ============================================================
# 9. LOSS FUNCTION AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

# Only train the new classifier
optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 10. TRAINING
# ============================================================

best_val_accuracy = 0.0

model_save_path = "best_mobilenetv2.pth"


for epoch in range(EPOCHS):

    # -------------------------
    # TRAINING
    # -------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

        # Find predicted class
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total


    # -------------------------
    # VALIDATION
    # -------------------------

    model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss_total += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss_total / len(val_loader)
    val_accuracy = 100 * val_correct / val_total


    # -------------------------
    # PRINT RESULTS
    # -------------------------

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )


    # -------------------------
    # SAVE BEST MODEL
    # -------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": full_dataset.classes
            },
            model_save_path
        )

        print("  --> Best model saved!")


# ============================================================
# 11. LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

checkpoint = torch.load(
    model_save_path,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


# ============================================================
# 12. FINAL TESTING
# ============================================================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()


test_accuracy = 100 * correct / total

print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")

print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
print("Model saved as:", model_save_path)

print("\nTraining completed!")