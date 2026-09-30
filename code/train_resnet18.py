from pathlib import Path
import copy
import csv
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "outputs" / "patch_dataset"

TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "validation"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "resnet18"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "best_resnet18_8epochs.pth"
HISTORY_PATH = OUTPUT_DIR / "training_history_8epochs.csv"

BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.001
NUM_CLASSES = 3

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

# Training augmentation is intentionally mild.
# We do NOT alter validation images except normalization.

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD TRAIN + VALIDATION ONLY
# ============================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)


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


print("=" * 70)
print("RESNET18 TRAINING")
print("=" * 70)

print(f"\nDevice: {device}")

print(
    f"Training patches: "
    f"{len(train_dataset)}"
)

print(
    f"Validation patches: "
    f"{len(val_dataset)}"
)

print(
    f"Classes: "
    f"{train_dataset.classes}"
)

print(
    f"Class mapping: "
    f"{train_dataset.class_to_idx}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)

print(
    f"Epochs: {EPOCHS}"
)

print(
    f"Learning rate: "
    f"{LEARNING_RATE}"
)


# ============================================================
# SAFETY CHECK
# ============================================================

if train_dataset.classes != val_dataset.classes:
    raise ValueError(
        "Train and validation class mappings do not match."
    )


# ============================================================
# LOAD PRETRAINED RESNET18
# ============================================================

weights = ResNet18_Weights.DEFAULT

model = resnet18(
    weights=weights
)


# ============================================================
# FREEZE MOST OF RESNET18
#
# For our first experiment:
# - pretrained convolutional backbone stays frozen
# - only final classification layer is trained
#
# This is safer for our small dataset and much faster on CPU.
# ============================================================

for parameter in model.parameters():
    parameter.requires_grad = False


# Replace ImageNet's 1000-class output with our 3 classes
num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    NUM_CLASSES
)

model = model.to(device)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0
best_epoch = 0
best_model_weights = copy.deepcopy(
    model.state_dict()
)

history = []

training_start = time.time()


for epoch in range(1, EPOCHS + 1):

    epoch_start = time.time()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0


    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss_sum += (
            loss.item()
            * images.size(0)
        )

        _, predictions = torch.max(
            outputs,
            1
        )

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)


    train_loss = (
        train_loss_sum
        / train_total
    )

    train_accuracy = (
        100.0
        * train_correct
        / train_total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss_sum += (
                loss.item()
                * images.size(0)
            )

            _, predictions = torch.max(
                outputs,
                1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)


    val_loss = (
        val_loss_sum
        / val_total
    )

    val_accuracy = (
        100.0
        * val_correct
        / val_total
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL IN MEMORY
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy
        best_epoch = epoch

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )


    epoch_time = (
        time.time()
        - epoch_start
    )


    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "epoch_seconds": epoch_time,
    })


    print(
        f"\nEpoch {epoch:02d}/{EPOCHS}"
    )

    print(
        f"  Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"  Val Loss:   {val_loss:.4f} | "
        f"Val Accuracy:   {val_accuracy:.2f}%"
    )

    print(
        f"  Time: {epoch_time:.1f} sec"
    )


# ============================================================
# RESTORE BEST VALIDATION MODEL
# ============================================================

model.load_state_dict(
    best_model_weights
)


# ============================================================
# SAVE MODEL CHECKPOINT
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),

        "class_to_idx":
            train_dataset.class_to_idx,

        "classes":
            train_dataset.classes,

        "best_epoch":
            best_epoch,

        "best_val_accuracy":
            best_val_accuracy,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "architecture":
            "ResNet18",

        "pretrained":
            True,

        "training_mode":
            "frozen_backbone_fc_only"
    },

    MODEL_PATH
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

with open(
    HISTORY_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "epoch_seconds"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(history)


# ============================================================
# FINAL SUMMARY
# ============================================================

total_time = (
    time.time()
    - training_start
)

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nBest epoch: "
    f"{best_epoch}"
)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Total training time: "
    f"{total_time / 60:.2f} minutes"
)

print(
    f"\nBest model saved to:\n"
    f"{MODEL_PATH}"
)

print(
    f"\nTraining history saved to:\n"
    f"{HISTORY_PATH}"
)

print("""
IMPORTANT:
The TEST dataset was not loaded or evaluated by this script.

Do not evaluate the test set yet.
""")