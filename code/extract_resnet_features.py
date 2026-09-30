from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PATCH_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "patch_dataset"
)

TRAIN_DIR = PATCH_ROOT / "train"
VAL_DIR = PATCH_ROOT / "validation"

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "resnet18"
    / "best_resnet18_8epochs.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "resnet_features"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_FEATURES_PATH = (
    OUTPUT_DIR / "train_features.npy"
)

TRAIN_LABELS_PATH = (
    OUTPUT_DIR / "train_labels.npy"
)

VAL_FEATURES_PATH = (
    OUTPUT_DIR / "validation_features.npy"
)

VAL_LABELS_PATH = (
    OUTPUT_DIR / "validation_labels.npy"
)

TRAIN_INFO_PATH = (
    OUTPUT_DIR / "train_patch_info.csv"
)

VAL_INFO_PATH = (
    OUTPUT_DIR / "validation_patch_info.csv"
)

BATCH_SIZE = 16

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# TRANSFORM
#
# IMPORTANT:
# No random augmentation during feature extraction.
# Every patch must produce a deterministic feature vector.
# ============================================================

feature_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=feature_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=feature_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print("=" * 72)
print("RESNET18 DEEP FEATURE EXTRACTION")
print("=" * 72)

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


# ============================================================
# LOAD SAVED CHECKPOINT
# ============================================================

print("\nLoading saved ResNet18 checkpoint...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

print(
    f"Checkpoint best epoch: "
    f"{checkpoint['best_epoch']}"
)

print(
    f"Checkpoint best validation accuracy: "
    f"{checkpoint['best_val_accuracy']:.2f}%"
)


# ============================================================
# RECREATE RESNET18
# ============================================================

# No need to download ImageNet weights here because
# the trained checkpoint contains the model parameters.

model = resnet18(
    weights=None
)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    3
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


# ============================================================
# REMOVE FINAL CLASSIFIER
#
# ResNet18:
#
# convolutional backbone
#       ↓
# global average pooling
#       ↓
# 512-dimensional vector
#       ↓
# fc classifier
#
# Replacing fc with Identity means model(image)
# returns the 512-dimensional feature vector.
# ============================================================

model.fc = nn.Identity()

model = model.to(device)

model.eval()

print(
    f"\nFinal classification layer removed."
)

print(
    f"Feature dimension: {num_features}"
)


# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(loader):

    all_features = []
    all_labels = []

    with torch.no_grad():

        for batch_number, (images, labels) in enumerate(
            loader,
            start=1
        ):

            images = images.to(device)

            features = model(images)

            features = (
                features
                .cpu()
                .numpy()
            )

            all_features.append(
                features
            )

            all_labels.append(
                labels.numpy()
            )

            if (
                batch_number % 10 == 0
                or batch_number == len(loader)
            ):

                processed = min(
                    batch_number * BATCH_SIZE,
                    len(loader.dataset)
                )

                print(
                    f"  Processed "
                    f"{processed}/"
                    f"{len(loader.dataset)}"
                )

    all_features = np.concatenate(
        all_features,
        axis=0
    )

    all_labels = np.concatenate(
        all_labels,
        axis=0
    )

    return (
        all_features,
        all_labels
    )


# ============================================================
# EXTRACT TRAINING FEATURES
# ============================================================

print("\n" + "=" * 72)
print("EXTRACTING TRAINING FEATURES")
print("=" * 72)

train_features, train_labels = (
    extract_features(
        train_loader
    )
)


# ============================================================
# EXTRACT VALIDATION FEATURES
# ============================================================

print("\n" + "=" * 72)
print("EXTRACTING VALIDATION FEATURES")
print("=" * 72)

val_features, val_labels = (
    extract_features(
        val_loader
    )
)


# ============================================================
# VALIDATE SHAPES
# ============================================================

print("\n" + "=" * 72)
print("FEATURE SHAPES")
print("=" * 72)

print(
    f"\nTrain features: "
    f"{train_features.shape}"
)

print(
    f"Train labels:   "
    f"{train_labels.shape}"
)

print(
    f"Validation features: "
    f"{val_features.shape}"
)

print(
    f"Validation labels:   "
    f"{val_labels.shape}"
)


if train_features.shape[1] != 512:
    raise ValueError(
        "Expected 512 ResNet18 features."
    )

if val_features.shape[1] != 512:
    raise ValueError(
        "Expected 512 ResNet18 features."
    )


# ============================================================
# SAVE NUMPY ARRAYS
# ============================================================

np.save(
    TRAIN_FEATURES_PATH,
    train_features
)

np.save(
    TRAIN_LABELS_PATH,
    train_labels
)

np.save(
    VAL_FEATURES_PATH,
    val_features
)

np.save(
    VAL_LABELS_PATH,
    val_labels
)


# ============================================================
# SAVE PATCH INFORMATION
#
# This keeps the original filename associated with each
# feature vector. It will later allow us to aggregate patch
# predictions back to the original bundle/sample level.
# ============================================================

def save_patch_info(
    dataset,
    output_path
):

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "index",
            "patch_path",
            "patch_name",
            "class_name",
            "class_index"
        ])

        for index, (
            path,
            class_index
        ) in enumerate(
            dataset.samples
        ):

            path = Path(path)

            class_name = (
                dataset.classes[
                    class_index
                ]
            )

            writer.writerow([
                index,
                str(path),
                path.name,
                class_name,
                class_index
            ])


save_patch_info(
    train_dataset,
    TRAIN_INFO_PATH
)

save_patch_info(
    val_dataset,
    VAL_INFO_PATH
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 72)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 72)

print(
    f"\nTraining feature matrix:"
    f"\n{train_features.shape}"
)

print(
    f"\nValidation feature matrix:"
    f"\n{val_features.shape}"
)

print(
    "\nEach image patch is now represented "
    "by 512 ResNet18 deep features."
)

print(
    f"\nFiles saved in:\n"
    f"{OUTPUT_DIR}"
)

print("""
IMPORTANT:
No test images were loaded.

These train/validation features can now be used to build
and tune the SVM without touching the final test set.
""")