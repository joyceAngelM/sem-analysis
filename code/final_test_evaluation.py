from pathlib import Path
import csv
import re
from collections import defaultdict

import joblib
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "patch_dataset"
    / "test"
)

RESNET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "resnet18"
    / "best_resnet18_8epochs.pth"
)

SVM_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "svm"
    / "best_svm.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "final_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PATCH_RESULTS_PATH = (
    OUTPUT_DIR
    / "test_patch_predictions.csv"
)

SAMPLE_RESULTS_PATH = (
    OUTPUT_DIR
    / "test_sample_predictions.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "final_test_summary.txt"
)

BATCH_SIZE = 16

CLASS_NAMES = [
    "CR_0",
    "CR_30",
    "CR_41"
]

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# DETERMINISTIC TEST TRANSFORM
# ============================================================

test_transform = transforms.Compose([
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
# LOAD TEST DATA
# ============================================================

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print("=" * 72)
print("FINAL UNTOUCHED TEST EVALUATION")
print("=" * 72)

print(f"\nDevice: {device}")

print(
    f"Test patches: "
    f"{len(test_dataset)}"
)

print(
    f"Classes: "
    f"{test_dataset.classes}"
)

print(
    f"Class mapping: "
    f"{test_dataset.class_to_idx}"
)


if test_dataset.classes != CLASS_NAMES:
    raise ValueError(
        "Unexpected test class ordering."
    )


# ============================================================
# LOAD FROZEN RESNET18 CHECKPOINT
# ============================================================

print(
    "\nLoading frozen ResNet18 checkpoint..."
)

checkpoint = torch.load(
    RESNET_PATH,
    map_location=device,
    weights_only=False
)

print(
    f"ResNet18 checkpoint selected "
    f"at epoch: {checkpoint['best_epoch']}"
)

print(
    f"Development validation accuracy: "
    f"{checkpoint['best_val_accuracy']:.2f}%"
)


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

# Remove final classifier so output = 512-D features
model.fc = nn.Identity()

model = model.to(device)
model.eval()


# ============================================================
# LOAD FROZEN SVM
# ============================================================

print(
    "\nLoading frozen SVM..."
)

svm_model = joblib.load(
    SVM_PATH
)

print(
    "SVM loaded successfully."
)


# ============================================================
# EXTRACT TEST FEATURES
# ============================================================

print("\n" + "=" * 72)
print("EXTRACTING TEST FEATURES")
print("=" * 72)

all_features = []
all_labels = []


with torch.no_grad():

    for batch_number, (
        images,
        labels
    ) in enumerate(
        test_loader,
        start=1
    ):

        images = images.to(device)

        features = model(
            images
        )

        all_features.append(
            features.cpu().numpy()
        )

        all_labels.append(
            labels.numpy()
        )

        if (
            batch_number % 5 == 0
            or batch_number
            == len(test_loader)
        ):

            processed = min(
                batch_number
                * BATCH_SIZE,
                len(test_dataset)
            )

            print(
                f"Processed "
                f"{processed}/"
                f"{len(test_dataset)}"
            )


X_test = np.concatenate(
    all_features,
    axis=0
)

y_test = np.concatenate(
    all_labels,
    axis=0
)


print(
    f"\nTest feature matrix: "
    f"{X_test.shape}"
)


if X_test.shape != (
    len(test_dataset),
    512
):
    raise ValueError(
        "Unexpected test feature shape."
    )


# ============================================================
# SVM PREDICTION
# ============================================================

print(
    "\nRunning frozen SVM..."
)

y_pred = svm_model.predict(
    X_test
)


# ============================================================
# PATCH-LEVEL RESULTS
# ============================================================

patch_accuracy = accuracy_score(
    y_test,
    y_pred
)

patch_report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

patch_cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2]
)


print("\n" + "=" * 72)
print("PATCH-LEVEL TEST RESULTS")
print("=" * 72)

print(
    f"\nPatch accuracy: "
    f"{patch_accuracy * 100:.2f}%"
)

print("\nClassification report:\n")

print(
    patch_report
)

print(
    "Confusion matrix:"
)

print(
    patch_cm
)


# ============================================================
# EXTRACT ORIGINAL SAMPLE IDs
# ============================================================

def get_sample_id(
    patch_name
):

    match = re.match(
        r"(.+?)_y\d+_x\d+",
        patch_name
    )

    if match is None:
        raise ValueError(
            f"Could not extract sample ID "
            f"from {patch_name}"
        )

    return match.group(1)


# ============================================================
# SAVE PATCH-LEVEL PREDICTIONS
# ============================================================

patch_rows = []

sample_groups = defaultdict(
    lambda: {
        "true_labels": [],
        "predictions": []
    }
)


for index, (
    sample_path,
    class_index
) in enumerate(
    test_dataset.samples
):

    patch_name = Path(
        sample_path
    ).name

    sample_id = get_sample_id(
        patch_name
    )

    true_label = int(
        y_test[index]
    )

    predicted_label = int(
        y_pred[index]
    )

    patch_rows.append({
        "patch_name":
            patch_name,

        "sample_id":
            sample_id,

        "true_class":
            CLASS_NAMES[
                true_label
            ],

        "predicted_class":
            CLASS_NAMES[
                predicted_label
            ],

        "correct":
            int(
                true_label
                == predicted_label
            )
    })

    sample_groups[
        sample_id
    ]["true_labels"].append(
        true_label
    )

    sample_groups[
        sample_id
    ]["predictions"].append(
        predicted_label
    )


with open(
    PATCH_RESULTS_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "patch_name",
        "sample_id",
        "true_class",
        "predicted_class",
        "correct"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(
        patch_rows
    )


# ============================================================
# ORIGINAL-SAMPLE MAJORITY VOTE
# ============================================================

sample_rows = []

sample_true = []
sample_pred = []


for sample_id in sorted(
    sample_groups
):

    information = (
        sample_groups[
            sample_id
        ]
    )

    true_labels = np.array(
        information[
            "true_labels"
        ]
    )

    predictions = np.array(
        information[
            "predictions"
        ]
    )

    unique_true = np.unique(
        true_labels
    )

    if len(unique_true) != 1:
        raise ValueError(
            f"Inconsistent true labels "
            f"for {sample_id}"
        )

    true_label = int(
        unique_true[0]
    )

    counts = np.bincount(
        predictions,
        minlength=3
    )

    predicted_label = int(
        np.argmax(counts)
    )

    number_patches = len(
        predictions
    )

    winning_votes = int(
        counts[
            predicted_label
        ]
    )

    vote_percentage = (
        100.0
        * winning_votes
        / number_patches
    )

    sample_true.append(
        true_label
    )

    sample_pred.append(
        predicted_label
    )

    sample_rows.append({
        "sample_id":
            sample_id,

        "true_class":
            CLASS_NAMES[
                true_label
            ],

        "predicted_class":
            CLASS_NAMES[
                predicted_label
            ],

        "number_of_patches":
            number_patches,

        "votes_CR_0":
            int(counts[0]),

        "votes_CR_30":
            int(counts[1]),

        "votes_CR_41":
            int(counts[2]),

        "winning_vote_percent":
            vote_percentage,

        "correct":
            int(
                true_label
                == predicted_label
            )
    })


sample_true = np.array(
    sample_true
)

sample_pred = np.array(
    sample_pred
)


# ============================================================
# SAMPLE-LEVEL RESULTS
# ============================================================

sample_accuracy = accuracy_score(
    sample_true,
    sample_pred
)

sample_report = classification_report(
    sample_true,
    sample_pred,
    labels=[0, 1, 2],
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

sample_cm = confusion_matrix(
    sample_true,
    sample_pred,
    labels=[0, 1, 2]
)


print("\n" + "=" * 72)
print("ORIGINAL-SAMPLE TEST RESULTS")
print("=" * 72)

print(
    f"\nOriginal samples: "
    f"{len(sample_true)}"
)

print(
    f"Original-sample accuracy: "
    f"{sample_accuracy * 100:.2f}%"
)

print("\nClassification report:\n")

print(
    sample_report
)

print(
    "Confusion matrix:"
)

print(
    sample_cm
)


# ============================================================
# DISPLAY EACH ORIGINAL BUNDLE
# ============================================================

print("\n" + "=" * 72)
print("INDIVIDUAL ORIGINAL-SAMPLE PREDICTIONS")
print("=" * 72)


for row in sample_rows:

    status = (
        "CORRECT"
        if row["correct"] == 1
        else "WRONG"
    )

    print(
        f"\n{row['sample_id']}"
    )

    print(
        f"  True:      "
        f"{row['true_class']}"
    )

    print(
        f"  Predicted: "
        f"{row['predicted_class']}"
    )

    print(
        f"  Votes: "
        f"CR0={row['votes_CR_0']}, "
        f"CR30={row['votes_CR_30']}, "
        f"CR41={row['votes_CR_41']}"
    )

    print(
        f"  Winning vote: "
        f"{row['winning_vote_percent']:.2f}%"
    )

    print(
        f"  Result: {status}"
    )


# ============================================================
# SAVE SAMPLE RESULTS
# ============================================================

with open(
    SAMPLE_RESULTS_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "sample_id",
        "true_class",
        "predicted_class",
        "number_of_patches",
        "votes_CR_0",
        "votes_CR_30",
        "votes_CR_41",
        "winning_vote_percent",
        "correct"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(
        sample_rows
    )


# ============================================================
# SAVE TEXT SUMMARY
# ============================================================

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "FINAL TEST EVALUATION\n"
    )

    file.write(
        "=" * 72 + "\n\n"
    )

    file.write(
        "Pipeline:\n"
        "SEM patch -> ResNet18 512-D features "
        "-> StandardScaler -> RBF SVM\n\n"
    )

    file.write(
        "SVM configuration:\n"
        "kernel = RBF\n"
        "C = 1\n"
        "gamma = scale\n\n"
    )

    file.write(
        f"Test patches: "
        f"{len(y_test)}\n"
    )

    file.write(
        f"Patch accuracy: "
        f"{patch_accuracy * 100:.2f}%\n\n"
    )

    file.write(
        patch_report
    )

    file.write(
        "\nPatch confusion matrix:\n"
    )

    file.write(
        np.array2string(
            patch_cm
        )
    )

    file.write(
        "\n\n"
    )

    file.write(
        f"Independent original test samples: "
        f"{len(sample_true)}\n"
    )

    file.write(
        f"Original-sample accuracy: "
        f"{sample_accuracy * 100:.2f}%\n\n"
    )

    file.write(
        sample_report
    )

    file.write(
        "\nOriginal-sample confusion matrix:\n"
    )

    file.write(
        np.array2string(
            sample_cm
        )
    )

    file.write(
        "\n"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 72)
print("FINAL TEST EVALUATION COMPLETE")
print("=" * 72)

print(
    f"\nPatch predictions:\n"
    f"{PATCH_RESULTS_PATH}"
)

print(
    f"\nOriginal-sample predictions:\n"
    f"{SAMPLE_RESULTS_PATH}"
)

print(
    f"\nSummary:\n"
    f"{SUMMARY_PATH}"
)

print("""
IMPORTANT:
This was the final untouched test evaluation.

Do NOT tune the ResNet18 checkpoint, SVM kernel, C, gamma,
patch threshold, or other model settings based on these
test results.

The test set should now be treated as consumed.
""")