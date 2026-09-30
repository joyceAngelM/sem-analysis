from pathlib import Path
import csv
import re
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "resnet_features"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "svm"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_FEATURES = FEATURE_DIR / "train_features.npy"
TRAIN_LABELS = FEATURE_DIR / "train_labels.npy"

VAL_FEATURES = FEATURE_DIR / "validation_features.npy"
VAL_LABELS = FEATURE_DIR / "validation_labels.npy"

VAL_INFO = FEATURE_DIR / "validation_patch_info.csv"

BEST_MODEL_PATH = OUTPUT_DIR / "best_svm.joblib"
RESULTS_PATH = OUTPUT_DIR / "svm_validation_results.csv"


# ============================================================
# LOAD FEATURES
# ============================================================

X_train = np.load(TRAIN_FEATURES)
y_train = np.load(TRAIN_LABELS)

X_val = np.load(VAL_FEATURES)
y_val = np.load(VAL_LABELS)


print("=" * 72)
print("SVM TRAINING ON RESNET18 DEEP FEATURES")
print("=" * 72)

print(f"\nTraining features: {X_train.shape}")
print(f"Training labels:   {y_train.shape}")

print(f"Validation features: {X_val.shape}")
print(f"Validation labels:   {y_val.shape}")


# ============================================================
# LOAD VALIDATION PATCH INFORMATION
# ============================================================

val_patch_names = []

with open(
    VAL_INFO,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:
        val_patch_names.append(
            row["patch_name"]
        )


if len(val_patch_names) != len(y_val):
    raise ValueError(
        "Validation patch information does not "
        "match validation feature count."
    )


# ============================================================
# GET ORIGINAL SAMPLE ID FROM PATCH NAME
#
# Example:
#
# CR30_L1M2_y224_x448.png
#
# becomes:
#
# CR30_L1M2
# ============================================================

def get_sample_id(patch_name):

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


val_sample_ids = [
    get_sample_id(name)
    for name in val_patch_names
]


# ============================================================
# SAMPLE-LEVEL MAJORITY VOTING
# ============================================================

def calculate_sample_accuracy(
    true_labels,
    predicted_labels,
    sample_ids
):

    grouped = {}

    for true_label, prediction, sample_id in zip(
        true_labels,
        predicted_labels,
        sample_ids
    ):

        if sample_id not in grouped:
            grouped[sample_id] = {
                "true": int(true_label),
                "predictions": []
            }

        grouped[sample_id][
            "predictions"
        ].append(
            int(prediction)
        )

    sample_true = []
    sample_pred = []

    for sample_id in sorted(grouped):

        information = grouped[sample_id]

        predictions = np.array(
            information["predictions"]
        )

        counts = np.bincount(
            predictions,
            minlength=3
        )

        majority_prediction = int(
            np.argmax(counts)
        )

        sample_true.append(
            information["true"]
        )

        sample_pred.append(
            majority_prediction
        )

    sample_accuracy = accuracy_score(
        sample_true,
        sample_pred
    )

    return (
        sample_accuracy,
        np.array(sample_true),
        np.array(sample_pred)
    )


# ============================================================
# SVM CONFIGURATIONS
# ============================================================

configurations = [
    {
        "name": "linear_C0.1",
        "kernel": "linear",
        "C": 0.1,
        "gamma": "scale"
    },
    {
        "name": "linear_C1",
        "kernel": "linear",
        "C": 1.0,
        "gamma": "scale"
    },
    {
        "name": "linear_C10",
        "kernel": "linear",
        "C": 10.0,
        "gamma": "scale"
    },
    {
        "name": "rbf_C0.1",
        "kernel": "rbf",
        "C": 0.1,
        "gamma": "scale"
    },
    {
        "name": "rbf_C1",
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale"
    },
    {
        "name": "rbf_C10",
        "kernel": "rbf",
        "C": 10.0,
        "gamma": "scale"
    }
]


# ============================================================
# TRAIN AND VALIDATE
# ============================================================

results = []

best_model = None
best_name = None

best_sample_accuracy = -1.0
best_patch_accuracy = -1.0


print("\n" + "=" * 72)
print("MODEL COMPARISON")
print("=" * 72)


for config in configurations:

    print(
        f"\nTraining: {config['name']}"
    )

    # StandardScaler is fitted ONLY using training data.
    # This avoids validation leakage.

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "svm",
            SVC(
                kernel=config["kernel"],
                C=config["C"],
                gamma=config["gamma"]
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_val
    )

    patch_accuracy = accuracy_score(
        y_val,
        predictions
    )

    (
        sample_accuracy,
        sample_true,
        sample_pred
    ) = calculate_sample_accuracy(
        y_val,
        predictions,
        val_sample_ids
    )

    print(
        f"  Patch accuracy:  "
        f"{patch_accuracy * 100:.2f}%"
    )

    print(
        f"  Sample accuracy: "
        f"{sample_accuracy * 100:.2f}%"
    )

    results.append({
        "model": config["name"],
        "kernel": config["kernel"],
        "C": config["C"],
        "gamma": config["gamma"],
        "patch_accuracy":
            patch_accuracy,
        "sample_accuracy":
            sample_accuracy
    })

    # --------------------------------------------------------
    # MODEL SELECTION
    #
    # Priority:
    # 1. Original-sample accuracy
    # 2. Patch accuracy as tie-breaker
    # --------------------------------------------------------

    is_better = False

    if (
        sample_accuracy
        > best_sample_accuracy
    ):
        is_better = True

    elif (
        sample_accuracy
        == best_sample_accuracy
        and patch_accuracy
        > best_patch_accuracy
    ):
        is_better = True

    if is_better:

        best_model = model
        best_name = config["name"]

        best_sample_accuracy = (
            sample_accuracy
        )

        best_patch_accuracy = (
            patch_accuracy
        )


# ============================================================
# SAVE COMPARISON RESULTS
# ============================================================

with open(
    RESULTS_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "model",
        "kernel",
        "C",
        "gamma",
        "patch_accuracy",
        "sample_accuracy"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


# ============================================================
# FINAL VALIDATION RESULTS FOR SELECTED MODEL
# ============================================================

best_predictions = best_model.predict(
    X_val
)

(
    final_sample_accuracy,
    final_sample_true,
    final_sample_pred
) = calculate_sample_accuracy(
    y_val,
    best_predictions,
    val_sample_ids
)


print("\n" + "=" * 72)
print("SELECTED SVM")
print("=" * 72)

print(f"\nModel: {best_name}")

print(
    f"Validation patch accuracy: "
    f"{best_patch_accuracy * 100:.2f}%"
)

print(
    f"Validation original-sample accuracy: "
    f"{best_sample_accuracy * 100:.2f}%"
)


# ============================================================
# PATCH-LEVEL REPORT
# ============================================================

print("\n" + "=" * 72)
print("PATCH-LEVEL CLASSIFICATION REPORT")
print("=" * 72)

print(
    classification_report(
        y_val,
        best_predictions,
        target_names=[
            "CR_0",
            "CR_30",
            "CR_41"
        ],
        digits=4,
        zero_division=0
    )
)


print(
    "Patch-level confusion matrix:"
)

print(
    confusion_matrix(
        y_val,
        best_predictions
    )
)


# ============================================================
# ORIGINAL-SAMPLE REPORT
# ============================================================

print("\n" + "=" * 72)
print("ORIGINAL-SAMPLE CLASSIFICATION REPORT")
print("=" * 72)

print(
    classification_report(
        final_sample_true,
        final_sample_pred,
        labels=[0, 1, 2],
        target_names=[
            "CR_0",
            "CR_30",
            "CR_41"
        ],
        digits=4,
        zero_division=0
    )
)

print(
    "Original-sample confusion matrix:"
)

print(
    confusion_matrix(
        final_sample_true,
        final_sample_pred,
        labels=[0, 1, 2]
    )
)


# ============================================================
# SAVE SELECTED MODEL
# ============================================================

joblib.dump(
    best_model,
    BEST_MODEL_PATH
)


print("\n" + "=" * 72)
print("SVM TRAINING COMPLETE")
print("=" * 72)

print(
    f"\nSelected model: "
    f"{best_name}"
)

print(
    f"\nBest SVM saved to:\n"
    f"{BEST_MODEL_PATH}"
)

print(
    f"\nValidation comparison saved to:\n"
    f"{RESULTS_PATH}"
)

print("""
IMPORTANT:
The SVM was selected using TRAINING + VALIDATION information only.

The TEST set has still NOT been loaded.

Do not evaluate the test set yet.
""")