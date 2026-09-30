from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_HISTORY = (
    PROJECT_ROOT
    / "outputs"
    / "resnet18"
    / "training_history_8epochs.csv"
)

SVM_RESULTS = (
    PROJECT_ROOT
    / "outputs"
    / "svm"
    / "svm_validation_results.csv"
)

PATCH_RESULTS = (
    PROJECT_ROOT
    / "outputs"
    / "final_test"
    / "test_patch_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "final_visualizations"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. RESNET18 TRAINING AND VALIDATION CURVES
# ============================================================

history = pd.read_csv(TRAIN_HISTORY)

epochs = history["epoch"]

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["train_accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    history["val_accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title(
    "ResNet18 Training and Validation Accuracy"
)

plt.xticks(epochs)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

accuracy_path = (
    OUTPUT_DIR
    / "01_resnet18_accuracy_curve.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# Loss curve
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    history["train_loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(
    "ResNet18 Training and Validation Loss"
)

plt.xticks(epochs)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

loss_path = (
    OUTPUT_DIR
    / "02_resnet18_loss_curve.png"
)

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 2. SVM MODEL COMPARISON
# ============================================================

svm = pd.read_csv(
    SVM_RESULTS
)

svm["patch_accuracy_percent"] = (
    svm["patch_accuracy"] * 100
)

svm["sample_accuracy_percent"] = (
    svm["sample_accuracy"] * 100
)


x = np.arange(
    len(svm)
)

width = 0.36


plt.figure(
    figsize=(10, 5.5)
)

plt.bar(
    x - width / 2,
    svm["patch_accuracy_percent"],
    width,
    label="Patch-level Accuracy"
)

plt.bar(
    x + width / 2,
    svm["sample_accuracy_percent"],
    width,
    label="Original-sample Accuracy"
)

plt.xlabel(
    "SVM Configuration"
)

plt.ylabel(
    "Validation Accuracy (%)"
)

plt.title(
    "SVM Validation Performance Comparison"
)

plt.xticks(
    x,
    svm["model"],
    rotation=25,
    ha="right"
)

plt.ylim(
    90,
    101
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

svm_path = (
    OUTPUT_DIR
    / "03_svm_comparison.png"
)

plt.savefig(
    svm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 3. FINAL TEST CONFUSION MATRIX
# ============================================================

patch_data = pd.read_csv(
    PATCH_RESULTS
)

class_names = [
    "CR_0",
    "CR_30",
    "CR_41"
]

confusion = pd.crosstab(
    patch_data["true_class"],
    patch_data["predicted_class"]
)

confusion = confusion.reindex(
    index=class_names,
    columns=class_names,
    fill_value=0
)


matrix = confusion.values


plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    matrix,
    interpolation="nearest"
)

plt.title(
    "Final Test Confusion Matrix"
)

plt.colorbar()

tick_marks = np.arange(
    len(class_names)
)

plt.xticks(
    tick_marks,
    class_names
)

plt.yticks(
    tick_marks,
    class_names
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)


threshold = (
    matrix.max() / 2.0
)


for i in range(
    matrix.shape[0]
):

    for j in range(
        matrix.shape[1]
    ):

        plt.text(
            j,
            i,
            str(matrix[i, j]),
            horizontalalignment="center",
            verticalalignment="center"
        )


plt.tight_layout()

confusion_path = (
    OUTPUT_DIR
    / "04_final_test_confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("=" * 72)
print("FINAL VISUALIZATIONS CREATED")
print("=" * 72)

print(
    f"\nSaved to:\n"
    f"{OUTPUT_DIR}"
)

print("\nCreated:")

print(
    "1. ResNet18 accuracy curve"
)

print(
    "2. ResNet18 loss curve"
)

print(
    "3. SVM comparison graph"
)

print(
    "4. Final test confusion matrix"
)

print(
    "\nExpected final test confusion matrix:"
)

print(matrix)

print(
    "\nVisualization generation complete."
)