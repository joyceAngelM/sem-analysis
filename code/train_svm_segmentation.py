from pathlib import Path
import time
import pickle

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "final_70_15_15"
    / "svm_segmentation"
    / "svm_training_features.npz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "final_70_15_15"
    / "svm_segmentation"
)

MODEL_PATH = (
    OUTPUT_DIR
    / "svm_gf_segmentation_model.pkl"
)

SCALER_PATH = (
    OUTPUT_DIR
    / "svm_feature_scaler.pkl"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "svm_training_summary.txt"
)

RANDOM_SEED = 42


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SVM GLASS-FIBRE SEGMENTATION - MODEL TRAINING")
    print("=" * 80)

    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if not INPUT_PATH.exists():

        raise FileNotFoundError(
            f"Training features not found:\n"
            f"{INPUT_PATH}"
        )

    # --------------------------------------------------------
    # LOAD FEATURES
    # --------------------------------------------------------

    print(
        f"\nLoading training features from:\n"
        f"{INPUT_PATH}"
    )

    data = np.load(
        INPUT_PATH,
        allow_pickle=False
    )

    X = data["X"].astype(
        np.float32
    )

    y = data["y"].astype(
        np.uint8
    )

    feature_names = data[
        "feature_names"
    ].tolist()

    print("\n" + "=" * 80)
    print("TRAINING DATA CHECK")
    print("=" * 80)

    print(
        f"\nFeature matrix shape: "
        f"{X.shape}"
    )

    print(
        f"Label vector shape  : "
        f"{y.shape}"
    )

    print(
        f"Number of features  : "
        f"{X.shape[1]}"
    )

    print("\nFeatures:")

    for name in feature_names:
        print(f"  - {name}")

    class_0 = int(
        np.sum(y == 0)
    )

    class_1 = int(
        np.sum(y == 1)
    )

    print(
        f"\nClass 0 (non-GF): "
        f"{class_0:,}"
    )

    print(
        f"Class 1 (GF)    : "
        f"{class_1:,}"
    )

    if X.shape != (357600, 6):

        raise ValueError(
            f"Unexpected feature matrix shape: "
            f"{X.shape}"
        )

    if len(y) != 357600:

        raise ValueError(
            f"Unexpected number of labels: "
            f"{len(y)}"
        )

    if not np.isfinite(X).all():

        raise RuntimeError(
            "NaN or infinite feature values found."
        )

    if set(
        np.unique(y).tolist()
    ) != {0, 1}:

        raise RuntimeError(
            "Expected binary labels 0 and 1."
        )

    # --------------------------------------------------------
    # STANDARDIZE FEATURES
    #
    # Fit ONLY on training pixels.
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FEATURE STANDARDIZATION")
    print("=" * 80)

    print(
        "\nFitting StandardScaler "
        "using TRAINING pixels only..."
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    ).astype(
        np.float32
    )

    print(
        "Feature standardization complete."
    )

    # --------------------------------------------------------
    # TRAIN LINEAR SVM
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("TRAINING LINEAR SVM")
    print("=" * 80)

    print(
        "\nThis is a pixel-level binary SVM:"
        "\n  0 = non-glass-fibre"
        "\n  1 = glass-fibre"
    )

    print(
        "\nValidation and test pixels are NOT "
        "used during this training."
    )

    svm = LinearSVC(
        C=1.0,
        class_weight=None,
        max_iter=10000,
        random_state=RANDOM_SEED
    )

    start_time = time.time()

    svm.fit(
        X_scaled,
        y
    )

    training_seconds = (
        time.time()
        - start_time
    )

    print(
        f"\nTraining completed in "
        f"{training_seconds:.2f} seconds."
    )

    print(
        f"Iterations used: "
        f"{svm.n_iter_}"
    )

    # --------------------------------------------------------
    # TRAINING-SET SANITY CHECK
    #
    # This is NOT final model performance.
    # It only confirms that training worked.
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("TRAINING-SET SANITY CHECK")
    print("=" * 80)

    train_predictions = svm.predict(
        X_scaled
    )

    train_accuracy = accuracy_score(
        y,
        train_predictions
    )

    train_precision = precision_score(
        y,
        train_predictions,
        zero_division=0
    )

    train_recall = recall_score(
        y,
        train_predictions,
        zero_division=0
    )

    train_f1 = f1_score(
        y,
        train_predictions,
        zero_division=0
    )

    train_cm = confusion_matrix(
        y,
        train_predictions,
        labels=[0, 1]
    )

    print(
        f"\nTraining Accuracy : "
        f"{train_accuracy:.6f} "
        f"({train_accuracy * 100:.2f}%)"
    )

    print(
        f"Training Precision: "
        f"{train_precision:.6f} "
        f"({train_precision * 100:.2f}%)"
    )

    print(
        f"Training Recall   : "
        f"{train_recall:.6f} "
        f"({train_recall * 100:.2f}%)"
    )

    print(
        f"Training F1       : "
        f"{train_f1:.6f} "
        f"({train_f1 * 100:.2f}%)"
    )

    print(
        "\nTraining confusion matrix "
        "[[TN, FP], [FN, TP]]:"
    )

    print(
        train_cm
    )

    print(
        "\nIMPORTANT:"
        "\nThese training metrics are NOT "
        "the final SVM segmentation results."
        "\nWe will evaluate the model separately "
        "on the untouched test set."
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    with open(
        MODEL_PATH,
        "wb"
    ) as file:

        pickle.dump(
            svm,
            file
        )

    with open(
        SCALER_PATH,
        "wb"
    ) as file:

        pickle.dump(
            scaler,
            file
        )

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "SVM GLASS-FIBRE SEGMENTATION "
            "TRAINING SUMMARY\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            "Task: Pixel-level glass-fibre "
            "segmentation\n"
        )

        file.write(
            "Model: LinearSVC\n"
        )

        file.write(
            "C: 1.0\n"
        )

        file.write(
            "Feature scaling: StandardScaler\n"
        )

        file.write(
            "Training specimens: 34\n"
        )

        file.write(
            "Original training patches: 894\n"
        )

        file.write(
            f"Training pixels: {len(y)}\n"
        )

        file.write(
            f"Non-GF pixels: {class_0}\n"
        )

        file.write(
            f"GF pixels: {class_1}\n"
        )

        file.write(
            f"Features per pixel: "
            f"{X.shape[1]}\n\n"
        )

        file.write(
            "Features:\n"
        )

        for name in feature_names:

            file.write(
                f"- {name}\n"
            )

        file.write(
            f"\nTraining time (seconds): "
            f"{training_seconds:.4f}\n"
        )

        file.write(
            f"Training accuracy: "
            f"{train_accuracy:.6f}\n"
        )

        file.write(
            f"Training precision: "
            f"{train_precision:.6f}\n"
        )

        file.write(
            f"Training recall: "
            f"{train_recall:.6f}\n"
        )

        file.write(
            f"Training F1: "
            f"{train_f1:.6f}\n"
        )

        file.write(
            "\nTraining metrics are only a "
            "sanity check and are NOT final "
            "test performance.\n"
        )

    # --------------------------------------------------------
    # FINAL MESSAGE
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("SVM TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"\nSVM model saved to:\n"
        f"{MODEL_PATH}"
    )

    print(
        f"\nScaler saved to:\n"
        f"{SCALER_PATH}"
    )

    print(
        f"\nTraining summary saved to:\n"
        f"{SUMMARY_PATH}"
    )

    print(
        "\nNo validation or test data "
        "was used to fit the scaler or SVM."
    )


if __name__ == "__main__":
    main()