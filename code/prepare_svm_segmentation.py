from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

from scipy.ndimage import (
    uniform_filter,
    sobel
)


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "final_70_15_15"
    / "unet_dataset_256"
    / "unet_patch_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "final_70_15_15"
    / "svm_segmentation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FEATURES_PATH = (
    OUTPUT_DIR
    / "svm_training_features.npz"
)

SAMPLE_SUMMARY_PATH = (
    OUTPUT_DIR
    / "svm_training_sampling_summary.csv"
)


# Up to this many pixels PER CLASS
# will be sampled from each training patch.
PIXELS_PER_CLASS_PER_PATCH = 200

RANDOM_SEED = 42

rng = np.random.default_rng(
    RANDOM_SEED
)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_sem(path):

    with Image.open(path) as image:

        gray = np.array(
            image.convert("L"),
            dtype=np.float32
        )

    # Scale intensity to 0-1
    gray = gray / 255.0

    return gray


def load_binary_mask(path):

    with Image.open(path) as image:

        mask = np.array(
            image.convert("L")
        )

    return mask > 0


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(gray):

    # --------------------------------------------------------
    # Original grayscale intensity
    # --------------------------------------------------------

    intensity = gray

    # --------------------------------------------------------
    # Local mean
    # --------------------------------------------------------

    mean_3 = uniform_filter(
        gray,
        size=3,
        mode="reflect"
    )

    mean_7 = uniform_filter(
        gray,
        size=7,
        mode="reflect"
    )

    # --------------------------------------------------------
    # Local standard deviation
    #
    # std = sqrt(E[x^2] - E[x]^2)
    # --------------------------------------------------------

    mean_sq_3 = uniform_filter(
        gray ** 2,
        size=3,
        mode="reflect"
    )

    variance_3 = (
        mean_sq_3
        - mean_3 ** 2
    )

    variance_3 = np.maximum(
        variance_3,
        0
    )

    std_3 = np.sqrt(
        variance_3
    )

    mean_sq_7 = uniform_filter(
        gray ** 2,
        size=7,
        mode="reflect"
    )

    variance_7 = (
        mean_sq_7
        - mean_7 ** 2
    )

    variance_7 = np.maximum(
        variance_7,
        0
    )

    std_7 = np.sqrt(
        variance_7
    )

    # --------------------------------------------------------
    # Sobel gradient magnitude
    # --------------------------------------------------------

    grad_x = sobel(
        gray,
        axis=1,
        mode="reflect"
    )

    grad_y = sobel(
        gray,
        axis=0,
        mode="reflect"
    )

    gradient = np.sqrt(
        grad_x ** 2
        + grad_y ** 2
    )

    # --------------------------------------------------------
    # Stack features:
    #
    # H × W × 6
    # --------------------------------------------------------

    features = np.stack(
        [
            intensity,
            mean_3,
            mean_7,
            std_3,
            std_7,
            gradient
        ],
        axis=-1
    )

    return features.astype(
        np.float32
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SVM GLASS-FIBRE SEGMENTATION - TRAINING DATA PREPARATION")
    print("=" * 80)

    print(
        f"\nManifest:\n{MANIFEST_PATH}"
    )

    print(
        f"\nPixels per class per patch: "
        f"{PIXELS_PER_CLASS_PER_PATCH}"
    )

    print(
        "\nFeatures:"
        "\n  1. Grayscale intensity"
        "\n  2. 3x3 local mean"
        "\n  3. 7x7 local mean"
        "\n  4. 3x3 local standard deviation"
        "\n  5. 7x7 local standard deviation"
        "\n  6. Sobel gradient magnitude"
    )

    # ========================================================
    # CHECK MANIFEST
    # ========================================================

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n"
            f"{MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    # ========================================================
    # USE ORIGINAL TRAINING PATCHES ONLY
    # ========================================================

    train_df = manifest[
        manifest["split"] == "train"
    ].copy()

    val_df = manifest[
        manifest["split"] == "validation"
    ].copy()

    test_df = manifest[
        manifest["split"] == "test"
    ].copy()

    print("\n" + "=" * 80)
    print("DATASET CHECK")
    print("=" * 80)

    print(
        f"\nOriginal training patches : "
        f"{len(train_df)}"
    )

    print(
        f"Validation patches        : "
        f"{len(val_df)}"
    )

    print(
        f"Test patches              : "
        f"{len(test_df)}"
    )

    if len(train_df) != 894:

        raise ValueError(
            f"Expected 894 training patches, "
            f"found {len(train_df)}."
        )

    if len(val_df) != 173:

        raise ValueError(
            f"Expected 173 validation patches, "
            f"found {len(val_df)}."
        )

    if len(test_df) != 179:

        raise ValueError(
            f"Expected 179 test patches, "
            f"found {len(test_df)}."
        )

    # ========================================================
    # SPECIMEN-LEVEL LEAKAGE CHECK
    # ========================================================

    train_ids = set(
        train_df["sample_id"]
        .astype(str)
    )

    val_ids = set(
        val_df["sample_id"]
        .astype(str)
    )

    test_ids = set(
        test_df["sample_id"]
        .astype(str)
    )

    print("\n" + "=" * 80)
    print("DATA LEAKAGE CHECK")
    print("=" * 80)

    print(
        "\nTRAIN IDs ∩ VALIDATION IDs =",
        sorted(train_ids & val_ids)
    )

    print(
        "TRAIN IDs ∩ TEST IDs       =",
        sorted(train_ids & test_ids)
    )

    print(
        "VALIDATION IDs ∩ TEST IDs  =",
        sorted(val_ids & test_ids)
    )

    if (
        train_ids & val_ids
        or train_ids & test_ids
        or val_ids & test_ids
    ):

        raise RuntimeError(
            "Specimen-level data leakage detected."
        )

    print(
        "\nPASS: No specimen-level leakage."
    )

    print(
        f"\nTraining specimens   : "
        f"{len(train_ids)}"
    )

    print(
        f"Validation specimens : "
        f"{len(val_ids)}"
    )

    print(
        f"Test specimens       : "
        f"{len(test_ids)}"
    )

    # ========================================================
    # PREPARE TRAINING PIXELS
    # ========================================================

    all_features = []
    all_labels = []

    summary_rows = []

    total_gf_available = 0
    total_non_gf_available = 0

    total_gf_sampled = 0
    total_non_gf_sampled = 0

    print("\n" + "=" * 80)
    print("EXTRACTING SVM TRAINING FEATURES")
    print("=" * 80)

    for position, (_, row) in enumerate(
        train_df.iterrows(),
        start=1
    ):

        # ----------------------------------------------------
        # LOAD SEM + GF + ROI
        # ----------------------------------------------------

        gray = load_sem(
            row["image_patch"]
        )

        gf_mask = load_binary_mask(
            row["gf_mask_patch"]
        )

        roi = load_binary_mask(
            row["bundle_mask_patch"]
        )

        # ----------------------------------------------------
        # DIMENSION CHECK
        # ----------------------------------------------------

        if gray.shape != (256, 256):

            raise ValueError(
                f"SEM dimension error: "
                f"{row['patch_id']}"
            )

        if gf_mask.shape != (256, 256):

            raise ValueError(
                f"GF mask dimension error: "
                f"{row['patch_id']}"
            )

        if roi.shape != (256, 256):

            raise ValueError(
                f"Bundle mask dimension error: "
                f"{row['patch_id']}"
            )

        # ----------------------------------------------------
        # EXTRACT FEATURES
        # ----------------------------------------------------

        features = extract_features(
            gray
        )

        # ----------------------------------------------------
        # TARGET DEFINITION
        #
        # Positive:
        # GF pixels inside valid bundle ROI.
        #
        # Negative:
        # non-GF pixels inside valid bundle ROI.
        # ----------------------------------------------------

        gf_valid = (
            gf_mask
            & roi
        )

        non_gf_valid = (
            (~gf_mask)
            & roi
        )

        gf_indices = np.flatnonzero(
            gf_valid.ravel()
        )

        non_gf_indices = np.flatnonzero(
            non_gf_valid.ravel()
        )

        gf_available = len(
            gf_indices
        )

        non_gf_available = len(
            non_gf_indices
        )

        total_gf_available += (
            gf_available
        )

        total_non_gf_available += (
            non_gf_available
        )

        # ----------------------------------------------------
        # SAMPLE EACH CLASS
        # ----------------------------------------------------

        gf_sample_count = min(
            PIXELS_PER_CLASS_PER_PATCH,
            gf_available
        )

        non_gf_sample_count = min(
            PIXELS_PER_CLASS_PER_PATCH,
            non_gf_available
        )

        if gf_sample_count > 0:

            selected_gf = rng.choice(
                gf_indices,
                size=gf_sample_count,
                replace=False
            )

        else:

            selected_gf = np.array(
                [],
                dtype=np.int64
            )

        if non_gf_sample_count > 0:

            selected_non_gf = rng.choice(
                non_gf_indices,
                size=non_gf_sample_count,
                replace=False
            )

        else:

            selected_non_gf = np.array(
                [],
                dtype=np.int64
            )

        flat_features = features.reshape(
            -1,
            features.shape[-1]
        )

        # ----------------------------------------------------
        # GF FEATURES
        # ----------------------------------------------------

        if len(selected_gf) > 0:

            gf_features = flat_features[
                selected_gf
            ]

            gf_labels = np.ones(
                len(selected_gf),
                dtype=np.uint8
            )

            all_features.append(
                gf_features
            )

            all_labels.append(
                gf_labels
            )

        # ----------------------------------------------------
        # NON-GF FEATURES
        # ----------------------------------------------------

        if len(selected_non_gf) > 0:

            non_gf_features = flat_features[
                selected_non_gf
            ]

            non_gf_labels = np.zeros(
                len(selected_non_gf),
                dtype=np.uint8
            )

            all_features.append(
                non_gf_features
            )

            all_labels.append(
                non_gf_labels
            )

        total_gf_sampled += (
            gf_sample_count
        )

        total_non_gf_sampled += (
            non_gf_sample_count
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        summary_rows.append({
            "sample_id":
                row["sample_id"],

            "compaction_ratio":
                row["compaction_ratio"],

            "layer":
                row["layer"],

            "patch_id":
                row["patch_id"],

            "gf_pixels_available":
                gf_available,

            "non_gf_pixels_available":
                non_gf_available,

            "gf_pixels_sampled":
                gf_sample_count,

            "non_gf_pixels_sampled":
                non_gf_sample_count
        })

        if (
            position % 100 == 0
            or position == len(train_df)
        ):

            print(
                f"Processed "
                f"{position}/{len(train_df)} "
                f"training patches"
            )

    # ========================================================
    # COMBINE
    # ========================================================

    if len(all_features) == 0:

        raise RuntimeError(
            "No SVM training features were generated."
        )

    X = np.concatenate(
        all_features,
        axis=0
    ).astype(np.float32)

    y = np.concatenate(
        all_labels,
        axis=0
    ).astype(np.uint8)

    # ========================================================
    # FINAL CHECKS
    # ========================================================

    if len(X) != len(y):

        raise RuntimeError(
            "Feature/label count mismatch."
        )

    if X.shape[1] != 6:

        raise RuntimeError(
            f"Expected 6 features, "
            f"found {X.shape[1]}."
        )

    if not np.isfinite(X).all():

        raise RuntimeError(
            "NaN or infinite values found "
            "in SVM features."
        )

    unique_labels = set(
        np.unique(y).tolist()
    )

    if not unique_labels.issubset(
        {0, 1}
    ):

        raise RuntimeError(
            f"Unexpected labels: "
            f"{unique_labels}"
        )

    # ========================================================
    # SHUFFLE ONCE
    # ========================================================

    permutation = rng.permutation(
        len(y)
    )

    X = X[
        permutation
    ]

    y = y[
        permutation
    ]

    # ========================================================
    # SAVE NPZ
    # ========================================================

    np.savez_compressed(
        FEATURES_PATH,
        X=X,
        y=y,
        feature_names=np.array(
            [
                "intensity",
                "mean_3x3",
                "mean_7x7",
                "std_3x3",
                "std_7x7",
                "sobel_gradient"
            ]
        )
    )

    # ========================================================
    # SAVE SAMPLING SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SAMPLE_SUMMARY_PATH,
        index=False
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    label_counts = Counter(
        y.tolist()
    )

    print("\n" + "=" * 80)
    print("SVM TRAINING DATA PREPARATION COMPLETE")
    print("=" * 80)

    print(
        f"\nOriginal training patches processed: "
        f"{len(train_df)}"
    )

    print(
        f"Independent training specimens: "
        f"{len(train_ids)}"
    )

    print(
        f"\nGF pixels available inside ROI: "
        f"{total_gf_available:,}"
    )

    print(
        f"Non-GF pixels available inside ROI: "
        f"{total_non_gf_available:,}"
    )

    print(
        f"\nGF pixels sampled: "
        f"{total_gf_sampled:,}"
    )

    print(
        f"Non-GF pixels sampled: "
        f"{total_non_gf_sampled:,}"
    )

    print(
        f"\nFinal feature matrix shape: "
        f"{X.shape}"
    )

    print(
        f"Final label vector shape: "
        f"{y.shape}"
    )

    print(
        f"\nClass 0 (non-GF): "
        f"{label_counts.get(0, 0):,}"
    )

    print(
        f"Class 1 (GF)    : "
        f"{label_counts.get(1, 0):,}"
    )

    print(
        f"\nFeatures saved to:\n"
        f"{FEATURES_PATH}"
    )

    print(
        f"\nSampling summary saved to:\n"
        f"{SAMPLE_SUMMARY_PATH}"
    )

    print(
        "\nIMPORTANT:"
        "\n- Only ORIGINAL TRAINING patches were used."
        "\n- Validation patches were NOT sampled."
        "\n- Test patches were NOT sampled."
        "\n- Only pixels inside the bundle ROI were eligible."
        "\n- GF mask was used only as the target label."
        "\n- No SVM model has been trained yet."
    )


if __name__ == "__main__":
    main()