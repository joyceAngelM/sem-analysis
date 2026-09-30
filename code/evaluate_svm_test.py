from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import uniform_filter, sobel


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

SVM_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "final_70_15_15"
    / "svm_segmentation"
)

MODEL_PATH = (
    SVM_DIR
    / "svm_gf_segmentation_model.pkl"
)

SCALER_PATH = (
    SVM_DIR
    / "svm_feature_scaler.pkl"
)

OUTPUT_DIR = (
    SVM_DIR
    / "test_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PATCH_RESULTS_PATH = (
    OUTPUT_DIR
    / "svm_test_patch_results.csv"
)

SPECIMEN_RESULTS_PATH = (
    OUTPUT_DIR
    / "svm_test_specimen_results.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "svm_test_summary.txt"
)


# ============================================================
# IMAGE LOADING
# ============================================================

def load_sem(path):

    with Image.open(path) as image:
        gray = np.array(
            image.convert("L"),
            dtype=np.float32
        )

    return gray / 255.0


def load_binary_mask(path):

    with Image.open(path) as image:
        mask = np.array(
            image.convert("L")
        )

    return mask > 0


# ============================================================
# SAME FEATURE EXTRACTION USED FOR TRAINING
# ============================================================

def extract_features(gray):

    intensity = gray

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

    mean_sq_3 = uniform_filter(
        gray ** 2,
        size=3,
        mode="reflect"
    )

    variance_3 = np.maximum(
        mean_sq_3 - mean_3 ** 2,
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

    variance_7 = np.maximum(
        mean_sq_7 - mean_7 ** 2,
        0
    )

    std_7 = np.sqrt(
        variance_7
    )

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
# METRICS
# ============================================================

def calculate_metrics(
    prediction,
    target,
    roi
):

    prediction = prediction.astype(bool)
    target = target.astype(bool)
    roi = roi.astype(bool)

    # Evaluate only inside valid bundle ROI
    prediction = prediction & roi
    target = target & roi

    tp = np.logical_and(
        prediction,
        target
    ).sum()

    fp = (
        np.logical_and(
            prediction,
            np.logical_not(target)
        )
        & roi
    ).sum()

    fn = (
        np.logical_and(
            np.logical_not(prediction),
            target
        )
        & roi
    ).sum()

    tn = (
        np.logical_and(
            np.logical_not(prediction),
            np.logical_not(target)
        )
        & roi
    ).sum()

    eps = 1e-8

    dice = (
        2.0 * tp
        / (
            2.0 * tp
            + fp
            + fn
            + eps
        )
    )

    iou = (
        tp
        / (
            tp
            + fp
            + fn
            + eps
        )
    )

    precision = (
        tp
        / (
            tp
            + fp
            + eps
        )
    )

    recall = (
        tp
        / (
            tp
            + fn
            + eps
        )
    )

    pixel_accuracy = (
        tp + tn
    ) / (
        tp
        + tn
        + fp
        + fn
        + eps
    )

    valid_roi_pixels = int(
        roi.sum()
    )

    gt_gf_pixels = int(
        target.sum()
    )

    pred_gf_pixels = int(
        prediction.sum()
    )

    gt_gf_percentage = (
        gt_gf_pixels
        / (
            valid_roi_pixels
            + eps
        )
        * 100.0
    )

    pred_gf_percentage = (
        pred_gf_pixels
        / (
            valid_roi_pixels
            + eps
        )
        * 100.0
    )

    gf_area_fraction_error = abs(
        pred_gf_percentage
        - gt_gf_percentage
    )

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),

        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "pixel_accuracy":
            float(pixel_accuracy),

        "valid_roi_pixels":
            valid_roi_pixels,

        "gt_gf_pixels":
            gt_gf_pixels,

        "pred_gf_pixels":
            pred_gf_pixels,

        "gt_gf_percentage":
            float(gt_gf_percentage),

        "pred_gf_percentage":
            float(pred_gf_percentage),

        "gf_area_fraction_error":
            float(
                gf_area_fraction_error
            )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("FINAL SVM GLASS-FIBRE SEGMENTATION TEST")
    print("=" * 80)

    # --------------------------------------------------------
    # FILE CHECK
    # --------------------------------------------------------

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n"
            f"{MANIFEST_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"SVM model not found:\n"
            f"{MODEL_PATH}"
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found:\n"
            f"{SCALER_PATH}"
        )

    # --------------------------------------------------------
    # LOAD MANIFEST
    # --------------------------------------------------------

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    test_df = manifest[
        manifest["split"] == "test"
    ].copy()

    test_ids = sorted(
        test_df[
            "sample_id"
        ].astype(str).unique()
    )

    print(
        f"\nUntouched test patches: "
        f"{len(test_df)}"
    )

    print(
        f"Independent test specimens: "
        f"{len(test_ids)}"
    )

    print("\nTest specimen IDs:")

    for sample_id in test_ids:
        print(
            f"  {sample_id}"
        )

    if len(test_df) != 179:
        raise ValueError(
            f"Expected 179 test patches, "
            f"found {len(test_df)}."
        )

    if len(test_ids) != 7:
        raise ValueError(
            f"Expected 7 test specimens, "
            f"found {len(test_ids)}."
        )

    # --------------------------------------------------------
    # LOAD TRAINED SCALER + SVM
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("LOADING TRAINED SVM")
    print("=" * 80)

    with open(
        SCALER_PATH,
        "rb"
    ) as file:
        scaler = pickle.load(file)

    with open(
        MODEL_PATH,
        "rb"
    ) as file:
        svm = pickle.load(file)

    print(
        "\nScaler loaded successfully."
    )

    print(
        "LinearSVC loaded successfully."
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("RUNNING SVM ON UNTOUCHED TEST SET")
    print("=" * 80)

    patch_results = []

    global_tp = 0
    global_fp = 0
    global_fn = 0
    global_tn = 0

    global_roi_pixels = 0
    global_gt_gf_pixels = 0
    global_pred_gf_pixels = 0

    for position, (_, row) in enumerate(
        test_df.iterrows(),
        start=1
    ):

        # ----------------------------------------------------
        # LOAD DATA
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
        # SAME 6 FEATURES AS TRAINING
        # ----------------------------------------------------

        feature_image = extract_features(
            gray
        )

        flat_features = (
            feature_image.reshape(
                -1,
                6
            )
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Predict only valid ROI pixels.
        # Pixels outside ROI are kept at zero.
        # ----------------------------------------------------

        roi_flat = roi.ravel()

        valid_indices = np.flatnonzero(
            roi_flat
        )

        prediction_flat = np.zeros(
            256 * 256,
            dtype=np.uint8
        )

        if len(valid_indices) > 0:

            valid_features = (
                flat_features[
                    valid_indices
                ]
            )

            # Use scaler fitted ONLY
            # on training pixels.
            valid_features_scaled = (
                scaler.transform(
                    valid_features
                )
            )

            valid_predictions = (
                svm.predict(
                    valid_features_scaled
                )
            )

            prediction_flat[
                valid_indices
            ] = (
                valid_predictions
                .astype(np.uint8)
            )

        prediction = (
            prediction_flat.reshape(
                256,
                256
            )
        )

        # ----------------------------------------------------
        # SAME METRICS AS U-NET
        # ----------------------------------------------------

        metrics = calculate_metrics(
            prediction,
            gf_mask,
            roi
        )

        global_tp += metrics["tp"]
        global_fp += metrics["fp"]
        global_fn += metrics["fn"]
        global_tn += metrics["tn"]

        global_roi_pixels += (
            metrics[
                "valid_roi_pixels"
            ]
        )

        global_gt_gf_pixels += (
            metrics[
                "gt_gf_pixels"
            ]
        )

        global_pred_gf_pixels += (
            metrics[
                "pred_gf_pixels"
            ]
        )

        patch_results.append({
            "patch_id":
                row["patch_id"],

            "sample_id":
                row["sample_id"],

            "compaction_ratio":
                row[
                    "compaction_ratio"
                ],

            "layer":
                int(row["layer"]),

            **metrics
        })

        if (
            position % 25 == 0
            or position
            == len(test_df)
        ):

            print(
                f"Processed "
                f"{position}/"
                f"{len(test_df)} "
                f"test patches"
            )

    # --------------------------------------------------------
    # SAVE PATCH RESULTS
    # --------------------------------------------------------

    patch_df = pd.DataFrame(
        patch_results
    )

    patch_df.to_csv(
        PATCH_RESULTS_PATH,
        index=False
    )

    # --------------------------------------------------------
    # GLOBAL METRICS
    #
    # Same aggregation method used
    # for U-Net evaluation.
    # --------------------------------------------------------

    eps = 1e-8

    global_dice = (
        2.0 * global_tp
        / (
            2.0 * global_tp
            + global_fp
            + global_fn
            + eps
        )
    )

    global_iou = (
        global_tp
        / (
            global_tp
            + global_fp
            + global_fn
            + eps
        )
    )

    global_precision = (
        global_tp
        / (
            global_tp
            + global_fp
            + eps
        )
    )

    global_recall = (
        global_tp
        / (
            global_tp
            + global_fn
            + eps
        )
    )

    global_pixel_accuracy = (
        global_tp
        + global_tn
    ) / (
        global_tp
        + global_tn
        + global_fp
        + global_fn
        + eps
    )

    global_gt_gf_percentage = (
        global_gt_gf_pixels
        / (
            global_roi_pixels
            + eps
        )
        * 100.0
    )

    global_pred_gf_percentage = (
        global_pred_gf_pixels
        / (
            global_roi_pixels
            + eps
        )
        * 100.0
    )

    global_gf_area_error = abs(
        global_pred_gf_percentage
        - global_gt_gf_percentage
    )

    # --------------------------------------------------------
    # SPECIMEN-LEVEL SUMMARY
    # --------------------------------------------------------

    specimen_rows = []

    for sample_id, group in (
        patch_df.groupby(
            "sample_id"
        )
    ):

        first = group.iloc[0]

        specimen_rows.append({
            "sample_id":
                sample_id,

            "compaction_ratio":
                first[
                    "compaction_ratio"
                ],

            "layer":
                int(
                    first["layer"]
                ),

            "number_of_patches":
                len(group),

            "mean_dice":
                group[
                    "dice"
                ].mean(),

            "mean_iou":
                group[
                    "iou"
                ].mean(),

            "mean_precision":
                group[
                    "precision"
                ].mean(),

            "mean_recall":
                group[
                    "recall"
                ].mean(),

            "mean_pixel_accuracy":
                group[
                    "pixel_accuracy"
                ].mean(),

            "mean_gt_gf_percentage":
                group[
                    "gt_gf_percentage"
                ].mean(),

            "mean_pred_gf_percentage":
                group[
                    "pred_gf_percentage"
                ].mean(),

            "mean_gf_area_fraction_error":
                group[
                    "gf_area_fraction_error"
                ].mean()
        })

    specimen_df = pd.DataFrame(
        specimen_rows
    )

    specimen_df.to_csv(
        SPECIMEN_RESULTS_PATH,
        index=False
    )

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL SVM UNTOUCHED TEST RESULTS")
    print("=" * 80)

    print(
        f"\nTest patches     : "
        f"{len(patch_df)}"
    )

    print(
        f"Test specimens   : "
        f"{len(specimen_df)}"
    )

    print(
        f"\nDice             : "
        f"{global_dice:.6f} "
        f"({global_dice * 100:.2f}%)"
    )

    print(
        f"IoU              : "
        f"{global_iou:.6f} "
        f"({global_iou * 100:.2f}%)"
    )

    print(
        f"Precision        : "
        f"{global_precision:.6f} "
        f"({global_precision * 100:.2f}%)"
    )

    print(
        f"Recall           : "
        f"{global_recall:.6f} "
        f"({global_recall * 100:.2f}%)"
    )

    print(
        f"Pixel Accuracy   : "
        f"{global_pixel_accuracy:.6f} "
        f"({global_pixel_accuracy * 100:.2f}%)"
    )

    print(
        f"\nGround-truth GF %: "
        f"{global_gt_gf_percentage:.4f}%"
    )

    print(
        f"Predicted GF %   : "
        f"{global_pred_gf_percentage:.4f}%"
    )

    print(
        f"GF area-fraction error: "
        f"{global_gf_area_error:.4f} "
        f"percentage points"
    )

    print("\n" + "=" * 80)
    print("SPECIMEN-LEVEL SUMMARY")
    print("=" * 80)

    print(
        specimen_df[
            [
                "sample_id",
                "compaction_ratio",
                "number_of_patches",
                "mean_dice",
                "mean_iou",
                "mean_gf_area_fraction_error"
            ]
        ].to_string(
            index=False
        )
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
            "FINAL SVM GLASS-FIBRE "
            "SEGMENTATION TEST RESULTS\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            "Model: LinearSVC\n"
        )

        file.write(
            "Task: Pixel-level GF segmentation\n"
        )

        file.write(
            "Test patches: "
            f"{len(patch_df)}\n"
        )

        file.write(
            "Test specimens: "
            f"{len(specimen_df)}\n\n"
        )

        file.write(
            f"Test Dice: "
            f"{global_dice:.6f}\n"
        )

        file.write(
            f"Test IoU: "
            f"{global_iou:.6f}\n"
        )

        file.write(
            f"Test Precision: "
            f"{global_precision:.6f}\n"
        )

        file.write(
            f"Test Recall: "
            f"{global_recall:.6f}\n"
        )

        file.write(
            f"Test Pixel Accuracy: "
            f"{global_pixel_accuracy:.6f}\n\n"
        )

        file.write(
            f"Ground-truth GF %: "
            f"{global_gt_gf_percentage:.6f}\n"
        )

        file.write(
            f"Predicted GF %: "
            f"{global_pred_gf_percentage:.6f}\n"
        )

        file.write(
            f"GF area-fraction error: "
            f"{global_gf_area_error:.6f} "
            f"percentage points\n"
        )

        file.write(
            "\nTest data was not used "
            "to fit the scaler or SVM.\n"
        )

    print("\n" + "=" * 80)
    print("SVM TEST EVALUATION COMPLETE")
    print("=" * 80)

    print(
        f"\nPatch results:\n"
        f"{PATCH_RESULTS_PATH}"
    )

    print(
        f"\nSpecimen results:\n"
        f"{SPECIMEN_RESULTS_PATH}"
    )

    print(
        f"\nSummary:\n"
        f"{SUMMARY_PATH}"
    )

    print(
        "\nIMPORTANT:"
        "\nThe SVM was evaluated on the same "
        "179 untouched test patches used "
        "for U-Net."
        "\nThe target is the same GF mask."
        "\nEvaluation is restricted to the "
        "same bundle ROI."
    )


if __name__ == "__main__":
    main()