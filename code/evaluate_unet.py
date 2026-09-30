from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image

import torch
import matplotlib.pyplot as plt

# ============================================================
# IMPORT U-NET ARCHITECTURE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT / "code")
)

from train_unet import UNet


# ============================================================
# CONFIGURATION
# ============================================================

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "unet_dataset"
    / "unet_patch_manifest.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "unet"
    / "best_unet_8epochs.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "unet_test"
)

PREDICTION_DIR = (
    OUTPUT_DIR
    / "predicted_masks"
)

VISUALIZATION_DIR = (
    OUTPUT_DIR
    / "visualizations"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VISUALIZATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

THRESHOLD = 0.5


# ============================================================
# LOAD IMAGE
# ============================================================

def load_rgb(path):

    image = Image.open(
        path
    ).convert("RGB")

    array = np.array(
        image
    )

    tensor = torch.from_numpy(
        array
    ).permute(
        2, 0, 1
    ).float() / 255.0

    return array, tensor


# ============================================================
# LOAD BINARY MASK
# ============================================================

def load_mask(path):

    mask = Image.open(
        path
    ).convert("L")

    array = (
        np.array(mask) > 0
    ).astype(np.uint8)

    tensor = torch.from_numpy(
        array.astype(np.float32)
    ).unsqueeze(0)

    return array, tensor


# ============================================================
# METRIC COUNTS
# ============================================================

def calculate_counts(
    prediction,
    target,
    roi
):

    prediction = (
        prediction.astype(bool)
    )

    target = (
        target.astype(bool)
    )

    roi = (
        roi.astype(bool)
    )

    # Only evaluate inside bundle ROI

    pred_roi = (
        prediction & roi
    )

    target_roi = (
        target & roi
    )

    tp = np.logical_and(
        pred_roi,
        target_roi
    ).sum()

    fp = np.logical_and(
        pred_roi,
        np.logical_and(
            np.logical_not(target_roi),
            roi
        )
    ).sum()

    fn = np.logical_and(
        np.logical_not(pred_roi),
        target_roi
    ).sum()

    tn = np.logical_and(
        np.logical_not(pred_roi),
        np.logical_and(
            np.logical_not(target_roi),
            roi
        )
    ).sum()

    return (
        int(tp),
        int(fp),
        int(fn),
        int(tn)
    )


# ============================================================
# METRICS
# ============================================================

def metrics_from_counts(
    tp,
    fp,
    fn,
    tn
):

    eps = 1e-8

    dice = (
        2 * tp
        /
        (2 * tp + fp + fn + eps)
    )

    iou = (
        tp
        /
        (tp + fp + fn + eps)
    )

    precision = (
        tp
        /
        (tp + fp + eps)
    )

    recall = (
        tp
        /
        (tp + fn + eps)
    )

    accuracy = (
        tp + tn
    ) / (
        tp + tn + fp + fn + eps
    )

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": dice,
        "accuracy": accuracy
    }


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_overlay(
    image,
    prediction,
    roi
):

    overlay = image.copy().astype(
        np.float32
    )

    prediction = (
        (prediction > 0)
        & (roi > 0)
    )

    # Green overlay for predicted glass fibre

    green = np.zeros_like(
        overlay
    )

    green[:, :, 1] = 255

    alpha = 0.40

    overlay[prediction] = (
        (1 - alpha)
        * overlay[prediction]
        +
        alpha
        * green[prediction]
    )

    return np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# SAVE VISUALIZATION
# ============================================================

def save_visualization(
    image,
    target,
    prediction,
    overlay,
    patch_id,
    metrics
):

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4)
    )

    axes[0].imshow(
        image
    )

    axes[0].set_title(
        "Original SEM"
    )

    axes[1].imshow(
        target,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[1].set_title(
        "Ground Truth GF Mask"
    )

    axes[2].imshow(
        prediction,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[2].set_title(
        "Predicted GF Mask"
    )

    axes[3].imshow(
        overlay
    )

    axes[3].set_title(
        "Prediction Overlay"
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{patch_id}\n"
        f"Dice={metrics['dice']:.4f} | "
        f"IoU={metrics['iou']:.4f}"
    )

    plt.tight_layout()

    output_path = (
        VISUALIZATION_DIR
        / f"{patch_id}_visualization.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("FINAL U-NET TEST EVALUATION")
    print("=" * 72)

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        f"Threshold: {THRESHOLD}"
    )

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"\nManifest not found:\n"
            f"{MANIFEST_PATH}"
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nBest U-Net model not found:\n"
            f"{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # LOAD TEST MANIFEST
    # --------------------------------------------------------

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    test_df = manifest[
        manifest["split"] == "test"
    ].copy()

    print(
        f"\nTest patches: "
        f"{len(test_df)}"
    )

    print(
        f"Test original specimens: "
        f"{test_df['sample_id'].nunique()}"
    )

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    model = UNet(
        in_channels=3,
        out_channels=1
    ).to(
        DEVICE
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        f"\nLoaded best model "
        f"from epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Saved validation Dice: "
        f"{checkpoint['val_dice']:.4f}"
    )

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    patch_results = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    # specimen-level accumulated counts

    specimen_counts = {}

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print(
        "\nEvaluating held-out "
        "segmentation test set..."
    )

    with torch.no_grad():

        for index, row in test_df.iterrows():

            patch_id = row[
                "patch_id"
            ]

            sample_id = row[
                "sample_id"
            ]

            # --------------------------------------------
            # LOAD
            # --------------------------------------------

            image_np, image_tensor = (
                load_rgb(
                    row["image_patch"]
                )
            )

            target_np, _ = (
                load_mask(
                    row["gf_mask_patch"]
                )
            )

            roi_np, _ = (
                load_mask(
                    row["bundle_mask_patch"]
                )
            )

            # --------------------------------------------
            # INFERENCE
            # --------------------------------------------

            image_tensor = (
                image_tensor
                .unsqueeze(0)
                .to(DEVICE)
            )

            logits = model(
                image_tensor
            )

            probability = (
                torch.sigmoid(
                    logits
                )
                .squeeze()
                .cpu()
                .numpy()
            )

            prediction = (
                probability
                >= THRESHOLD
            ).astype(
                np.uint8
            )

            # Restrict final mask to bundle

            prediction = (
                prediction
                * roi_np
            )

            # --------------------------------------------
            # METRICS
            # --------------------------------------------

            tp, fp, fn, tn = (
                calculate_counts(
                    prediction,
                    target_np,
                    roi_np
                )
            )

            metrics = (
                metrics_from_counts(
                    tp,
                    fp,
                    fn,
                    tn
                )
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            # --------------------------------------------
            # SPECIMEN COUNTS
            # --------------------------------------------

            if (
                sample_id
                not in specimen_counts
            ):

                specimen_counts[
                    sample_id
                ] = {
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "tn": 0
                }

            specimen_counts[
                sample_id
            ]["tp"] += tp

            specimen_counts[
                sample_id
            ]["fp"] += fp

            specimen_counts[
                sample_id
            ]["fn"] += fn

            specimen_counts[
                sample_id
            ]["tn"] += tn

            # --------------------------------------------
            # AREA FRACTIONS INSIDE BUNDLE
            # --------------------------------------------

            roi_pixels = (
                roi_np > 0
            ).sum()

            gt_gf_pixels = np.logical_and(
                target_np > 0,
                roi_np > 0
            ).sum()

            pred_gf_pixels = np.logical_and(
                prediction > 0,
                roi_np > 0
            ).sum()

            gt_gf_fraction = (
                gt_gf_pixels
                / roi_pixels
                if roi_pixels > 0
                else 0.0
            )

            pred_gf_fraction = (
                pred_gf_pixels
                / roi_pixels
                if roi_pixels > 0
                else 0.0
            )

            area_error = (
                pred_gf_fraction
                - gt_gf_fraction
            )

            # --------------------------------------------
            # SAVE PREDICTED MASK
            # --------------------------------------------

            prediction_image = (
                prediction * 255
            ).astype(
                np.uint8
            )

            Image.fromarray(
                prediction_image
            ).save(
                PREDICTION_DIR
                / f"{patch_id}_predicted.png"
            )

            # --------------------------------------------
            # OVERLAY
            # --------------------------------------------

            overlay = create_overlay(
                image_np,
                prediction,
                roi_np
            )

            # --------------------------------------------
            # SAVE SELECTED VISUALIZATIONS
            #
            # Save first 12 only to avoid creating
            # hundreds of large figures.
            # --------------------------------------------

            if len(
                patch_results
            ) < 12:

                save_visualization(
                    image_np,
                    target_np,
                    prediction,
                    overlay,
                    patch_id,
                    metrics
                )

            # --------------------------------------------
            # PATCH RESULT
            # --------------------------------------------

            patch_results.append({

                "sample_id":
                    sample_id,

                "patch_id":
                    patch_id,

                "dice":
                    metrics["dice"],

                "iou":
                    metrics["iou"],

                "precision":
                    metrics["precision"],

                "recall":
                    metrics["recall"],

                "f1":
                    metrics["f1"],

                "pixel_accuracy":
                    metrics["accuracy"],

                "gt_gf_fraction":
                    gt_gf_fraction,

                "pred_gf_fraction":
                    pred_gf_fraction,

                "gf_fraction_error":
                    area_error,

                "tp":
                    tp,

                "fp":
                    fp,

                "fn":
                    fn,

                "tn":
                    tn
            })

    # ========================================================
    # OVERALL TEST METRICS
    # ========================================================

    overall_metrics = (
        metrics_from_counts(
            total_tp,
            total_fp,
            total_fn,
            total_tn
        )
    )

    # ========================================================
    # SPECIMEN-LEVEL RESULTS
    # ========================================================

    specimen_results = []

    for (
        sample_id,
        counts
    ) in specimen_counts.items():

        metrics = (
            metrics_from_counts(
                counts["tp"],
                counts["fp"],
                counts["fn"],
                counts["tn"]
            )
        )

        specimen_results.append({

            "sample_id":
                sample_id,

            "dice":
                metrics["dice"],

            "iou":
                metrics["iou"],

            "precision":
                metrics["precision"],

            "recall":
                metrics["recall"],

            "f1":
                metrics["f1"],

            "pixel_accuracy":
                metrics["accuracy"]
        })

    # ========================================================
    # SAVE CSV FILES
    # ========================================================

    patch_results_df = pd.DataFrame(
        patch_results
    )

    specimen_results_df = pd.DataFrame(
        specimen_results
    )

    patch_results_path = (
        OUTPUT_DIR
        / "test_patch_metrics.csv"
    )

    specimen_results_path = (
        OUTPUT_DIR
        / "test_specimen_metrics.csv"
    )

    patch_results_df.to_csv(
        patch_results_path,
        index=False
    )

    specimen_results_df.to_csv(
        specimen_results_path,
        index=False
    )

    # ========================================================
    # AREA-FRACTION SUMMARY
    # ========================================================

    mean_abs_area_error = (
        patch_results_df[
            "gf_fraction_error"
        ]
        .abs()
        .mean()
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "FINAL HELD-OUT TEST RESULTS"
    )

    print(
        "=" * 72
    )

    print(
        f"\nDice: "
        f"{overall_metrics['dice']:.4f}"
    )

    print(
        f"IoU: "
        f"{overall_metrics['iou']:.4f}"
    )

    print(
        f"Precision: "
        f"{overall_metrics['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{overall_metrics['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{overall_metrics['f1']:.4f}"
    )

    print(
        f"Pixel Accuracy: "
        f"{overall_metrics['accuracy']:.4f}"
    )

    print(
        "\nMean absolute GF area-fraction error:"
    )

    print(
        f"{mean_abs_area_error:.4f} "
        f"({mean_abs_area_error * 100:.2f} percentage points)"
    )

    # ========================================================
    # SPECIMEN RESULTS
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "TEST RESULTS BY ORIGINAL SPECIMEN"
    )

    print(
        "=" * 72
    )

    for _, row in (
        specimen_results_df
        .sort_values(
            "sample_id"
        )
        .iterrows()
    ):

        print(
            f"\n{row['sample_id']}: "
            f"Dice={row['dice']:.4f}, "
            f"IoU={row['iou']:.4f}"
        )

    # ========================================================
    # SAVE SUMMARY TXT
    # ========================================================

    summary_path = (
        OUTPUT_DIR
        / "final_unet_test_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "FINAL U-NET SEGMENTATION TEST RESULTS\n"
        )

        f.write(
            "=" * 50
            + "\n\n"
        )

        f.write(
            f"Test patches: "
            f"{len(test_df)}\n"
        )

        f.write(
            f"Test original specimens: "
            f"{test_df['sample_id'].nunique()}\n\n"
        )

        f.write(
            f"Dice: "
            f"{overall_metrics['dice']:.6f}\n"
        )

        f.write(
            f"IoU: "
            f"{overall_metrics['iou']:.6f}\n"
        )

        f.write(
            f"Precision: "
            f"{overall_metrics['precision']:.6f}\n"
        )

        f.write(
            f"Recall: "
            f"{overall_metrics['recall']:.6f}\n"
        )

        f.write(
            f"F1: "
            f"{overall_metrics['f1']:.6f}\n"
        )

        f.write(
            f"Pixel Accuracy: "
            f"{overall_metrics['accuracy']:.6f}\n"
        )

        f.write(
            "\nMean absolute GF area-fraction error: "
            f"{mean_abs_area_error:.6f}\n"
        )

        f.write(
            "\nModel checkpoint:\n"
        )

        f.write(
            str(MODEL_PATH)
            + "\n"
        )

    # ========================================================
    # FINAL PATHS
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 72
    )

    print(
        "\nPatch metrics:"
    )

    print(
        patch_results_path
    )

    print(
        "\nSpecimen metrics:"
    )

    print(
        specimen_results_path
    )

    print(
        "\nSummary:"
    )

    print(
        summary_path
    )

    print(
        "\nPredicted masks:"
    )

    print(
        PREDICTION_DIR
    )

    print(
        "\nVisualizations:"
    )

    print(
        VISUALIZATION_DIR
    )

    print(
        "\nFINAL TEST EVALUATION COMPLETE."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()