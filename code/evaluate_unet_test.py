from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ORIGINAL_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "final_70_15_15"
    / "unet_dataset_256"
    / "unet_patch_manifest.csv"
)

# Keep this filename because this is the checkpoint
# that actually contains our verified best Epoch-7 model.
MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "final_70_15_15"
    / "unet_32epochs"
    / "best_unet_32epochs.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "final_70_15_15"
    / "unet_test_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PATCH_RESULTS_PATH = (
    OUTPUT_DIR
    / "unet_test_patch_results.csv"
)

SPECIMEN_RESULTS_PATH = (
    OUTPUT_DIR
    / "unet_test_specimen_results.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "unet_test_summary.txt"
)

BATCH_SIZE = 16
NUM_WORKERS = 0
THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET
# ============================================================

class TestDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(drop=True)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        with Image.open(row["image_patch"]) as f:
            image = np.array(f.convert("RGB"))

        with Image.open(row["gf_mask_patch"]) as f:
            mask = np.array(f.convert("L"))

        with Image.open(row["bundle_mask_patch"]) as f:
            roi = np.array(f.convert("L"))

        if image.shape[:2] != (256, 256):
            raise ValueError(
                f"Incorrect image size for {row['patch_id']}: "
                f"{image.shape}"
            )

        if mask.shape != (256, 256):
            raise ValueError(
                f"Incorrect GF mask size for {row['patch_id']}: "
                f"{mask.shape}"
            )

        if roi.shape != (256, 256):
            raise ValueError(
                f"Incorrect bundle mask size for {row['patch_id']}: "
                f"{roi.shape}"
            )

        image = self.to_tensor(image)

        mask = torch.from_numpy(
            (mask > 0).astype(np.float32)
        ).unsqueeze(0)

        roi = torch.from_numpy(
            (roi > 0).astype(np.float32)
        ).unsqueeze(0)

        return (
            image,
            mask,
            roi,
            str(row["patch_id"]),
            str(row["sample_id"]),
            str(row["compaction_ratio"]),
            int(row["layer"])
        )


# ============================================================
# SAME U-NET ARCHITECTURE USED FOR TRAINING
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return self.block(x)


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1):

        super().__init__()

        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.up4 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2
        )
        self.dec4 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(
            64, 32, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv(64, 32)

        self.output = nn.Conv2d(
            32,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        e3 = self.enc3(
            self.pool2(e2)
        )

        e4 = self.enc4(
            self.pool3(e3)
        )

        b = self.bottleneck(
            self.pool4(e4)
        )

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.output(d1)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(prediction, target, roi):

    prediction = prediction.astype(bool)
    target = target.astype(bool)
    roi = roi.astype(bool)

    # Evaluate ONLY inside valid bundle ROI
    prediction = prediction & roi
    target = target & roi

    tp = np.logical_and(
        prediction,
        target
    ).sum()

    fp = np.logical_and(
        prediction,
        np.logical_not(target)
    ) & roi

    fp = fp.sum()

    fn = np.logical_and(
        np.logical_not(prediction),
        target
    ) & roi

    fn = fn.sum()

    tn = np.logical_and(
        np.logical_not(prediction),
        np.logical_not(target)
    ) & roi

    tn = tn.sum()

    eps = 1e-8

    dice = (
        2.0 * tp
        / (2.0 * tp + fp + fn + eps)
    )

    iou = (
        tp
        / (tp + fp + fn + eps)
    )

    precision = (
        tp
        / (tp + fp + eps)
    )

    recall = (
        tp
        / (tp + fn + eps)
    )

    pixel_accuracy = (
        (tp + tn)
        / (tp + tn + fp + fn + eps)
    )

    valid_roi_pixels = roi.sum()

    gt_gf_pixels = target.sum()
    pred_gf_pixels = prediction.sum()

    gt_gf_percentage = (
        gt_gf_pixels
        / (valid_roi_pixels + eps)
        * 100.0
    )

    pred_gf_percentage = (
        pred_gf_pixels
        / (valid_roi_pixels + eps)
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
        "pixel_accuracy": float(pixel_accuracy),

        "valid_roi_pixels":
            int(valid_roi_pixels),

        "gt_gf_pixels":
            int(gt_gf_pixels),

        "pred_gf_pixels":
            int(pred_gf_pixels),

        "gt_gf_percentage":
            float(gt_gf_percentage),

        "pred_gf_percentage":
            float(pred_gf_percentage),

        "gf_area_fraction_error":
            float(gf_area_fraction_error)
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("FINAL U-NET TEST EVALUATION")
    print("=" * 80)

    print(f"\nDevice: {DEVICE}")

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not ORIGINAL_MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{ORIGINAL_MANIFEST}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # LOAD MANIFEST
    # --------------------------------------------------------

    manifest = pd.read_csv(
        ORIGINAL_MANIFEST
    )

    test_df = manifest[
        manifest["split"] == "test"
    ].copy()

    print(
        f"\nUntouched test patches found: "
        f"{len(test_df)}"
    )

    test_specimens = sorted(
        test_df["sample_id"]
        .astype(str)
        .unique()
        .tolist()
    )

    print(
        f"Independent test specimens: "
        f"{len(test_specimens)}"
    )

    print("\nTest specimen IDs:")

    for sample_id in test_specimens:
        print(f"  {sample_id}")

    if len(test_df) != 179:
        raise ValueError(
            f"Expected 179 test patches, "
            f"found {len(test_df)}."
        )

    if len(test_specimens) != 7:
        raise ValueError(
            f"Expected 7 independent test specimens, "
            f"found {len(test_specimens)}."
        )

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    print("\n" + "=" * 80)
    print("MODEL CHECK")
    print("=" * 80)

    print(
        f"\nCheckpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Saved validation Dice: "
        f"{checkpoint['val_dice']:.6f}"
    )

    print(
        f"Saved validation IoU: "
        f"{checkpoint['val_iou']:.6f}"
    )

    # We verified this checkpoint previously.
    if int(checkpoint["epoch"]) != 7:
        raise RuntimeError(
            "Expected the verified best Epoch-7 checkpoint."
        )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = UNet(
        in_channels=3,
        out_channels=1
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # --------------------------------------------------------
    # TEST LOADER
    # --------------------------------------------------------

    test_dataset = TestDataset(
        test_df
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("RUNNING TEST SET")
    print("=" * 80)

    patch_results = []

    global_tp = 0
    global_fp = 0
    global_fn = 0
    global_tn = 0

    global_roi_pixels = 0
    global_gt_gf_pixels = 0
    global_pred_gf_pixels = 0

    with torch.no_grad():

        for batch in test_loader:

            (
                images,
                masks,
                rois,
                patch_ids,
                sample_ids,
                compaction_ratios,
                layers
            ) = batch

            images = images.to(DEVICE)

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= THRESHOLD
            ).cpu().numpy()

            masks_np = masks.numpy()
            rois_np = rois.numpy()

            batch_size = images.size(0)

            for i in range(batch_size):

                prediction = predictions[
                    i, 0
                ]

                target = masks_np[
                    i, 0
                ]

                roi = rois_np[
                    i, 0
                ]

                metrics = calculate_metrics(
                    prediction,
                    target,
                    roi
                )

                global_tp += metrics["tp"]
                global_fp += metrics["fp"]
                global_fn += metrics["fn"]
                global_tn += metrics["tn"]

                global_roi_pixels += (
                    metrics["valid_roi_pixels"]
                )

                global_gt_gf_pixels += (
                    metrics["gt_gf_pixels"]
                )

                global_pred_gf_pixels += (
                    metrics["pred_gf_pixels"]
                )

                patch_results.append({
                    "patch_id":
                        patch_ids[i],

                    "sample_id":
                        sample_ids[i],

                    "compaction_ratio":
                        compaction_ratios[i],

                    "layer":
                        int(layers[i]),

                    **metrics
                })

    # --------------------------------------------------------
    # PATCH RESULTS
    # --------------------------------------------------------

    patch_results_df = pd.DataFrame(
        patch_results
    )

    patch_results_df.to_csv(
        PATCH_RESULTS_PATH,
        index=False
    )

    # --------------------------------------------------------
    # GLOBAL TEST METRICS
    #
    # Calculated from ALL valid test pixels together.
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
        global_tp + global_tn
    ) / (
        global_tp
        + global_tn
        + global_fp
        + global_fn
        + eps
    )

    global_gt_gf_percentage = (
        global_gt_gf_pixels
        / (global_roi_pixels + eps)
        * 100.0
    )

    global_pred_gf_percentage = (
        global_pred_gf_pixels
        / (global_roi_pixels + eps)
        * 100.0
    )

    global_gf_area_error = abs(
        global_pred_gf_percentage
        - global_gt_gf_percentage
    )

    # --------------------------------------------------------
    # SPECIMEN-LEVEL RESULTS
    # --------------------------------------------------------

    specimen_rows = []

    for sample_id, group in patch_results_df.groupby(
        "sample_id"
    ):

        first_row = group.iloc[0]

        specimen_rows.append({
            "sample_id":
                sample_id,

            "compaction_ratio":
                first_row["compaction_ratio"],

            "layer":
                int(first_row["layer"]),

            "number_of_patches":
                len(group),

            "mean_dice":
                group["dice"].mean(),

            "mean_iou":
                group["iou"].mean(),

            "mean_precision":
                group["precision"].mean(),

            "mean_recall":
                group["recall"].mean(),

            "mean_pixel_accuracy":
                group["pixel_accuracy"].mean(),

            "mean_gt_gf_percentage":
                group["gt_gf_percentage"].mean(),

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
    # PRINT FINAL RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL UNTOUCHED TEST RESULTS")
    print("=" * 80)

    print(
        f"\nTest patches     : "
        f"{len(patch_results_df)}"
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
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # SAVE TEXT SUMMARY
    # --------------------------------------------------------

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "FINAL U-NET UNTOUCHED TEST RESULTS\n"
        )

        f.write("=" * 60 + "\n\n")

        f.write(
            f"Best checkpoint epoch: "
            f"{checkpoint['epoch']}\n"
        )

        f.write(
            f"Validation Dice of checkpoint: "
            f"{checkpoint['val_dice']:.6f}\n"
        )

        f.write(
            f"Validation IoU of checkpoint: "
            f"{checkpoint['val_iou']:.6f}\n\n"
        )

        f.write(
            f"Test patches: "
            f"{len(patch_results_df)}\n"
        )

        f.write(
            f"Test specimens: "
            f"{len(specimen_df)}\n\n"
        )

        f.write(
            f"Test Dice: "
            f"{global_dice:.6f}\n"
        )

        f.write(
            f"Test IoU: "
            f"{global_iou:.6f}\n"
        )

        f.write(
            f"Test Precision: "
            f"{global_precision:.6f}\n"
        )

        f.write(
            f"Test Recall: "
            f"{global_recall:.6f}\n"
        )

        f.write(
            f"Test Pixel Accuracy: "
            f"{global_pixel_accuracy:.6f}\n\n"
        )

        f.write(
            f"Ground-truth GF %: "
            f"{global_gt_gf_percentage:.6f}\n"
        )

        f.write(
            f"Predicted GF %: "
            f"{global_pred_gf_percentage:.6f}\n"
        )

        f.write(
            f"GF area-fraction error: "
            f"{global_gf_area_error:.6f} "
            f"percentage points\n"
        )

    print("\n" + "=" * 80)
    print("TEST EVALUATION COMPLETE")
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
        "\nThese are TEST results from the "
        "179 untouched test patches."
        "\nThey are separate from validation results."
        "\nNo test data was used to train the model."
    )


if __name__ == "__main__":
    main()