from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 7000 training pairs created in STEP 5
TRAIN_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "final_70_15_15"
    / "unet_dataset_augmented"
    / "augmented_train_manifest.csv"
)

# ORIGINAL patch manifest.
# Validation is loaded from here so validation remains untouched.
ORIGINAL_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "final_70_15_15"
    / "unet_dataset_256"
    / "unet_patch_manifest.csv"
)

# Keep all new model outputs separate from old experiment.
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "final_70_15_15"
    / "unet_32epochs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    OUTPUT_DIR
    / "best_unet_32epochs.pth"
)

HISTORY_PATH = (
    OUTPUT_DIR
    / "training_history_32epochs.csv"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 16

# Professor requirement:
# epoch count must be a multiple of 8.
EPOCHS = 8

LEARNING_RATE = 1e-3

NUM_WORKERS = 0

SEED = 42

THRESHOLD = 0.5


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# DATASET
# ============================================================

class SEMSegmentationDataset(Dataset):

    def __init__(
        self,
        dataframe
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.to_tensor = transforms.ToTensor()

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        # ----------------------------------------------------
        # Load SEM
        # ----------------------------------------------------

        with Image.open(
            row["image_patch"]
        ) as image_file:

            image = np.array(
                image_file.convert("RGB")
            )

        # ----------------------------------------------------
        # Load GF target
        # ----------------------------------------------------

        with Image.open(
            row["gf_mask_patch"]
        ) as mask_file:

            mask = np.array(
                mask_file.convert("L")
            )

        # ----------------------------------------------------
        # Load bundle ROI
        # ----------------------------------------------------

        with Image.open(
            row["bundle_mask_patch"]
        ) as roi_file:

            roi = np.array(
                roi_file.convert("L")
            )

        # ----------------------------------------------------
        # Verify shape
        # ----------------------------------------------------

        if image.shape[:2] != (256, 256):

            raise ValueError(
                f"Incorrect SEM size: "
                f"{image.shape}"
            )

        if mask.shape != (256, 256):

            raise ValueError(
                f"Incorrect GF mask size: "
                f"{mask.shape}"
            )

        if roi.shape != (256, 256):

            raise ValueError(
                f"Incorrect ROI size: "
                f"{roi.shape}"
            )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # NO online augmentation here.
        #
        # Training augmentation has already been performed
        # and saved during STEP 5.
        # Validation must remain untouched.
        # ----------------------------------------------------

        image = self.to_tensor(
            image
        )

        mask = torch.from_numpy(
            (mask > 0)
            .astype(np.float32)
        ).unsqueeze(0)

        roi = torch.from_numpy(
            (roi > 0)
            .astype(np.float32)
        ).unsqueeze(0)

        return (
            image,
            mask,
            roi
        )


# ============================================================
# U-NET BUILDING BLOCK
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# U-NET
# ============================================================

class UNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=1
    ):

        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(
            in_channels,
            32
        )

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(
            32,
            64
        )

        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(
            64,
            128
        )

        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(
            128,
            256
        )

        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(
            256,
            512
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(
            512,
            256
        )

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            64,
            32
        )

        # One output channel:
        # GF vs non-GF
        self.output = nn.Conv2d(
            32,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
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

        # Bottleneck
        b = self.bottleneck(
            self.pool4(e4)
        )

        # Decoder
        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        return self.output(d1)


# ============================================================
# ROI-AWARE BCE LOSS
# ============================================================

bce_loss_function = nn.BCEWithLogitsLoss(
    reduction="none"
)


def roi_bce_loss(
    logits,
    targets,
    roi
):

    pixel_loss = bce_loss_function(
        logits,
        targets
    )

    pixel_loss = (
        pixel_loss
        * roi
    )

    return (
        pixel_loss.sum()
        /
        (roi.sum() + 1e-8)
    )


# ============================================================
# ROI-AWARE DICE LOSS
# ============================================================

def dice_loss(
    logits,
    targets,
    roi,
    smooth=1.0
):

    probabilities = torch.sigmoid(
        logits
    )

    probabilities = (
        probabilities
        * roi
    )

    targets = (
        targets
        * roi
    )

    intersection = (
        probabilities
        * targets
    ).sum()

    denominator = (
        probabilities.sum()
        + targets.sum()
    )

    dice = (
        (2.0 * intersection + smooth)
        /
        (denominator + smooth)
    )

    return 1.0 - dice


# ============================================================
# COMBINED BCE + DICE LOSS
# ============================================================

def combined_loss(
    logits,
    targets,
    roi
):

    bce = roi_bce_loss(
        logits,
        targets,
        roi
    )

    dice = dice_loss(
        logits,
        targets,
        roi
    )

    return bce + dice


# ============================================================
# METRIC COUNTS
# ============================================================

def calculate_counts(
    logits,
    targets,
    roi,
    threshold=0.5
):

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= threshold
    ).float()

    # Restrict both prediction and GT evaluation
    # to valid bundle ROI.
    predictions = (
        predictions
        * roi
    )

    targets = (
        targets
        * roi
    )

    tp = (
        predictions
        * targets
    ).sum().item()

    fp = (
        predictions
        * (1 - targets)
        * roi
    ).sum().item()

    fn = (
        (1 - predictions)
        * targets
        * roi
    ).sum().item()

    tn = (
        (1 - predictions)
        * (1 - targets)
        * roi
    ).sum().item()

    return (
        tp,
        fp,
        fn,
        tn
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

    pixel_accuracy = (
        tp + tn
    ) / (
        tp + tn + fp + fn + eps
    )

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "pixel_accuracy": pixel_accuracy
    }


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer
):

    model.train()

    running_loss = 0.0

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_tn = 0.0

    for (
        images,
        masks,
        rois
    ) in loader:

        images = images.to(
            DEVICE
        )

        masks = masks.to(
            DEVICE
        )

        rois = rois.to(
            DEVICE
        )

        optimizer.zero_grad()

        logits = model(
            images
        )

        loss = combined_loss(
            logits,
            masks,
            rois
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        (
            tp,
            fp,
            fn,
            tn

        ) = calculate_counts(
            logits.detach(),
            masks,
            rois,
            THRESHOLD
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    metrics = metrics_from_counts(
        total_tp,
        total_fp,
        total_fn,
        total_tn
    )

    return (
        epoch_loss,
        metrics
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader
):

    model.eval()

    running_loss = 0.0

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_tn = 0.0

    with torch.no_grad():

        for (
            images,
            masks,
            rois
        ) in loader:

            images = images.to(
                DEVICE
            )

            masks = masks.to(
                DEVICE
            )

            rois = rois.to(
                DEVICE
            )

            logits = model(
                images
            )

            loss = combined_loss(
                logits,
                masks,
                rois
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            (
                tp,
                fp,
                fn,
                tn

            ) = calculate_counts(
                logits,
                masks,
                rois,
                THRESHOLD
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    metrics = metrics_from_counts(
        total_tp,
        total_fp,
        total_fn,
        total_tn
    )

    return (
        epoch_loss,
        metrics
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("FINAL U-NET GLASS-FIBRE SEGMENTATION EXPERIMENT")
    print("=" * 80)

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        f"PyTorch version: "
        f"{torch.__version__}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    print(
        f"\nEpochs: {EPOCHS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        "\nLoss:"
        "\nROI-aware BCE + Dice Loss"
    )

    print(
        "\nModel selection:"
        "\nHighest validation Dice"
    )

    print(
        "\nIMPORTANT:"
        "\nThe test dataset is NOT loaded "
        "during training."
    )

    # ========================================================
    # CHECK MANIFESTS
    # ========================================================

    if not TRAIN_MANIFEST.exists():

        raise FileNotFoundError(
            f"\nTraining manifest not found:\n"
            f"{TRAIN_MANIFEST}"
        )

    if not ORIGINAL_MANIFEST.exists():

        raise FileNotFoundError(
            f"\nOriginal manifest not found:\n"
            f"{ORIGINAL_MANIFEST}"
        )

    # ========================================================
    # LOAD 7000 TRAINING PAIRS
    # ========================================================

    train_df = pd.read_csv(
        TRAIN_MANIFEST
    )

    # The augmented manifest contains only training data.
    if len(train_df) != 7000:

        raise ValueError(
            f"Expected exactly 7000 training pairs, "
            f"but found {len(train_df)}."
        )

    # ========================================================
    # LOAD ORIGINAL VALIDATION ONLY
    # ========================================================

    original_df = pd.read_csv(
        ORIGINAL_MANIFEST
    )

    val_df = original_df[
        original_df["split"]
        == "validation"
    ].copy()

    if len(val_df) != 173:

        raise ValueError(
            f"Expected 173 untouched validation patches, "
            f"but found {len(val_df)}."
        )

    # ========================================================
    # IMPORTANT TEST CHECK
    # ========================================================

    test_count = int(
        (
            original_df["split"]
            == "test"
        ).sum()
    )

    if test_count != 179:

        raise ValueError(
            f"Expected 179 untouched test patches, "
            f"found {test_count}."
        )

    print("\n" + "=" * 80)
    print("DATASET CHECK")
    print("=" * 80)

    print(
        f"\nTraining pairs     : "
        f"{len(train_df)}"
    )

    print(
        f"Validation patches : "
        f"{len(val_df)}"
    )

    print(
        f"Test patches found : "
        f"{test_count}"
    )

    print(
        "\nTest patches are verified "
        "but NOT loaded into a DataLoader."
    )

    # ========================================================
    # CHECK TRAINING SPECIMENS
    # ========================================================

    train_specimens = set(
        train_df["sample_id"]
        .astype(str)
    )

    val_specimens = set(
        val_df["sample_id"]
        .astype(str)
    )

    test_specimens = set(
        original_df.loc[
            original_df["split"] == "test",
            "sample_id"
        ].astype(str)
    )

    print(
        f"\nIndependent training specimens   : "
        f"{len(train_specimens)}"
    )

    print(
        f"Independent validation specimens : "
        f"{len(val_specimens)}"
    )

    print(
        f"Independent test specimens       : "
        f"{len(test_specimens)}"
    )

    # ========================================================
    # LEAKAGE CHECK AGAIN
    # ========================================================

    train_val_overlap = (
        train_specimens
        & val_specimens
    )

    train_test_overlap = (
        train_specimens
        & test_specimens
    )

    val_test_overlap = (
        val_specimens
        & test_specimens
    )

    print("\n" + "=" * 80)
    print("DATA LEAKAGE CHECK BEFORE TRAINING")
    print("=" * 80)

    print(
        "TRAIN ∩ VALIDATION =",
        sorted(train_val_overlap)
    )

    print(
        "TRAIN ∩ TEST       =",
        sorted(train_test_overlap)
    )

    print(
        "VALIDATION ∩ TEST  =",
        sorted(val_test_overlap)
    )

    if (
        train_val_overlap
        or train_test_overlap
        or val_test_overlap
    ):

        raise RuntimeError(
            "Data leakage detected."
        )

    if len(train_specimens) != 34:

        raise RuntimeError(
            "Expected 34 independent "
            "training specimens."
        )

    if len(val_specimens) != 7:

        raise RuntimeError(
            "Expected 7 validation specimens."
        )

    if len(test_specimens) != 7:

        raise RuntimeError(
            "Expected 7 test specimens."
        )

    print(
        "\nPASS: No specimen-level "
        "data leakage detected."
    )

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = SEMSegmentationDataset(
        train_df
    )

    val_dataset = SEMSegmentationDataset(
        val_df
    )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_generator = torch.Generator()

    train_generator.manual_seed(
        SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
        generator=train_generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = UNet(
        in_channels=3,
        out_channels=1
    ).to(
        DEVICE
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"\nTrainable parameters: "
        f"{trainable_parameters:,}"
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ========================================================
    # TRAINING
    # ========================================================

    history = []

    best_val_dice = -1.0
    best_val_iou = -1.0
    best_epoch = 0

    training_start = time.time()

    print("\n" + "=" * 80)
    print("TRAINING STARTED")
    print("=" * 80)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        epoch_start = time.time()

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        (
            train_loss,
            train_metrics

        ) = train_one_epoch(
            model,
            train_loader,
            optimizer
        )

        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        (
            val_loss,
            val_metrics

        ) = validate(
            model,
            val_loader
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        history.append({

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "train_dice":
                train_metrics["dice"],

            "train_iou":
                train_metrics["iou"],

            "train_precision":
                train_metrics["precision"],

            "train_recall":
                train_metrics["recall"],

            "train_pixel_accuracy":
                train_metrics[
                    "pixel_accuracy"
                ],

            "val_loss":
                val_loss,

            "val_dice":
                val_metrics["dice"],

            "val_iou":
                val_metrics["iou"],

            "val_precision":
                val_metrics["precision"],

            "val_recall":
                val_metrics["recall"],

            "val_pixel_accuracy":
                val_metrics[
                    "pixel_accuracy"
                ],

            "epoch_seconds":
                epoch_time
        })

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print(
            f"\nEpoch [{epoch}/{EPOCHS}]"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
            f" | Dice: "
            f"{train_metrics['dice']:.4f}"
            f" | IoU: "
            f"{train_metrics['iou']:.4f}"
        )

        print(
            f"Val   Loss: {val_loss:.4f}"
            f" | Dice: "
            f"{val_metrics['dice']:.4f}"
            f" | IoU: "
            f"{val_metrics['iou']:.4f}"
        )

        print(
            f"Val Precision: "
            f"{val_metrics['precision']:.4f}"
            f" | Recall: "
            f"{val_metrics['recall']:.4f}"
            f" | Pixel Accuracy: "
            f"{val_metrics['pixel_accuracy']:.4f}"
        )

        print(
            f"Epoch time: "
            f"{epoch_time:.1f} sec"
        )

        # ----------------------------------------------------
        # SAVE BEST VALIDATION CHECKPOINT
        # ----------------------------------------------------

        if (
            val_metrics["dice"]
            > best_val_dice
        ):

            best_val_dice = (
                val_metrics["dice"]
            )

            best_val_iou = (
                val_metrics["iou"]
            )

            best_epoch = epoch

            torch.save(
                {
                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "val_dice":
                        best_val_dice,

                    "val_iou":
                        best_val_iou,

                    "patch_size":
                        256,

                    "threshold":
                        THRESHOLD,

                    "epochs_planned":
                        EPOCHS,

                    "target":
                        "SEM glass-fibre segmentation",

                    "roi":
                        "SEM bundle mask",

                    "loss":
                        "ROI-aware BCE + Dice",

                    "training_pairs":
                        len(train_df),

                    "validation_patches":
                        len(val_df),

                    "independent_train_specimens":
                        len(train_specimens),

                    "independent_validation_specimens":
                        len(val_specimens),

                    "independent_test_specimens":
                        len(test_specimens)
                },
                MODEL_PATH
            )

            print(
                "✓ New best validation "
                "checkpoint saved."
            )

        # Save history after EVERY epoch
        pd.DataFrame(
            history
        ).to_csv(
            HISTORY_PATH,
            index=False
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total_time = (
        time.time()
        - training_start
    )

    print("\n" + "=" * 80)
    print("32-EPOCH U-NET TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"\nBest epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation Dice: "
        f"{best_val_dice:.4f}"
    )

    print(
        f"Best validation IoU: "
        f"{best_val_iou:.4f}"
    )

    print(
        f"\nTotal training time: "
        f"{total_time / 60:.2f} minutes"
    )

    print(
        f"\nBest model:\n"
        f"{MODEL_PATH}"
    )

    print(
        f"\nTraining history:\n"
        f"{HISTORY_PATH}"
    )

    print(
        "\nIMPORTANT:"
        "\n- 32 epochs were completed."
        "\n- The best checkpoint was selected using VALIDATION Dice."
        "\n- The 179 test patches were NOT used for training."
        "\n- Final test performance has NOT been calculated yet."
        "\n- Do not report validation Dice as test Dice."
    )


if __name__ == "__main__":
    main()