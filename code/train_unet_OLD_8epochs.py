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

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "unet_dataset"
    / "unet_patch_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "unet"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    OUTPUT_DIR
    / "best_unet_8epochs.pth"
)

HISTORY_PATH = (
    OUTPUT_DIR
    / "training_history_8epochs.csv"
)

BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 1e-3
NUM_WORKERS = 0

SEED = 42

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
        dataframe,
        augment=False
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.augment = augment

        self.to_tensor = transforms.ToTensor()

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        # --------------------------------------------
        # Load SEM image
        # --------------------------------------------

        image = Image.open(
            row["image_patch"]
        ).convert("RGB")

        # --------------------------------------------
        # Load GF target mask
        # --------------------------------------------

        mask = Image.open(
            row["gf_mask_patch"]
        ).convert("L")

        # --------------------------------------------
        # Load bundle ROI
        # --------------------------------------------

        roi = Image.open(
            row["bundle_mask_patch"]
        ).convert("L")

        # --------------------------------------------
        # Convert to NumPy
        # --------------------------------------------

        image = np.array(image)
        mask = np.array(mask)
        roi = np.array(roi)

        # --------------------------------------------
        # Synchronized augmentation
        #
        # Image and masks MUST undergo the exact
        # same transformation.
        # --------------------------------------------

        if self.augment:

            # Horizontal flip
            if random.random() > 0.5:

                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
                roi = np.fliplr(roi).copy()

            # Vertical flip
            if random.random() > 0.5:

                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()
                roi = np.flipud(roi).copy()

            # Random 90-degree rotation
            k = random.randint(0, 3)

            if k > 0:

                image = np.rot90(
                    image,
                    k
                ).copy()

                mask = np.rot90(
                    mask,
                    k
                ).copy()

                roi = np.rot90(
                    roi,
                    k
                ).copy()

        # --------------------------------------------
        # Image tensor
        # Values become 0-1
        # --------------------------------------------

        image = self.to_tensor(
            image
        )

        # --------------------------------------------
        # Binary target mask
        # --------------------------------------------

        mask = torch.from_numpy(
            (mask > 0)
            .astype(np.float32)
        ).unsqueeze(0)

        # --------------------------------------------
        # Binary bundle ROI
        # --------------------------------------------

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
# U-NET MODEL
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

        # Binary segmentation output

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

        # Decoder + skip connections

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
# COMBINED LOSS
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

    return (
        bce
        + dice
    )


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
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

    # Restrict evaluation to bundle ROI
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

    return tp, fp, fn, tn


# ============================================================
# METRICS FROM COUNTS
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
        "accuracy": accuracy
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

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

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

        tp, fp, fn, tn = (
            calculate_metrics(
                logits.detach(),
                masks,
                rois
            )
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

    return epoch_loss, metrics


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader
):

    model.eval()

    running_loss = 0.0

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

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

            tp, fp, fn, tn = (
                calculate_metrics(
                    logits,
                    masks,
                    rois
                )
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

    return epoch_loss, metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("U-NET GLASS-FIBRE SEGMENTATION")
    print("=" * 72)

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        "\nLoss:"
        "\nROI-aware BCE + Dice"
    )

    print(
        "\nModel selection:"
        "\nBest validation Dice"
    )

    # --------------------------------------------------------
    # LOAD MANIFEST
    # --------------------------------------------------------

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"\nManifest not found:\n"
            f"{MANIFEST_PATH}"
        )

    df = pd.read_csv(
        MANIFEST_PATH
    )

    train_df = df[
        df["split"] == "train"
    ].copy()

    val_df = df[
        df["split"] == "validation"
    ].copy()

    # IMPORTANT:
    # Test data is intentionally NOT loaded here.

    print(
        "\nDataset:"
    )

    print(
        f"Train patches: "
        f"{len(train_df)}"
    )

    print(
        f"Validation patches: "
        f"{len(val_df)}"
    )

    print(
        "\nTest split is NOT used "
        "during training/model selection."
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = (
        SEMSegmentationDataset(
            train_df,
            augment=True
        )
    )

    val_dataset = (
        SEMSegmentationDataset(
            val_df,
            augment=False
        )
    )

    # --------------------------------------------------------
    # DATA LOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        )
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

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = UNet(
        in_channels=3,
        out_channels=1
    ).to(
        DEVICE
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nTrainable parameters: "
        f"{trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    history = []

    best_val_dice = -1.0
    best_epoch = 0

    training_start = time.time()

    print(
        "\n"
        + "=" * 72
    )

    print(
        "TRAINING STARTED"
    )

    print(
        "=" * 72
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        epoch_start = time.time()

        # TRAIN

        train_loss, train_metrics = (
            train_one_epoch(
                model,
                train_loader,
                optimizer
            )
        )

        # VALIDATE

        val_loss, val_metrics = (
            validate(
                model,
                val_loader
            )
        )

        epoch_time = (
            time.time()
            - epoch_start
        )

        # ----------------------------------------------------
        # SAVE HISTORY
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

            "train_accuracy":
                train_metrics["accuracy"],

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

            "val_accuracy":
                val_metrics["accuracy"],

            "epoch_seconds":
                epoch_time
        })

        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        print(
            f"\nEpoch "
            f"[{epoch}/{EPOCHS}]"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
            f" | Dice: "
            f"{train_metrics['dice']:.4f}"
            f" | IoU: "
            f"{train_metrics['iou']:.4f}"
        )

        print(
            f"Val   Loss: "
            f"{val_loss:.4f}"
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
            f" | Accuracy: "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"Epoch time: "
            f"{epoch_time:.1f} sec"
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL BY VALIDATION DICE
        # ----------------------------------------------------

        if (
            val_metrics["dice"]
            > best_val_dice
        ):

            best_val_dice = (
                val_metrics["dice"]
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

                    "patch_size":
                        256,

                    "target":
                        "SEM glass-fibre mask",

                    "roi":
                        "SEM bundle mask"
                },
                MODEL_PATH
            )

            print(
                "✓ Best model saved."
            )

        # Save history after every epoch
        pd.DataFrame(
            history
        ).to_csv(
            HISTORY_PATH,
            index=False
        )

    # ========================================================
    # FINAL TRAINING SUMMARY
    # ========================================================

    total_time = (
        time.time()
        - training_start
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "U-NET TRAINING COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nBest epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation Dice: "
        f"{best_val_dice:.4f}"
    )

    print(
        f"\nTotal training time: "
        f"{total_time / 60:.2f} minutes"
    )

    print(
        "\nBest model saved to:"
    )

    print(
        MODEL_PATH
    )

    print(
        "\nTraining history saved to:"
    )

    print(
        HISTORY_PATH
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The test split was not used."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()