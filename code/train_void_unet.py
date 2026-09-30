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


# 7000 PRE-GENERATED AUGMENTED TRAINING PAIRS
TRAIN_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "final_70_15_15"
    / "void_unet_dataset_augmented"
    / "augmented_void_train_manifest.csv"
)


# ORIGINAL DATASET MANIFEST
# Validation and test come ONLY from this untouched manifest.
ORIGINAL_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "final_70_15_15"
    / "void_unet_dataset_256"
    / "void_unet_patch_manifest.csv"
)


# MODEL OUTPUT DIRECTORY
OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "final_70_15_15"
    / "void_unet_8epochs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_PATH = (
    OUTPUT_DIR
    / "best_void_unet.pth"
)


HISTORY_PATH = (
    OUTPUT_DIR
    / "void_training_history.csv"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 16

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

class VoidSegmentationDataset(Dataset):

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


    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[index]


        # ====================================================
        # LOAD SEM IMAGE
        # ====================================================

        with Image.open(
            row["image_patch"]
        ) as image_file:

            image = np.array(
                image_file.convert("RGB")
            )


        # ====================================================
        # LOAD VOID / NoGF TARGET MASK
        # ====================================================

        with Image.open(
            row["void_mask_patch"]
        ) as mask_file:

            mask = np.array(
                mask_file.convert("L")
            )


        # ====================================================
        # LOAD BUNDLE ROI MASK
        # ====================================================

        with Image.open(
            row["bundle_mask_patch"]
        ) as roi_file:

            roi = np.array(
                roi_file.convert("L")
            )


        # ====================================================
        # VERIFY SHAPES
        # ====================================================

        if image.shape[:2] != (256, 256):

            raise ValueError(
                f"Incorrect SEM size: "
                f"{image.shape}"
            )


        if mask.shape != (256, 256):

            raise ValueError(
                f"Incorrect Void mask size: "
                f"{mask.shape}"
            )


        if roi.shape != (256, 256):

            raise ValueError(
                f"Incorrect ROI size: "
                f"{roi.shape}"
            )


        # ====================================================
        # IMPORTANT
        #
        # NO ONLINE AUGMENTATION.
        #
        # 7000 training pairs were already generated and saved.
        #
        # Validation and test remain untouched.
        # ====================================================


        # SEM -> tensor
        image = self.to_tensor(
            image
        )


        # Void mask -> binary tensor
        mask = torch.from_numpy(
            (mask > 0)
            .astype(np.float32)
        ).unsqueeze(0)


        # Bundle ROI -> binary tensor
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
# U-NET DOUBLE CONVOLUTION BLOCK
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


    def forward(
        self,
        x
    ):

        return self.block(x)


# ============================================================
# U-NET MODEL
# ============================================================

class UNet(nn.Module):

    def __init__(self):

        super().__init__()


        # ====================================================
        # ENCODER
        # ====================================================

        self.enc1 = DoubleConv(
            3,
            64
        )

        self.pool1 = nn.MaxPool2d(
            2
        )


        self.enc2 = DoubleConv(
            64,
            128
        )

        self.pool2 = nn.MaxPool2d(
            2
        )


        self.enc3 = DoubleConv(
            128,
            256
        )

        self.pool3 = nn.MaxPool2d(
            2
        )


        self.enc4 = DoubleConv(
            256,
            512
        )

        self.pool4 = nn.MaxPool2d(
            2
        )


        # ====================================================
        # BOTTLENECK
        # ====================================================

        self.bottleneck = DoubleConv(
            512,
            1024
        )


        # ====================================================
        # DECODER
        # ====================================================

        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(
            1024,
            512
        )


        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            512,
            256
        )


        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            256,
            128
        )


        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            128,
            64
        )


        # ====================================================
        # OUTPUT
        # ====================================================

        self.output = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )


    def forward(
        self,
        x
    ):

        # ====================================================
        # ENCODER
        # ====================================================

        e1 = self.enc1(
            x
        )


        e2 = self.enc2(
            self.pool1(
                e1
            )
        )


        e3 = self.enc3(
            self.pool2(
                e2
            )
        )


        e4 = self.enc4(
            self.pool3(
                e3
            )
        )


        # ====================================================
        # BOTTLENECK
        # ====================================================

        b = self.bottleneck(
            self.pool4(
                e4
            )
        )


        # ====================================================
        # DECODER LEVEL 4
        # ====================================================

        d4 = self.up4(
            b
        )

        d4 = torch.cat(
            [
                d4,
                e4
            ],
            dim=1
        )

        d4 = self.dec4(
            d4
        )


        # ====================================================
        # DECODER LEVEL 3
        # ====================================================

        d3 = self.up3(
            d4
        )

        d3 = torch.cat(
            [
                d3,
                e3
            ],
            dim=1
        )

        d3 = self.dec3(
            d3
        )


        # ====================================================
        # DECODER LEVEL 2
        # ====================================================

        d2 = self.up2(
            d3
        )

        d2 = torch.cat(
            [
                d2,
                e2
            ],
            dim=1
        )

        d2 = self.dec2(
            d2
        )


        # ====================================================
        # DECODER LEVEL 1
        # ====================================================

        d1 = self.up1(
            d2
        )

        d1 = torch.cat(
            [
                d1,
                e1
            ],
            dim=1
        )

        d1 = self.dec1(
            d1
        )


        # RAW LOGITS
        return self.output(
            d1
        )


# ============================================================
# ROI-AWARE BCE + DICE LOSS
# ============================================================

class BCEDiceLoss(nn.Module):

    def __init__(self):

        super().__init__()


        self.bce = nn.BCEWithLogitsLoss(
            reduction="none"
        )


    def forward(
        self,
        logits,
        targets,
        roi
    ):

        # ====================================================
        # BCE ONLY INSIDE BUNDLE ROI
        # ====================================================

        bce_map = self.bce(
            logits,
            targets
        )


        roi_pixels = (
            roi.sum()
            .clamp_min(1.0)
        )


        bce_loss = (
            (bce_map * roi).sum()
            / roi_pixels
        )


        # ====================================================
        # DICE ONLY INSIDE BUNDLE ROI
        # ====================================================

        probabilities = torch.sigmoid(
            logits
        )


        probabilities = (
            probabilities
            * roi
        )


        targets_roi = (
            targets
            * roi
        )


        intersection = (
            probabilities
            * targets_roi
        ).sum()


        denominator = (
            probabilities.sum()
            + targets_roi.sum()
        )


        smooth = 1.0


        dice = (
            (2.0 * intersection + smooth)
            / (denominator + smooth)
        )


        dice_loss = (
            1.0
            - dice
        )


        return (
            bce_loss
            + dice_loss
        )


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    logits,
    targets,
    roi
):

    probabilities = torch.sigmoid(
        logits
    )


    predictions = (
        probabilities
        >= THRESHOLD
    )


    targets = (
        targets
        >= 0.5
    )


    roi = (
        roi
        >= 0.5
    )


    # ========================================================
    # ONLY EVALUATE INSIDE BUNDLE ROI
    # ========================================================

    predictions = (
        predictions
        & roi
    )


    targets = (
        targets
        & roi
    )


    tp = (
        predictions
        & targets
    ).sum().item()


    fp = (
        predictions
        & (~targets)
        & roi
    ).sum().item()


    fn = (
        (~predictions)
        & targets
        & roi
    ).sum().item()


    tn = (
        (~predictions)
        & (~targets)
        & roi
    ).sum().item()


    return (
        tp,
        fp,
        fn,
        tn
    )


# ============================================================
# CONVERT COUNTS TO METRICS
# ============================================================

def metrics_from_counts(
    tp,
    fp,
    fn,
    tn
):

    eps = 1e-8


    dice = (
        (2 * tp)
        /
        (
            2 * tp
            + fp
            + fn
            + eps
        )
    )


    iou = (
        tp
        /
        (
            tp
            + fp
            + fn
            + eps
        )
    )


    precision = (
        tp
        /
        (
            tp
            + fp
            + eps
        )
    )


    recall = (
        tp
        /
        (
            tp
            + fn
            + eps
        )
    )


    accuracy = (
        (tp + tn)
        /
        (
            tp
            + tn
            + fp
            + fn
            + eps
        )
    )


    return {

        "dice":
            dice,

        "iou":
            iou,

        "precision":
            precision,

        "recall":
            recall,

        "accuracy":
            accuracy
    }


# ============================================================
# RUN ONE TRAINING / VALIDATION EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    epoch_number=None,
    phase="Training"
):

    training = (
        optimizer is not None
    )


    if training:

        model.train()

    else:

        model.eval()


    running_loss = 0.0


    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0


    total_batches = len(
        loader
    )


    print(
        f"\n{phase} started - "
        f"{total_batches} batches",
        flush=True
    )


    phase_start = time.time()


    # ========================================================
    # BATCH LOOP
    # ========================================================

    for batch_index, (
        images,
        masks,
        rois
    ) in enumerate(
        loader,
        start=1
    ):

        images = images.to(
            DEVICE
        )

        masks = masks.to(
            DEVICE
        )

        rois = rois.to(
            DEVICE
        )


        if training:

            optimizer.zero_grad()


        with torch.set_grad_enabled(
            training
        ):

            logits = model(
                images
            )


            loss = criterion(
                logits,
                masks,
                rois
            )


            if training:

                loss.backward()

                optimizer.step()


        # ====================================================
        # LOSS
        # ====================================================

        running_loss += (
            loss.item()
            * images.size(0)
        )


        # ====================================================
        # METRICS
        # ====================================================

        (
            tp,
            fp,
            fn,
            tn
        ) = calculate_metrics(
            logits.detach(),
            masks,
            rois
        )


        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn


        # ====================================================
        # PROGRESS DISPLAY
        #
        # Print:
        # batch 1
        # every 10 batches
        # final batch
        # ====================================================

        if (
            batch_index == 1
            or batch_index % 10 == 0
            or batch_index == total_batches
        ):

            elapsed = (
                time.time()
                - phase_start
            )


            average_batch_time = (
                elapsed
                / batch_index
            )


            remaining_batches = (
                total_batches
                - batch_index
            )


            estimated_remaining = (
                average_batch_time
                * remaining_batches
            )


            print(
                f"Epoch "
                f"{epoch_number}/{EPOCHS} | "
                f"{phase} batch "
                f"{batch_index}/{total_batches} | "
                f"Loss "
                f"{loss.item():.4f} | "
                f"Elapsed "
                f"{elapsed / 60:.1f} min | "
                f"ETA "
                f"{estimated_remaining / 60:.1f} min",
                flush=True
            )


    # ========================================================
    # END OF PHASE
    # ========================================================

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

    print(
        "=" * 80
    )

    print(
        "VOID / NoGF U-NET TRAINING"
    )

    print(
        "=" * 80
    )


    print(
        f"\nDevice: "
        f"{DEVICE}"
    )


    print(
        f"Epochs: "
        f"{EPOCHS}"
    )


    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )


    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )


    print(
        "\nLoss: "
        "ROI-aware BCE + Dice"
    )


    print(
        "Best checkpoint criterion: "
        "Validation Dice"
    )


    # ========================================================
    # CHECK MANIFESTS
    # ========================================================

    if not TRAIN_MANIFEST.exists():

        raise FileNotFoundError(
            f"\nAugmented training manifest "
            f"not found:\n"
            f"{TRAIN_MANIFEST}"
        )


    if not ORIGINAL_MANIFEST.exists():

        raise FileNotFoundError(
            f"\nOriginal manifest "
            f"not found:\n"
            f"{ORIGINAL_MANIFEST}"
        )


    # ========================================================
    # LOAD 7000 TRAINING PAIRS
    # ========================================================

    train_df = pd.read_csv(
        TRAIN_MANIFEST
    )


    # ========================================================
    # LOAD ORIGINAL DATASET MANIFEST
    # ========================================================

    original_manifest = pd.read_csv(
        ORIGINAL_MANIFEST
    )


    # ========================================================
    # UNTOUCHED VALIDATION
    # ========================================================

    val_df = original_manifest[
        original_manifest["split"]
        == "validation"
    ].copy()


    # ========================================================
    # UNTOUCHED TEST
    #
    # Loaded ONLY for count verification.
    # It is NOT passed into a DataLoader here.
    # ========================================================

    test_df = original_manifest[
        original_manifest["split"]
        == "test"
    ].copy()


    # ========================================================
    # VERIFY REQUIRED COLUMNS
    # ========================================================

    required_train_columns = {

        "image_patch",

        "void_mask_patch",

        "bundle_mask_patch"
    }


    missing_train_columns = (
        required_train_columns
        - set(
            train_df.columns
        )
    )


    if missing_train_columns:

        raise ValueError(
            f"\nTraining manifest "
            f"missing columns: "
            f"{missing_train_columns}"
        )


    required_original_columns = {

        "split",

        "image_patch",

        "void_mask_patch",

        "bundle_mask_patch"
    }


    missing_original_columns = (
        required_original_columns
        - set(
            original_manifest.columns
        )
    )


    if missing_original_columns:

        raise ValueError(
            f"\nOriginal manifest "
            f"missing columns: "
            f"{missing_original_columns}"
        )


    # ========================================================
    # VERIFY COUNTS
    # ========================================================

    if len(train_df) != 7000:

        raise ValueError(
            f"\nExpected 7000 training pairs, "
            f"found {len(train_df)}."
        )


    if len(val_df) != 173:

        raise ValueError(
            f"\nExpected 173 validation patches, "
            f"found {len(val_df)}."
        )


    if len(test_df) != 179:

        raise ValueError(
            f"\nExpected 179 test patches, "
            f"found {len(test_df)}."
        )


    # ========================================================
    # DISPLAY DATASET INFORMATION
    # ========================================================

    print(
        f"\nTraining pairs     : "
        f"{len(train_df)}"
    )


    print(
        f"Validation patches : "
        f"{len(val_df)}"
    )


    print(
        f"Test patches       : "
        f"{len(test_df)}"
    )


    print(
        "\nTraining source: "
        "PRE-GENERATED AUGMENTED VOID DATASET"
    )


    print(
        "Online augmentation: "
        "DISABLED"
    )


    print(
        "\nIMPORTANT: Test patches are "
        "NOT used during training."
    )


    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = (
        VoidSegmentationDataset(
            train_df
        )
    )


    val_dataset = (
        VoidSegmentationDataset(
            val_df
        )
    )


    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS
    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS
    )


    # ========================================================
    # SHOW EXPECTED BATCH COUNTS
    # ========================================================

    print(
        f"\nTraining batches per epoch   : "
        f"{len(train_loader)}"
    )


    print(
        f"Validation batches per epoch : "
        f"{len(val_loader)}"
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = UNet().to(
        DEVICE
    )


    criterion = (
        BCEDiceLoss()
    )


    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE
    )


    # ========================================================
    # TRAINING VARIABLES
    # ========================================================

    best_val_dice = -1.0


    history = []


    training_start = time.time()


    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        epoch_start = time.time()


        print(
            "\n"
            + "#" * 80
        )


        print(
            f"STARTING EPOCH "
            f"{epoch}/{EPOCHS}"
        )


        print(
            "#" * 80,
            flush=True
        )


        # ====================================================
        # TRAINING
        # ====================================================

        (
            train_loss,
            train_metrics
        ) = run_epoch(

            model,

            train_loader,

            criterion,

            optimizer=optimizer,

            epoch_number=epoch,

            phase="Training"
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        (
            val_loss,
            val_metrics
        ) = run_epoch(

            model,

            val_loader,

            criterion,

            optimizer=None,

            epoch_number=epoch,

            phase="Validation"
        )


        epoch_seconds = (
            time.time()
            - epoch_start
        )


        # ====================================================
        # EPOCH RESULTS
        # ====================================================

        print(
            "\n"
            + "=" * 80
        )


        print(
            f"EPOCH "
            f"{epoch}/{EPOCHS} "
            f"COMPLETE"
        )


        print(
            "=" * 80
        )


        print(
            f"Train Loss      : "
            f"{train_loss:.4f}"
        )


        print(
            f"Train Dice      : "
            f"{train_metrics['dice']:.4f}"
        )


        print(
            f"Train IoU       : "
            f"{train_metrics['iou']:.4f}"
        )


        print(
            f"Validation Loss : "
            f"{val_loss:.4f}"
        )


        print(
            f"Validation Dice : "
            f"{val_metrics['dice']:.4f}"
        )


        print(
            f"Validation IoU  : "
            f"{val_metrics['iou']:.4f}"
        )


        print(
            f"Precision       : "
            f"{val_metrics['precision']:.4f}"
        )


        print(
            f"Recall          : "
            f"{val_metrics['recall']:.4f}"
        )


        print(
            f"Accuracy        : "
            f"{val_metrics['accuracy']:.4f}"
        )


        print(
            f"Epoch time      : "
            f"{epoch_seconds / 60:.2f} minutes"
        )


        # ====================================================
        # HISTORY
        # ====================================================

        history.append({

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "train_dice":
                train_metrics["dice"],

            "train_iou":
                train_metrics["iou"],

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
                epoch_seconds
        })


        # ====================================================
        # SAVE HISTORY AFTER EVERY EPOCH
        #
        # This means even if training is interrupted later,
        # completed epoch history is preserved.
        # ====================================================

        history_df = pd.DataFrame(
            history
        )


        history_df.to_csv(
            HISTORY_PATH,
            index=False
        )


        # ====================================================
        # SAVE BEST CHECKPOINT
        # ====================================================

        if (
            val_metrics["dice"]
            > best_val_dice
        ):

            best_val_dice = (
                val_metrics["dice"]
            )


            torch.save(
                {

                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "best_val_dice":
                        best_val_dice,

                    "threshold":
                        THRESHOLD,

                    "target":
                        "FLM_Void_NoGF",

                    "loss":
                        "ROI-aware BCE + Dice",

                    "patch_size":
                        256,

                    "training_pairs":
                        len(train_df),

                    "validation_patches":
                        len(val_df),

                    "test_patches":
                        len(test_df),

                    "augmentation":
                        "pre-generated",

                    "online_augmentation":
                        False
                },

                MODEL_PATH
            )


            print(
                "\n*** NEW BEST VOID MODEL SAVED ***"
            )


            print(
                f"Validation Dice: "
                f"{best_val_dice:.4f}"
            )


            print(
                f"Checkpoint:\n"
                f"{MODEL_PATH}"
            )


    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    total_seconds = (
        time.time()
        - training_start
    )


    print(
        "\n"
        + "=" * 80
    )


    print(
        "VOID / NoGF U-NET TRAINING COMPLETE"
    )


    print(
        "=" * 80
    )


    print(
        f"\nBest validation Dice: "
        f"{best_val_dice:.4f}"
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
        f"\nTotal training time: "
        f"{total_seconds / 60:.2f} minutes"
    )


    print("""
IMPORTANT:
- Target = FLM_Void_NoGF.
- Training pairs = 7000 pre-generated pairs.
- Training pairs originate from 34 independent training specimens.
- Loss = ROI-aware BCE + Dice.
- Metrics are calculated only inside the bundle ROI.
- Online augmentation = DISABLED.
- Validation = 173 untouched patches.
- Test = 179 untouched patches.
- Test set was NOT evaluated during training.
- Best checkpoint was selected using validation Dice.
- Training history is saved after every completed epoch.
- Existing Glass-Fibre U-Net was NOT modified.
""")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()