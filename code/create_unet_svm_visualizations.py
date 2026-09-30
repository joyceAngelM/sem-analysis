from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

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

UNET_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "final_70_15_15"
    / "unet_32epochs"
    / "best_unet_32epochs.pth"
)

SVM_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "final_70_15_15"
    / "svm_segmentation"
)

SVM_MODEL_PATH = (
    SVM_DIR
    / "svm_gf_segmentation_model.pkl"
)

SCALER_PATH = (
    SVM_DIR
    / "svm_feature_scaler.pkl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "final_70_15_15"
    / "comparison_visualizations"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

THRESHOLD = 0.5


# ============================================================
# IMAGE FUNCTIONS
# ============================================================

def load_sem(path):

    with Image.open(path) as image:

        rgb = np.array(
            image.convert("RGB")
        )

        gray = np.array(
            image.convert("L"),
            dtype=np.float32
        ) / 255.0

    return rgb, gray


def load_mask(path):

    with Image.open(path) as image:

        mask = np.array(
            image.convert("L")
        )

    return mask > 0


# ============================================================
# SVM FEATURE EXTRACTION
# SAME FEATURES USED DURING TRAINING
# ============================================================

def extract_svm_features(gray):

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
# U-NET
# SAME ARCHITECTURE USED FOR TEST EVALUATION
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


class UNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=1
    ):

        super().__init__()

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

        self.bottleneck = DoubleConv(
            256,
            512
        )

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
# OVERLAY
# ============================================================

def create_overlay(
    rgb,
    mask
):

    overlay = (
        rgb.astype(np.float32)
        / 255.0
    ).copy()

    mask = mask.astype(bool)

    # Highlight predicted GF regions.
    overlay[mask, 0] = 1.0

    overlay[mask, 1] *= 0.35
    overlay[mask, 2] *= 0.35

    return np.clip(
        overlay,
        0,
        1
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("U-NET VS SVM SEGMENTATION VISUALIZATIONS")
    print("=" * 80)

    print(
        f"\nDevice: {DEVICE}"
    )

    # ========================================================
    # CHECK REQUIRED FILES
    # ========================================================

    required_files = [
        MANIFEST_PATH,
        UNET_MODEL_PATH,
        SVM_MODEL_PATH,
        SCALER_PATH
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{path}"
            )

    # ========================================================
    # LOAD TEST DATA
    # ========================================================

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    test_df = manifest[
        manifest["split"] == "test"
    ].copy()

    print(
        f"\nUntouched test patches: "
        f"{len(test_df)}"
    )

    print(
        f"Independent test specimens: "
        f"{test_df['sample_id'].nunique()}"
    )

    if len(test_df) != 179:

        raise ValueError(
            f"Expected 179 test patches, "
            f"found {len(test_df)}."
        )

    # ========================================================
    # LOAD U-NET
    # ========================================================

    checkpoint = torch.load(
        UNET_MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    print(
        f"\nU-Net checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    model = UNet(
        in_channels=3,
        out_channels=1
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    # ========================================================
    # LOAD SVM + SCALER
    # ========================================================

    with open(
        SVM_MODEL_PATH,
        "rb"
    ) as file:

        svm = pickle.load(file)

    with open(
        SCALER_PATH,
        "rb"
    ) as file:

        scaler = pickle.load(file)

    print(
        "SVM and scaler loaded successfully."
    )

    # ========================================================
    # SELECT ONE REPRESENTATIVE PATCH
    # FROM EACH TEST SPECIMEN
    #
    # We use the patch closest to the specimen's
    # median GF fraction. This avoids manually choosing
    # only unusually easy-looking patches.
    # ========================================================

    selected_rows = []

    for sample_id, group in (
        test_df.groupby(
            "sample_id"
        )
    ):

        group = group.copy()

        median_gf = group[
            "gf_fraction"
        ].median()

        group[
            "distance_from_median"
        ] = np.abs(
            group[
                "gf_fraction"
            ]
            - median_gf
        )

        selected = group.sort_values(
            [
                "distance_from_median",
                "patch_id"
            ]
        ).iloc[0]

        selected_rows.append(
            selected
        )

    selected_df = pd.DataFrame(
        selected_rows
    )

    print(
        f"\nRepresentative patches selected: "
        f"{len(selected_df)}"
    )

    # ========================================================
    # GENERATE VISUALIZATIONS
    # ========================================================

    for number, (_, row) in enumerate(
        selected_df.iterrows(),
        start=1
    ):

        sample_id = str(
            row["sample_id"]
        )

        patch_id = str(
            row["patch_id"]
        )

        print(
            f"\n[{number}/{len(selected_df)}] "
            f"{sample_id} -> {patch_id}"
        )

        # ----------------------------------------------------
        # LOAD IMAGE + MASKS
        # ----------------------------------------------------

        rgb, gray = load_sem(
            row["image_patch"]
        )

        gt = load_mask(
            row["gf_mask_patch"]
        )

        roi = load_mask(
            row["bundle_mask_patch"]
        )

        if rgb.shape[:2] != (256, 256):
            raise ValueError(
                f"Incorrect SEM size: "
                f"{patch_id}"
            )

        # Ground truth restricted to valid ROI
        gt = gt & roi

        # ----------------------------------------------------
        # U-NET PREDICTION
        # ----------------------------------------------------

        image_tensor = (
            torch.from_numpy(
                rgb.astype(
                    np.float32
                )
                / 255.0
            )
            .permute(
                2,
                0,
                1
            )
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():

            logits = model(
                image_tensor
            )

            probability = (
                torch.sigmoid(
                    logits
                )[0, 0]
                .cpu()
                .numpy()
            )

        unet_prediction = (
            probability
            >= THRESHOLD
        )

        unet_prediction = (
            unet_prediction
            & roi
        )

        # ----------------------------------------------------
        # SVM PREDICTION
        # ----------------------------------------------------

        svm_features = (
            extract_svm_features(
                gray
            )
        )

        flat_features = (
            svm_features.reshape(
                -1,
                6
            )
        )

        roi_flat = roi.ravel()

        valid_indices = (
            np.flatnonzero(
                roi_flat
            )
        )

        svm_flat = np.zeros(
            256 * 256,
            dtype=np.uint8
        )

        if len(valid_indices) > 0:

            valid_features = (
                flat_features[
                    valid_indices
                ]
            )

            valid_scaled = (
                scaler.transform(
                    valid_features
                )
            )

            predictions = (
                svm.predict(
                    valid_scaled
                )
            )

            svm_flat[
                valid_indices
            ] = predictions.astype(
                np.uint8
            )

        svm_prediction = (
            svm_flat.reshape(
                256,
                256
            ).astype(bool)
        )

        # ----------------------------------------------------
        # OVERLAYS
        # ----------------------------------------------------

        unet_overlay = create_overlay(
            rgb,
            unet_prediction
        )

        svm_overlay = create_overlay(
            rgb,
            svm_prediction
        )

        # ----------------------------------------------------
        # FIGURE
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(14, 9)
        )

        axes[0, 0].imshow(
            rgb
        )

        axes[0, 0].set_title(
            "Original SEM"
        )

        axes[0, 1].imshow(
            gt,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[0, 1].set_title(
            "Ground Truth GF"
        )

        axes[0, 2].imshow(
            unet_prediction,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[0, 2].set_title(
            "U-Net Prediction"
        )

        axes[1, 0].imshow(
            svm_prediction,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[1, 0].set_title(
            "SVM Prediction"
        )

        axes[1, 1].imshow(
            unet_overlay
        )

        axes[1, 1].set_title(
            "U-Net Overlay"
        )

        axes[1, 2].imshow(
            svm_overlay
        )

        axes[1, 2].set_title(
            "SVM Overlay"
        )

        for ax in axes.ravel():

            ax.axis(
                "off"
            )

        fig.suptitle(
            f"{sample_id} | {patch_id}",
            fontsize=14
        )

        plt.tight_layout()

        output_path = (
            OUTPUT_DIR
            / f"{sample_id}_comparison.png"
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(
            fig
        )

    # ========================================================
    # SAVE SELECTION INFORMATION
    # ========================================================

    selection_path = (
        OUTPUT_DIR
        / "selected_visualization_patches.csv"
    )

    selected_df[
        [
            "sample_id",
            "compaction_ratio",
            "layer",
            "patch_id",
            "x",
            "y",
            "gf_fraction"
        ]
    ].to_csv(
        selection_path,
        index=False
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)

    print(
        f"\nComparison images created: "
        f"{len(selected_df)}"
    )

    print(
        f"\nSaved in:\n"
        f"{OUTPUT_DIR}"
    )

    print(
        f"\nSelected patch record:\n"
        f"{selection_path}"
    )

    print(
        "\nIMPORTANT:"
        "\n- Only untouched TEST patches were visualized."
        "\n- No training occurred."
        "\n- Ground truth was not used to generate predictions."
        "\n- U-Net and SVM predictions came from the saved models."
        "\n- One representative patch was selected automatically "
        "from each of the 7 test specimens."
    )


if __name__ == "__main__":
    main()