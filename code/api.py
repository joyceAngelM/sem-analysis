from pathlib import Path
from io import BytesIO
import base64

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "final_70_15_15"
    / "unet_32epochs"
    / "best_unet_32epochs.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

THRESHOLD = 0.5
PATCH_SIZE = 256


# ============================================================
# U-NET ARCHITECTURE
# Same architecture used during training/testing
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
# LOAD MODEL ONCE WHEN API STARTS
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"U-Net checkpoint not found: {MODEL_PATH}"
    )


print("=" * 70)
print("MICROVISION AI BACKEND")
print("=" * 70)

print(f"Device: {DEVICE}")

print(f"Loading model from:\n{MODEL_PATH}")


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


model = UNet(
    in_channels=3,
    out_channels=1
).to(DEVICE)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model.eval()


print(
    f"U-Net loaded successfully "
    f"(checkpoint epoch {checkpoint['epoch']})"
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="MicroVision AI API",
    description=(
        "U-Net glass-fibre segmentation backend "
        "for SEM microscopy images."
    ),
    version="1.0.0"
)


# Allow the local Next.js frontend to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# IMAGE HELPERS
# ============================================================

to_tensor = transforms.ToTensor()


def mask_to_base64(mask):

    mask_image = Image.fromarray(
        (mask.astype(np.uint8) * 255),
        mode="L"
    )

    buffer = BytesIO()

    mask_image.save(
        buffer,
        format="PNG"
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return encoded


def run_unet_on_image(image):

    """
    Run U-Net over an arbitrary-size SEM image using
    non-overlapping 256x256 patches.

    Edge patches are padded to 256x256 and cropped back
    to their original size after prediction.
    """

    image = image.convert("RGB")

    image_np = np.array(image)

    height, width = image_np.shape[:2]

    full_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    probability_sum = 0.0
    number_of_pixels = 0

    total_patches = 0

    with torch.no_grad():

        for y in range(
            0,
            height,
            PATCH_SIZE
        ):

            for x in range(
                0,
                width,
                PATCH_SIZE
            ):

                patch = image_np[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ]

                patch_height = patch.shape[0]
                patch_width = patch.shape[1]

                padded = np.zeros(
                    (
                        PATCH_SIZE,
                        PATCH_SIZE,
                        3
                    ),
                    dtype=np.uint8
                )

                padded[
                    :patch_height,
                    :patch_width
                ] = patch

                tensor = to_tensor(
                    padded
                ).unsqueeze(0).to(DEVICE)

                logits = model(tensor)

                probabilities = torch.sigmoid(
                    logits
                )[0, 0].cpu().numpy()

                probabilities = probabilities[
                    :patch_height,
                    :patch_width
                ]

                prediction = (
                    probabilities >= THRESHOLD
                ).astype(np.uint8)

                full_mask[
                    y:y + patch_height,
                    x:x + patch_width
                ] = prediction

                probability_sum += float(
                    probabilities.sum()
                )

                number_of_pixels += (
                    patch_height
                    * patch_width
                )

                total_patches += 1


    mean_probability = (
        probability_sum
        / max(number_of_pixels, 1)
    )


    gf_percentage = (
        full_mask.sum()
        / max(full_mask.size, 1)
        * 100.0
    )


    return (
        full_mask,
        gf_percentage,
        mean_probability,
        total_patches
    )


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "MicroVision AI",
        "model": "U-Net",
        "task": "glass-fibre segmentation",
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "device": str(DEVICE)
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        )
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    try:

        contents = await file.read()

        if not contents:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )


        try:

            image = Image.open(
                BytesIO(contents)
            )

            image.load()

        except Exception:

            raise HTTPException(
                status_code=400,
                detail="Unable to read the uploaded image."
            )


        (
            prediction_mask,
            gf_percentage,
            mean_probability,
            total_patches
        ) = run_unet_on_image(image)


        mask_base64 = mask_to_base64(
            prediction_mask
        )


        return {

            "status": "completed",

            "model": "U-Net",

            "segmentation_target":
                "glass_fibre",

            "checkpoint_epoch":
                int(checkpoint["epoch"]),

            "original_width":
                int(image.width),

            "original_height":
                int(image.height),

            "patch_size":
                PATCH_SIZE,

            "patches_processed":
                int(total_patches),

            "gf_percentage":
                round(
                    float(gf_percentage),
                    4
                ),

            "mean_gf_probability":
                round(
                    float(mean_probability),
                    6
                ),

            "mask_image":
                (
                    "data:image/png;base64,"
                    + mask_base64
                )
        }


    except HTTPException:

        raise


    except Exception as error:

        print(
            "Prediction error:",
            repr(error)
        )

        raise HTTPException(
            status_code=500,
            detail="Model inference failed."
        )