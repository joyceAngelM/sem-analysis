from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"

SAMPLE_ID = "CR0_L1M1"


# ============================================================
# FIND FILES
# ============================================================

print("=" * 80)
print("INSPECTING SEM / GF / BUNDLE / VOID MASK ALIGNMENT")
print("=" * 80)

print(f"\nSample: {SAMPLE_ID}")

stack_files = list(
    DATASET_ROOT.rglob(f"0deg_Bundles_Stack/{SAMPLE_ID}.tif")
)

void_files = list(
    DATASET_ROOT.rglob(f"{SAMPLE_ID}/FLM_Void_NoGF.tif")
)

bundle_files = list(
    DATASET_ROOT.rglob(f"{SAMPLE_ID}/SEM_Bundle_Mask.tif")
)

gf_files = list(
    DATASET_ROOT.rglob(f"{SAMPLE_ID}/SEM_GF_Green_Large.tif")
)

# Fallback in case another dataset version uses SEM_GF_Mask.tif
if not gf_files:
    gf_files = list(
        DATASET_ROOT.rglob(f"{SAMPLE_ID}/SEM_GF_Mask.tif")
    )


# ============================================================
# VERIFY FILES
# ============================================================

if not stack_files:
    raise FileNotFoundError(
        f"Could not find stack TIFF for {SAMPLE_ID}"
    )

if not void_files:
    raise FileNotFoundError(
        f"Could not find FLM_Void_NoGF.tif for {SAMPLE_ID}"
    )

if not bundle_files:
    raise FileNotFoundError(
        f"Could not find SEM_Bundle_Mask.tif for {SAMPLE_ID}"
    )

if not gf_files:
    raise FileNotFoundError(
        f"Could not find GF mask for {SAMPLE_ID}"
    )


stack_path = stack_files[0]
void_path = void_files[0]
bundle_path = bundle_files[0]
gf_path = gf_files[0]


print("\nFiles found:")
print(f"\nStack:\n{stack_path}")
print(f"\nBundle mask:\n{bundle_path}")
print(f"\nGF mask:\n{gf_path}")
print(f"\nVoid/NoGF mask:\n{void_path}")


# ============================================================
# LOAD STACK
# ============================================================

stack_image = Image.open(stack_path)

number_of_frames = getattr(
    stack_image,
    "n_frames",
    1
)

print("\n" + "=" * 80)
print("STACK INFORMATION")
print("=" * 80)

print(f"\nNumber of frames: {number_of_frames}")
print(f"Image size: {stack_image.size}")
print(f"Mode: {stack_image.mode}")


frames = []

for frame_index in range(number_of_frames):

    stack_image.seek(frame_index)

    frame = stack_image.convert("RGB")

    frame_array = np.array(frame)

    frames.append(frame_array)

    print(
        f"Frame {frame_index}: "
        f"shape={frame_array.shape}, "
        f"min={frame_array.min()}, "
        f"max={frame_array.max()}, "
        f"mean={frame_array.mean():.2f}"
    )


# ============================================================
# LOAD MASKS
# ============================================================

bundle_mask = np.array(
    Image.open(bundle_path).convert("L")
)

gf_mask = np.array(
    Image.open(gf_path).convert("L")
)

void_mask = np.array(
    Image.open(void_path).convert("L")
)


print("\n" + "=" * 80)
print("MASK INFORMATION")
print("=" * 80)

print(
    f"\nBundle mask: "
    f"shape={bundle_mask.shape}, "
    f"min={bundle_mask.min()}, "
    f"max={bundle_mask.max()}, "
    f"mean={bundle_mask.mean():.2f}"
)

print(
    f"GF mask: "
    f"shape={gf_mask.shape}, "
    f"min={gf_mask.min()}, "
    f"max={gf_mask.max()}, "
    f"mean={gf_mask.mean():.2f}"
)

print(
    f"Void/NoGF mask: "
    f"shape={void_mask.shape}, "
    f"min={void_mask.min()}, "
    f"max={void_mask.max()}, "
    f"mean={void_mask.mean():.2f}"
)


# ============================================================
# BINARIZE MASKS
# ============================================================

bundle_binary = bundle_mask > 0
gf_binary = gf_mask > 0
void_binary = void_mask > 0


# ============================================================
# BASIC PIXEL STATISTICS
# ============================================================

bundle_pixels = int(bundle_binary.sum())
gf_pixels = int(gf_binary.sum())
void_pixels = int(void_binary.sum())


print("\n" + "=" * 80)
print("PIXEL STATISTICS")
print("=" * 80)

print(f"\nBundle-positive pixels : {bundle_pixels:,}")
print(f"GF-positive pixels     : {gf_pixels:,}")
print(f"Void/NoGF pixels       : {void_pixels:,}")


if bundle_pixels > 0:

    gf_inside_bundle = (
        gf_binary & bundle_binary
    ).sum()

    void_inside_bundle = (
        void_binary & bundle_binary
    ).sum()

    gf_percentage = (
        gf_inside_bundle /
        bundle_pixels
    ) * 100

    void_percentage = (
        void_inside_bundle /
        bundle_pixels
    ) * 100

    print(
        f"\nGF inside bundle       : "
        f"{gf_inside_bundle:,} "
        f"({gf_percentage:.2f}%)"
    )

    print(
        f"Void/NoGF inside bundle: "
        f"{void_inside_bundle:,} "
        f"({void_percentage:.2f}%)"
    )


# ============================================================
# OVERLAP CHECK
# ============================================================

gf_void_overlap = (
    gf_binary &
    void_binary &
    bundle_binary
).sum()

print("\n" + "=" * 80)
print("GF / VOID OVERLAP CHECK")
print("=" * 80)

print(
    f"\nPixels labelled as BOTH GF and Void/NoGF "
    f"inside bundle: {gf_void_overlap:,}"
)

if gf_void_overlap == 0:
    print("No GF/Void overlap detected.")
else:
    print(
        "Some overlap exists between GF and Void/NoGF masks."
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT /
    "outputs" /
    "final_70_15_15" /
    "void_mask_inspection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output_path = (
    OUTPUT_DIR /
    f"{SAMPLE_ID}_stack_and_masks.png"
)


# ============================================================
# VISUALIZATION
# ============================================================

number_of_panels = len(frames) + 3

fig, axes = plt.subplots(
    1,
    number_of_panels,
    figsize=(4 * number_of_panels, 5)
)

# Ensure axes is iterable
axes = np.atleast_1d(axes)


for i, frame in enumerate(frames):

    axes[i].imshow(frame)

    axes[i].set_title(
        f"Stack Frame {i}"
    )

    axes[i].axis("off")


mask_start = len(frames)


axes[mask_start].imshow(
    bundle_binary,
    cmap="gray"
)

axes[mask_start].set_title(
    "SEM Bundle Mask"
)

axes[mask_start].axis("off")


axes[mask_start + 1].imshow(
    gf_binary,
    cmap="gray"
)

axes[mask_start + 1].set_title(
    "Glass Fibre Mask"
)

axes[mask_start + 1].axis("off")


axes[mask_start + 2].imshow(
    void_binary,
    cmap="gray"
)

axes[mask_start + 2].set_title(
    "FLM Void / NoGF Mask"
)

axes[mask_start + 2].axis("off")


fig.suptitle(
    f"{SAMPLE_ID} - Multimodal Stack and Quantitative Masks",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)

print(
    f"\nVisualization saved to:\n{output_path}"
)

print("""
IMPORTANT:
No dataset files were modified.
No training was performed.
No model checkpoint was changed.
This script only inspected one original specimen.
""")