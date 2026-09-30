from pathlib import Path
from PIL import Image
import numpy as np
import csv
from collections import Counter, defaultdict


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "Multi_Modal_Microscopy_DataSet_paper"
    / "Multi_Modal_Microscopy_DataSet_paper"
)

BUNDLE_ROOT = DATASET_ROOT / "Bundles_0_degree_SEM_FLM"

SPLIT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_split"
    / "sample_split.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "patch_dataset"
)

PATCH_SIZE = 224
STRIDE = 224

# At least 50% of patch must belong to bundle mask
MIN_BUNDLE_FRACTION = 0.50


# ============================================================
# DATASET FOLDER STRUCTURE
# ============================================================

CR_INFO = {
    "CR_0": {
        "image_folder": "0deg_Bundles_Stack",
        "mask_folder": "0deg_Binary_Masks_Quantitative",
    },

    "CR_30": {
        "image_folder": "0deg_Bundles_Stack",
        "mask_folder": "0deg_Binary_Masks_Quantitative",
    },

    "CR_41": {
        "image_folder": "Bundles_Stack",
        "mask_folder": "Binary_Masks",
    },
}


# ============================================================
# READ FIXED SAMPLE SPLIT
# ============================================================

samples = []

with open(SPLIT_CSV, "r", encoding="utf-8") as file:

    reader = csv.DictReader(file)

    for row in reader:

        samples.append({
            "sample_id": row["sample_id"],
            "cr": row["compaction_ratio"],
            "layer": int(row["layer"]),
            "split": row["split"],
        })


print("=" * 72)
print("PATCH GENERATION")
print("=" * 72)

print(f"\nOriginal samples loaded: {len(samples)}")
print(f"Patch size: {PATCH_SIZE} x {PATCH_SIZE}")
print(f"Stride: {STRIDE}")
print(
    f"Minimum bundle coverage: "
    f"{MIN_BUNDLE_FRACTION * 100:.0f}%"
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "validation", "test"]:

    for cr in ["CR_0", "CR_30", "CR_41"]:

        folder = OUTPUT_ROOT / split / cr

        folder.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# FIND ORIGINAL IMAGE
# ============================================================

def find_original_image(folder, sample_id):

    possible_extensions = [
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg",
    ]

    for extension in possible_extensions:

        candidate = folder / f"{sample_id}{extension}"

        if candidate.exists():
            return candidate

    # Fallback in case filenames contain extra text
    for path in folder.iterdir():

        if (
            path.is_file()
            and path.stem.startswith(sample_id)
        ):
            return path

    return None


# ============================================================
# PATCH STATISTICS
# ============================================================

patch_counts = Counter()

sample_patch_counts = defaultdict(int)

manifest_rows = []


# ============================================================
# PROCESS EACH ORIGINAL SAMPLE
# ============================================================

for sample in samples:

    sample_id = sample["sample_id"]
    cr = sample["cr"]
    split = sample["split"]

    info = CR_INFO[cr]

    image_folder = (
        BUNDLE_ROOT
        / cr
        / info["image_folder"]
    )

    mask_sample_folder = (
        BUNDLE_ROOT
        / cr
        / info["mask_folder"]
        / sample_id
    )

    image_path = find_original_image(
        image_folder,
        sample_id
    )

    bundle_mask_path = (
        mask_sample_folder
        / "SEM_Bundle_Mask.tif"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if image_path is None:

        print(
            f"WARNING: Original image not found "
            f"for {sample_id}"
        )

        continue

    if not bundle_mask_path.exists():

        print(
            f"WARNING: Bundle mask not found "
            f"for {sample_id}"
        )

        continue

    # --------------------------------------------------------
    # Load original image and bundle mask
    # --------------------------------------------------------

    with Image.open(image_path) as img:
        image = np.array(img.convert("RGB"))

    with Image.open(bundle_mask_path) as img:
        bundle_mask = np.array(
            img.convert("L")
        ) > 0

    # --------------------------------------------------------
    # Verify dimensions
    # --------------------------------------------------------

    if image.shape[:2] != bundle_mask.shape:

        print(
            f"WARNING: Dimension mismatch for "
            f"{sample_id}"
        )

        print(
            f"  Image: {image.shape[:2]}"
        )

        print(
            f"  Mask : {bundle_mask.shape}"
        )

        continue

    height, width = bundle_mask.shape

    accepted = 0

    # --------------------------------------------------------
    # Generate non-overlapping 224 x 224 patches
    # --------------------------------------------------------

    for y in range(
        0,
        height - PATCH_SIZE + 1,
        STRIDE
    ):

        for x in range(
            0,
            width - PATCH_SIZE + 1,
            STRIDE
        ):

            mask_patch = bundle_mask[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            bundle_fraction = (
                np.count_nonzero(mask_patch)
                / mask_patch.size
            )

            # Reject mostly-background patches
            if bundle_fraction < MIN_BUNDLE_FRACTION:
                continue

            image_patch = image[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            # ------------------------------------------------
            # Save patch
            # ------------------------------------------------

            patch_name = (
                f"{sample_id}"
                f"_y{y}"
                f"_x{x}.png"
            )

            output_path = (
                OUTPUT_ROOT
                / split
                / cr
                / patch_name
            )

            Image.fromarray(
                image_patch
            ).save(output_path)

            accepted += 1

            patch_counts[(split, cr)] += 1

            sample_patch_counts[
                sample_id
            ] += 1

            manifest_rows.append({
                "patch_name": patch_name,
                "sample_id": sample_id,
                "compaction_ratio": cr,
                "layer": sample["layer"],
                "split": split,
                "x": x,
                "y": y,
                "bundle_fraction": bundle_fraction,
            })

    print(
        f"{sample_id:<12} | "
        f"{split:<10} | "
        f"{cr:<5} | "
        f"patches={accepted}"
    )


# ============================================================
# SAVE PATCH MANIFEST
# ============================================================

manifest_path = (
    OUTPUT_ROOT
    / "patch_manifest.csv"
)

with open(
    manifest_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "patch_name",
        "sample_id",
        "compaction_ratio",
        "layer",
        "split",
        "x",
        "y",
        "bundle_fraction",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(manifest_rows)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 72)
print("PATCH COUNTS BY SPLIT AND CLASS")
print("=" * 72)

for split in [
    "train",
    "validation",
    "test"
]:

    print(f"\n{split.upper()}")

    split_total = 0

    for cr in [
        "CR_0",
        "CR_30",
        "CR_41"
    ]:

        count = patch_counts[
            (split, cr)
        ]

        split_total += count

        print(
            f"  {cr}: {count} patches"
        )

    print(
        f"  TOTAL: {split_total} patches"
    )


# ============================================================
# CHECK FOR ZERO-PATCH ORIGINALS
# ============================================================

print("\n" + "=" * 72)
print("ORIGINAL SAMPLE CHECK")
print("=" * 72)

zero_patch_samples = []

for sample in samples:

    sample_id = sample["sample_id"]

    count = sample_patch_counts[
        sample_id
    ]

    if count == 0:
        zero_patch_samples.append(
            sample_id
        )


if zero_patch_samples:

    print(
        "\nWARNING: These original samples "
        "generated zero patches:"
    )

    for sample_id in zero_patch_samples:
        print(f"  {sample_id}")

else:

    print(
        "\nAll 48 original samples "
        "generated at least one patch."
    )


# ============================================================
# LEAKAGE CHECK
# ============================================================

sample_to_splits = defaultdict(set)

for row in manifest_rows:

    sample_to_splits[
        row["sample_id"]
    ].add(
        row["split"]
    )


leakage_samples = []

for sample_id, splits in sample_to_splits.items():

    if len(splits) > 1:

        leakage_samples.append(
            (sample_id, splits)
        )


print("\n" + "=" * 72)
print("DATA LEAKAGE CHECK")
print("=" * 72)


if leakage_samples:

    print(
        "\nERROR: DATA LEAKAGE DETECTED!"
    )

    for sample_id, splits in leakage_samples:

        print(
            sample_id,
            splits
        )

else:

    print(
        "\nPASS: No original sample appears "
        "in more than one split."
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 72)
print("PATCH GENERATION COMPLETE")
print("=" * 72)

print(
    f"\nTotal accepted patches: "
    f"{len(manifest_rows)}"
)

print(
    f"\nPatch dataset saved to:\n"
    f"{OUTPUT_ROOT}"
)

print(
    f"\nManifest saved to:\n"
    f"{manifest_path}"
)

print("""
IMPORTANT:
Do not move patches between train, validation, and test.

The test patches must remain untouched until final model
evaluation.
""")