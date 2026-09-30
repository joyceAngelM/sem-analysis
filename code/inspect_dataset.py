from pathlib import Path
from collections import Counter
from PIL import Image
import numpy as np

# ============================================================
# RQ2GEI DATASET INSPECTION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "Multi_Modal_Microscopy_DataSet_paper"
    / "Multi_Modal_Microscopy_DataSet_paper"
)

BUNDLE_ROOT = DATASET_ROOT / "Bundles_0_degree_SEM_FLM"

CR_FOLDERS = {
    "CR_0": {
        "mask_folder": "0deg_Binary_Masks_Quantitative",
        "image_folder": "0deg_Bundles_Stack",
    },
    "CR_30": {
        "mask_folder": "0deg_Binary_Masks_Quantitative",
        "image_folder": "0deg_Bundles_Stack",
    },
    "CR_41": {
        "mask_folder": "Binary_Masks",
        "image_folder": "Bundles_Stack",
    },
}


def separator(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def image_info(path):
    try:
        with Image.open(path) as img:
            arr = np.array(img)

            return {
                "size": img.size,       # width, height
                "mode": img.mode,
                "format": img.format,
                "shape": arr.shape,
                "dtype": str(arr.dtype),
                "min": arr.min(),
                "max": arr.max(),
            }

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# 1. VERIFY DATASET
# ============================================================

separator("1. DATASET LOCATION")

print("Project root:")
print(PROJECT_ROOT)

print("\nDataset root:")
print(DATASET_ROOT)

if not DATASET_ROOT.exists():
    print("\nERROR: Dataset root was not found.")
    print("Check the folder structure.")
    raise SystemExit

print("\nDataset root found successfully.")


if not BUNDLE_ROOT.exists():
    print("\nERROR: Bundle dataset folder was not found.")
    raise SystemExit

print("Bundle dataset folder found successfully.")


# ============================================================
# 2. INSPECT EACH COMPACTION RATIO
# ============================================================

separator("2. BUNDLE DATASET SUMMARY")

grand_total_images = 0
grand_total_mask_sample_folders = 0

all_dimensions = Counter()
all_modes = Counter()
all_dtypes = Counter()

problems = []

for cr_name, config in CR_FOLDERS.items():

    print(f"\n--- {cr_name} ---")

    cr_root = BUNDLE_ROOT / cr_name

    image_dir = cr_root / config["image_folder"]
    mask_dir = cr_root / config["mask_folder"]

    print("Image directory:")
    print(image_dir)

    print("Mask directory:")
    print(mask_dir)

    if not image_dir.exists():
        print("ERROR: Image directory missing.")
        problems.append(f"{cr_name}: image directory missing")
        continue

    if not mask_dir.exists():
        print("ERROR: Mask directory missing.")
        problems.append(f"{cr_name}: mask directory missing")
        continue

    # --------------------------------------------------------
    # Original bundle images
    # --------------------------------------------------------

    images = sorted(image_dir.glob("*.tif"))

    # Each mask sample is stored in its own directory
    mask_sample_dirs = sorted(
        p for p in mask_dir.iterdir()
        if p.is_dir()
    )

    print(f"\nOriginal bundle images : {len(images)}")
    print(f"Mask sample folders    : {len(mask_sample_dirs)}")

    grand_total_images += len(images)
    grand_total_mask_sample_folders += len(mask_sample_dirs)

    # --------------------------------------------------------
    # Check sample IDs
    # --------------------------------------------------------

    image_ids = {p.stem for p in images}
    mask_ids = {p.name for p in mask_sample_dirs}

    missing_masks = sorted(image_ids - mask_ids)
    missing_images = sorted(mask_ids - image_ids)

    print(f"Images without matching mask folder : {len(missing_masks)}")
    print(f"Mask folders without matching image : {len(missing_images)}")

    if missing_masks:
        print("\nImages without mask folder:")
        for x in missing_masks:
            print("  ", x)

    if missing_images:
        print("\nMask folders without image:")
        for x in missing_images:
            print("  ", x)

    # --------------------------------------------------------
    # Inspect original bundle images
    # --------------------------------------------------------

    dimension_counts = Counter()
    mode_counts = Counter()
    dtype_counts = Counter()

    for image_path in images:

        info = image_info(image_path)

        if "error" in info:
            problems.append(
                f"Cannot read {image_path}: {info['error']}"
            )
            continue

        dimension_counts[info["size"]] += 1
        mode_counts[info["mode"]] += 1
        dtype_counts[info["dtype"]] += 1

        all_dimensions[info["size"]] += 1
        all_modes[info["mode"]] += 1
        all_dtypes[info["dtype"]] += 1

    print("\nBundle image dimensions:")

    for dim, count in dimension_counts.items():
        print(f"  {dim[0]} x {dim[1]} : {count}")

    print("\nBundle image modes:")

    for mode, count in mode_counts.items():
        print(f"  {mode}: {count}")

    print("\nBundle image data types:")

    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count}")

    # --------------------------------------------------------
    # Inspect mask contents
    # --------------------------------------------------------

    mask_type_counts = Counter()
    mask_dimension_counts = Counter()
    mask_mode_counts = Counter()
    mask_dtype_counts = Counter()

    print("\nMask files per sample:")

    for sample_dir in mask_sample_dirs:

        mask_files = sorted(sample_dir.glob("*.tif"))

        print(
            f"  {sample_dir.name}: "
            f"{len(mask_files)} files -> "
            f"{[p.name for p in mask_files]}"
        )

        for mask_path in mask_files:

            mask_type_counts[mask_path.name] += 1

            info = image_info(mask_path)

            if "error" in info:
                problems.append(
                    f"Cannot read {mask_path}: {info['error']}"
                )
                continue

            mask_dimension_counts[info["size"]] += 1
            mask_mode_counts[info["mode"]] += 1
            mask_dtype_counts[info["dtype"]] += 1

    print("\nMask filename counts:")

    for name, count in mask_type_counts.most_common():
        print(f"  {name}: {count}")

    print("\nMask dimensions:")

    for dim, count in mask_dimension_counts.items():
        print(f"  {dim[0]} x {dim[1]} : {count}")

    print("\nMask modes:")

    for mode, count in mask_mode_counts.items():
        print(f"  {mode}: {count}")

    print("\nMask data types:")

    for dtype, count in mask_dtype_counts.items():
        print(f"  {dtype}: {count}")


# ============================================================
# 3. TOTAL COUNTS
# ============================================================

separator("3. TOTAL ORIGINAL BUNDLE DATA")

print(f"Total original bundle images : {grand_total_images}")
print(
    f"Total mask sample folders    : "
    f"{grand_total_mask_sample_folders}"
)

print("\nAll original image dimensions:")

for dim, count in all_dimensions.most_common():
    print(f"  {dim[0]} x {dim[1]} : {count}")

print("\nAll original image modes:")

for mode, count in all_modes.items():
    print(f"  {mode}: {count}")

print("\nAll original image data types:")

for dtype, count in all_dtypes.items():
    print(f"  {dtype}: {count}")


# ============================================================
# 4. CHECK MASK VALUES FOR ONE SAMPLE FROM EACH CR
# ============================================================

separator("4. EXAMPLE MASK PIXEL VALUES")

for cr_name, config in CR_FOLDERS.items():

    mask_dir = (
        BUNDLE_ROOT
        / cr_name
        / config["mask_folder"]
    )

    sample_dirs = sorted(
        p for p in mask_dir.iterdir()
        if p.is_dir()
    )

    if not sample_dirs:
        continue

    sample = sample_dirs[0]

    print(f"\n{cr_name} example: {sample.name}")

    for mask_path in sorted(sample.glob("*.tif")):

        try:
            with Image.open(mask_path) as img:

                arr = np.array(img)

                unique_values = np.unique(arr)

                print(f"\n  {mask_path.name}")
                print(f"    Shape: {arr.shape}")
                print(f"    Dtype: {arr.dtype}")
                print(f"    Min: {arr.min()}")
                print(f"    Max: {arr.max()}")

                # Avoid printing thousands of values
                if len(unique_values) <= 20:
                    print(
                        f"    Unique values: "
                        f"{unique_values.tolist()}"
                    )
                else:
                    print(
                        f"    Number of unique values: "
                        f"{len(unique_values)}"
                    )
                    print(
                        f"    First 20 values: "
                        f"{unique_values[:20].tolist()}"
                    )

        except Exception as e:
            print("    ERROR:", e)


# ============================================================
# 5. PROBLEM REPORT
# ============================================================

separator("5. PROBLEM REPORT")

if not problems:
    print("No obvious file-reading or folder problems detected.")
else:
    print(f"Problems detected: {len(problems)}")

    for problem in problems:
        print(" -", problem)


separator("INSPECTION COMPLETE")

print("""
No files were modified.
No train/validation/test split was created.
No patches were generated.
No model was trained.
""")