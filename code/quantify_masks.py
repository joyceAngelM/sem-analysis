from pathlib import Path
from PIL import Image
import numpy as np
import csv

# ============================================================
# RQ2GEI MASK QUANTIFICATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "Multi_Modal_Microscopy_DataSet_paper"
    / "Multi_Modal_Microscopy_DataSet_paper"
)

BUNDLE_ROOT = DATASET_ROOT / "Bundles_0_degree_SEM_FLM"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "mask_quantification.csv"

CR_FOLDERS = {
    "CR_0": "0deg_Binary_Masks_Quantitative",
    "CR_30": "0deg_Binary_Masks_Quantitative",
    "CR_41": "Binary_Masks",
}


def load_binary_mask(path):
    """Load mask and convert it to True/False."""
    with Image.open(path) as img:
        arr = np.array(img.convert("L"))

    return arr > 0


results = []


# ============================================================
# PROCESS ALL 48 SAMPLES
# ============================================================

for cr_name, mask_folder_name in CR_FOLDERS.items():

    mask_root = (
        BUNDLE_ROOT
        / cr_name
        / mask_folder_name
    )

    sample_dirs = sorted(
        p for p in mask_root.iterdir()
        if p.is_dir()
    )

    print("\n" + "=" * 70)
    print(cr_name)
    print("=" * 70)

    for sample_dir in sample_dirs:

        sample_id = sample_dir.name

        bundle_path = sample_dir / "SEM_Bundle_Mask.tif"

        gf_path = sample_dir / "SEM_GF_Mask.tif"

        # Special filename used by one CR0 sample
        if not gf_path.exists():
            alternative = (
                sample_dir / "SEM_GF_Green_Large.tif"
            )

            if alternative.exists():
                gf_path = alternative

        void_path = sample_dir / "FLM_Void_NoGF.tif"

        if not (
            bundle_path.exists()
            and gf_path.exists()
            and void_path.exists()
        ):
            print(f"Skipping {sample_id}: missing mask.")
            continue

        # ----------------------------------------------------
        # Load binary masks
        # ----------------------------------------------------

        bundle_mask = load_binary_mask(bundle_path)
        gf_mask = load_binary_mask(gf_path)
        void_mask = load_binary_mask(void_path)

        # ----------------------------------------------------
        # Basic areas in pixels
        # ----------------------------------------------------

        total_pixels = bundle_mask.size

        bundle_area = np.count_nonzero(bundle_mask)

        # Restrict measurements to bundle region
        gf_inside_bundle = gf_mask & bundle_mask
        void_inside_bundle = void_mask & bundle_mask

        gf_area = np.count_nonzero(gf_inside_bundle)
        void_area = np.count_nonzero(void_inside_bundle)

        # ----------------------------------------------------
        # Fractions relative to bundle area
        # ----------------------------------------------------

        if bundle_area > 0:
            gf_fraction = gf_area / bundle_area
            void_fraction = void_area / bundle_area
        else:
            gf_fraction = 0
            void_fraction = 0

        gf_percent = gf_fraction * 100
        void_percent = void_fraction * 100

        # ----------------------------------------------------
        # Extract layer and mesh number
        #
        # Example:
        # CR30_L3M2
        #      L3 = Layer 3
        #      M2 = bundle number 2
        # ----------------------------------------------------

        remainder = sample_id.split("_", 1)[1]

        layer_part = remainder.split("M")[0]
        mesh_part = remainder.split("M")[1]

        layer = int(layer_part.replace("L", ""))
        mesh = int(mesh_part)

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result = {
            "sample_id": sample_id,
            "compaction_ratio": cr_name,
            "layer": layer,
            "mesh": mesh,
            "image_width": bundle_mask.shape[1],
            "image_height": bundle_mask.shape[0],
            "total_pixels": total_pixels,
            "bundle_area_pixels": bundle_area,
            "glass_fibre_area_pixels": gf_area,
            "void_noGF_area_pixels": void_area,
            "glass_fibre_fraction": gf_fraction,
            "glass_fibre_percent": gf_percent,
            "void_noGF_fraction": void_fraction,
            "void_noGF_percent": void_percent,
        }

        results.append(result)

        print(
            f"{sample_id:<12} | "
            f"GF = {gf_percent:6.2f}% | "
            f"Void/NoGF = {void_percent:6.2f}%"
        )


# ============================================================
# SAVE CSV
# ============================================================

fieldnames = [
    "sample_id",
    "compaction_ratio",
    "layer",
    "mesh",
    "image_width",
    "image_height",
    "total_pixels",
    "bundle_area_pixels",
    "glass_fibre_area_pixels",
    "void_noGF_area_pixels",
    "glass_fibre_fraction",
    "glass_fibre_percent",
    "void_noGF_fraction",
    "void_noGF_percent",
]

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as csvfile:

    writer = csv.DictWriter(
        csvfile,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


# ============================================================
# GROUP SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("SUMMARY BY COMPACTION RATIO")
print("=" * 70)

for cr_name in CR_FOLDERS:

    group = [
        row for row in results
        if row["compaction_ratio"] == cr_name
    ]

    if not group:
        continue

    gf_values = np.array(
        [row["glass_fibre_percent"] for row in group]
    )

    void_values = np.array(
        [row["void_noGF_percent"] for row in group]
    )

    print(f"\n{cr_name}")
    print(f"Samples: {len(group)}")

    print(
        "Glass fibre %:"
        f" mean={gf_values.mean():.2f},"
        f" std={gf_values.std():.2f},"
        f" min={gf_values.min():.2f},"
        f" max={gf_values.max():.2f}"
    )

    print(
        "Void/NoGF %:"
        f" mean={void_values.mean():.2f},"
        f" std={void_values.std():.2f},"
        f" min={void_values.min():.2f},"
        f" max={void_values.max():.2f}"
    )


# ============================================================
# OVERALL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("QUANTIFICATION COMPLETE")
print("=" * 70)

print(f"Samples processed: {len(results)}")
print(f"CSV saved to:\n{OUTPUT_CSV}")

print("""
Important:
These are pixel-area measurements derived from the provided masks.

The FLM_Void_NoGF measurement is being kept explicitly as
'Void/NoGF' rather than automatically calling it true porosity.
We will interpret it according to the dataset documentation
before using it as an ML feature or scientific measurement.
""")