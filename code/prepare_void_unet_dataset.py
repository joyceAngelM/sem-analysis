from pathlib import Path
import shutil

import pandas as pd
import numpy as np
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "Multi_Modal_Microscopy_DataSet_paper"
    / "Multi_Modal_Microscopy_DataSet_paper"
)

SPLIT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "final_70_15_15"
    / "sample_split_70_15_15.csv"
)

# Completely separate from the existing GF dataset
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_70_15_15"
    / "void_unet_dataset_256"
)

PATCH_SIZE = 256
STRIDE = 256

# Same ROI rule used by the GF experiment
MIN_BUNDLE_COVERAGE = 0.50


# ============================================================
# FIND FILES FOR ONE SPECIMEN
# ============================================================

def find_sample_files(sample_name):

    # --------------------------------------------------------
    # Original SEM
    # --------------------------------------------------------

    image_candidates = [
        p
        for p in DATASET_ROOT.rglob(f"{sample_name}.tif")
        if p.is_file()
        and "Bundles_Stack" in str(p)
    ]

    if len(image_candidates) != 1:
        raise FileNotFoundError(
            f"\nExpected exactly 1 SEM image for "
            f"{sample_name}, but found "
            f"{len(image_candidates)}.\n"
            + "\n".join(str(p) for p in image_candidates)
        )

    image_path = image_candidates[0]

    # --------------------------------------------------------
    # Quantitative mask folder
    # --------------------------------------------------------

    mask_folders = [
        p
        for p in DATASET_ROOT.rglob(sample_name)
        if p.is_dir()
        and "Binary_Masks" in str(p)
    ]

    if len(mask_folders) != 1:
        raise FileNotFoundError(
            f"\nExpected exactly 1 mask folder for "
            f"{sample_name}, but found "
            f"{len(mask_folders)}.\n"
            + "\n".join(str(p) for p in mask_folders)
        )

    mask_folder = mask_folders[0]

    # --------------------------------------------------------
    # Bundle ROI
    # --------------------------------------------------------

    bundle_mask_path = (
        mask_folder
        / "SEM_Bundle_Mask.tif"
    )

    if not bundle_mask_path.exists():
        raise FileNotFoundError(
            f"\nBundle mask missing for {sample_name}:\n"
            f"{bundle_mask_path}"
        )

    # --------------------------------------------------------
    # VOID / NoGF GROUND-TRUTH TARGET
    # --------------------------------------------------------

    void_mask_path = (
        mask_folder
        / "FLM_Void_NoGF.tif"
    )

    if not void_mask_path.exists():
        raise FileNotFoundError(
            f"\nVoid/NoGF mask missing for {sample_name}:\n"
            f"{void_mask_path}"
        )

    return (
        image_path,
        void_mask_path,
        bundle_mask_path
    )


# ============================================================
# SAVE PATCH
# ============================================================

def save_patch(array, output_path):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Image.fromarray(array).save(
        output_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("VOID / NoGF U-NET DATASET PREPARATION")
    print("FINAL 70/15/15 SPECIMEN SPLIT")
    print("=" * 80)

    print("\nDataset root:")
    print(DATASET_ROOT)

    print("\nSplit CSV:")
    print(SPLIT_CSV)

    print("\nOutput root:")
    print(OUTPUT_ROOT)

    # ========================================================
    # CHECK INPUTS
    # ========================================================

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"\nDataset root not found:\n{DATASET_ROOT}"
        )

    if not SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"\nSplit CSV not found:\n{SPLIT_CSV}"
        )

    split_df = pd.read_csv(SPLIT_CSV)

    required_columns = {
        "sample_id",
        "compaction_ratio",
        "layer",
        "split"
    }

    missing_columns = (
        required_columns
        - set(split_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"\nMissing columns in split CSV: "
            f"{missing_columns}"
        )

    # ========================================================
    # VERIFY SPECIMEN SPLIT
    # ========================================================

    specimen_counts = (
        split_df["split"]
        .value_counts()
        .to_dict()
    )

    print("\nSpecimen counts:")
    print(specimen_counts)

    if specimen_counts.get("train", 0) != 34:
        raise ValueError(
            "Expected 34 training specimens."
        )

    if specimen_counts.get("validation", 0) != 7:
        raise ValueError(
            "Expected 7 validation specimens."
        )

    if specimen_counts.get("test", 0) != 7:
        raise ValueError(
            "Expected 7 test specimens."
        )

    # ========================================================
    # SPECIMEN LEAKAGE CHECK
    # ========================================================

    train_ids = set(
        split_df.loc[
            split_df["split"] == "train",
            "sample_id"
        ]
    )

    validation_ids = set(
        split_df.loc[
            split_df["split"] == "validation",
            "sample_id"
        ]
    )

    test_ids = set(
        split_df.loc[
            split_df["split"] == "test",
            "sample_id"
        ]
    )

    train_val_overlap = train_ids & validation_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = validation_ids & test_ids

    print("\n" + "=" * 80)
    print("PRE-PATCH DATA LEAKAGE CHECK")
    print("=" * 80)

    print(
        "TRAIN IDs intersection VALIDATION IDs =",
        sorted(train_val_overlap)
    )

    print(
        "TRAIN IDs intersection TEST IDs       =",
        sorted(train_test_overlap)
    )

    print(
        "VALIDATION IDs intersection TEST IDs  =",
        sorted(val_test_overlap)
    )

    if (
        train_val_overlap
        or train_test_overlap
        or val_test_overlap
    ):
        raise RuntimeError(
            "Specimen-level data leakage detected."
        )

    print("\nPASS: No specimen-level leakage.")

    # ========================================================
    # CLEAN ONLY VOID OUTPUT DIRECTORY
    # ========================================================

    if OUTPUT_ROOT.exists():

        print(
            "\nRemoving previous VOID dataset output:"
        )

        print(OUTPUT_ROOT)

        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # COUNTERS
    # ========================================================

    manifest_rows = []

    split_saved_counts = {
        "train": 0,
        "validation": 0,
        "test": 0
    }

    split_candidate_counts = {
        "train": 0,
        "validation": 0,
        "test": 0
    }

    split_removed_counts = {
        "train": 0,
        "validation": 0,
        "test": 0
    }

    split_void_positive_counts = {
        "train": 0,
        "validation": 0,
        "test": 0
    }

    split_void_negative_counts = {
        "train": 0,
        "validation": 0,
        "test": 0
    }

    sample_statistics = []

    # ========================================================
    # PROCESS ORIGINAL SPECIMENS
    # ========================================================

    for _, row in split_df.iterrows():

        sample_name = row["sample_id"]
        split_name = row["split"]

        cr = str(row["compaction_ratio"])
        layer = int(row["layer"])

        if split_name not in {
            "train",
            "validation",
            "test"
        }:
            raise ValueError(
                f"Invalid split '{split_name}' "
                f"for {sample_name}"
            )

        print("\n" + "-" * 80)

        print(
            f"Processing {sample_name} "
            f"[{split_name}]"
        )

        # ====================================================
        # FIND ORIGINAL FILES
        # ====================================================

        (
            image_path,
            void_mask_path,
            bundle_mask_path
        ) = find_sample_files(sample_name)

        print(f"  SEM       : {image_path.name}")
        print(f"  Void mask : {void_mask_path.name}")
        print(f"  Bundle    : {bundle_mask_path.name}")

        # ====================================================
        # LOAD DATA
        # ====================================================

        with Image.open(image_path) as img:

            # Match existing GF pipeline:
            # first/default TIFF frame converted to RGB.
            image_np = np.array(
                img.convert("RGB")
            )

        with Image.open(void_mask_path) as img:

            void_np = np.array(
                img.convert("L")
            )

        with Image.open(bundle_mask_path) as img:

            bundle_np = np.array(
                img.convert("L")
            )

        h, w = image_np.shape[:2]

        # ====================================================
        # DIMENSION CHECK
        # ====================================================

        if void_np.shape != (h, w):

            raise ValueError(
                f"\nVoid mask dimension mismatch "
                f"for {sample_name}\n"
                f"SEM : {image_np.shape}\n"
                f"Void: {void_np.shape}"
            )

        if bundle_np.shape != (h, w):

            raise ValueError(
                f"\nBundle mask dimension mismatch "
                f"for {sample_name}\n"
                f"SEM   : {image_np.shape}\n"
                f"Bundle: {bundle_np.shape}"
            )

        print(
            f"  Dimensions verified: "
            f"{w} x {h}"
        )

        # ====================================================
        # STRICT BINARY MASKS
        # ====================================================

        void_binary = (
            (void_np > 0)
            .astype(np.uint8)
            * 255
        )

        bundle_binary = (
            (bundle_np > 0)
            .astype(np.uint8)
            * 255
        )

        # ====================================================
        # SPECIMEN COUNTERS
        # ====================================================

        candidate_count = 0
        removed_count = 0
        saved_count = 0

        void_positive_patch_count = 0
        void_negative_patch_count = 0

        specimen_void_pixels = 0
        specimen_roi_pixels = 0

        # ====================================================
        # EXTRACT ALIGNED 256 x 256 PATCHES
        # ====================================================

        for y in range(
            0,
            h - PATCH_SIZE + 1,
            STRIDE
        ):

            for x in range(
                0,
                w - PATCH_SIZE + 1,
                STRIDE
            ):

                candidate_count += 1

                split_candidate_counts[
                    split_name
                ] += 1

                # SEM patch
                image_patch = image_np[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ]

                # VOID target - SAME coordinates
                void_patch = void_binary[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ]

                # Bundle ROI - SAME coordinates
                bundle_patch = bundle_binary[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ]

                # --------------------------------------------
                # SAFETY CHECK
                # --------------------------------------------

                if image_patch.shape[:2] != (
                    PATCH_SIZE,
                    PATCH_SIZE
                ):
                    raise RuntimeError(
                        "Unexpected SEM patch size."
                    )

                if void_patch.shape != (
                    PATCH_SIZE,
                    PATCH_SIZE
                ):
                    raise RuntimeError(
                        "Unexpected Void patch size."
                    )

                if bundle_patch.shape != (
                    PATCH_SIZE,
                    PATCH_SIZE
                ):
                    raise RuntimeError(
                        "Unexpected bundle patch size."
                    )

                # --------------------------------------------
                # ROI COVERAGE
                # --------------------------------------------

                bundle_bool = bundle_patch > 0

                bundle_coverage = float(
                    np.mean(bundle_bool)
                )

                if (
                    bundle_coverage
                    < MIN_BUNDLE_COVERAGE
                ):

                    removed_count += 1

                    split_removed_counts[
                        split_name
                    ] += 1

                    continue

                # --------------------------------------------
                # VOID INFORMATION
                # --------------------------------------------

                void_bool = void_patch > 0

                # Quantification within valid bundle ROI
                void_inside_roi = (
                    void_bool & bundle_bool
                )

                void_positive_pixels = int(
                    void_inside_roi.sum()
                )

                roi_positive_pixels = int(
                    bundle_bool.sum()
                )

                if roi_positive_pixels > 0:

                    void_fraction_roi = float(
                        void_positive_pixels
                        / roi_positive_pixels
                    )

                else:

                    void_fraction_roi = 0.0

                # Full patch fraction retained for reference
                void_fraction_full_patch = float(
                    np.mean(void_bool)
                )

                specimen_void_pixels += (
                    void_positive_pixels
                )

                specimen_roi_pixels += (
                    roi_positive_pixels
                )

                if void_positive_pixels > 0:

                    void_positive_patch_count += 1

                    split_void_positive_counts[
                        split_name
                    ] += 1

                else:

                    void_negative_patch_count += 1

                    split_void_negative_counts[
                        split_name
                    ] += 1

                # --------------------------------------------
                # PATCH ID
                # --------------------------------------------

                patch_id = (
                    f"{sample_name}"
                    f"_y{y:04d}"
                    f"_x{x:04d}"
                )

                # --------------------------------------------
                # OUTPUT PATHS
                # --------------------------------------------

                image_output = (
                    OUTPUT_ROOT
                    / split_name
                    / "images"
                    / f"{patch_id}.png"
                )

                mask_output = (
                    OUTPUT_ROOT
                    / split_name
                    / "masks"
                    / f"{patch_id}.png"
                )

                bundle_output = (
                    OUTPUT_ROOT
                    / split_name
                    / "bundle_masks"
                    / f"{patch_id}.png"
                )

                # --------------------------------------------
                # SAVE EXACT ALIGNED PAIRS
                # --------------------------------------------

                save_patch(
                    image_patch,
                    image_output
                )

                save_patch(
                    void_patch,
                    mask_output
                )

                save_patch(
                    bundle_patch,
                    bundle_output
                )

                # --------------------------------------------
                # MANIFEST
                # --------------------------------------------

                manifest_rows.append({

                    "sample_id":
                        sample_name,

                    "compaction_ratio":
                        cr,

                    "layer":
                        layer,

                    "split":
                        split_name,

                    "patch_id":
                        patch_id,

                    "x":
                        x,

                    "y":
                        y,

                    "patch_size":
                        PATCH_SIZE,

                    "bundle_coverage":
                        bundle_coverage,

                    "roi_positive_pixels":
                        roi_positive_pixels,

                    "void_positive_pixels":
                        void_positive_pixels,

                    "void_fraction_roi":
                        void_fraction_roi,

                    "void_fraction_full_patch":
                        void_fraction_full_patch,

                    "image_patch":
                        str(image_output),

                    "void_mask_patch":
                        str(mask_output),

                    "bundle_mask_patch":
                        str(bundle_output)
                })

                saved_count += 1

                split_saved_counts[
                    split_name
                ] += 1

        # ====================================================
        # PER-SPECIMEN SUMMARY
        # ====================================================

        if specimen_roi_pixels > 0:

            specimen_void_fraction = (
                specimen_void_pixels
                / specimen_roi_pixels
            )

        else:

            specimen_void_fraction = 0.0

        sample_statistics.append({

            "sample_id":
                sample_name,

            "compaction_ratio":
                cr,

            "layer":
                layer,

            "split":
                split_name,

            "width":
                w,

            "height":
                h,

            "candidate_patches":
                candidate_count,

            "removed_roi_patches":
                removed_count,

            "usable_patches":
                saved_count,

            "void_positive_patches":
                void_positive_patch_count,

            "void_negative_patches":
                void_negative_patch_count,

            "void_pixels_inside_roi":
                specimen_void_pixels,

            "roi_pixels":
                specimen_roi_pixels,

            "void_fraction_inside_roi":
                specimen_void_fraction,

            "void_percent_inside_roi":
                specimen_void_fraction * 100.0
        })

        print(
            f"  Candidate patches   : "
            f"{candidate_count}"
        )

        print(
            f"  Removed (<50% ROI)  : "
            f"{removed_count}"
        )

        print(
            f"  Final usable        : "
            f"{saved_count}"
        )

        print(
            f"  Void-positive       : "
            f"{void_positive_patch_count}"
        )

        print(
            f"  Void-negative       : "
            f"{void_negative_patch_count}"
        )

        print(
            f"  Void inside ROI     : "
            f"{specimen_void_fraction * 100:.2f}%"
        )

    # ========================================================
    # CREATE MANIFEST
    # ========================================================

    if not manifest_rows:

        raise RuntimeError(
            "No usable patches were generated."
        )

    manifest_df = pd.DataFrame(
        manifest_rows
    )

    manifest_path = (
        OUTPUT_ROOT
        / "void_unet_patch_manifest.csv"
    )

    manifest_df.to_csv(
        manifest_path,
        index=False
    )

    # ========================================================
    # SAVE SPECIMEN STATISTICS
    # ========================================================

    statistics_df = pd.DataFrame(
        sample_statistics
    )

    statistics_path = (
        OUTPUT_ROOT
        / "void_specimen_patch_statistics.csv"
    )

    statistics_df.to_csv(
        statistics_path,
        index=False
    )

    # ========================================================
    # VERIFY SAVED PAIRS
    # ========================================================

    print("\n" + "=" * 80)
    print("VERIFYING SEM / VOID / ROI PATCH PAIRS")
    print("=" * 80)

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        image_dir = (
            OUTPUT_ROOT
            / split_name
            / "images"
        )

        mask_dir = (
            OUTPUT_ROOT
            / split_name
            / "masks"
        )

        roi_dir = (
            OUTPUT_ROOT
            / split_name
            / "bundle_masks"
        )

        images = sorted(
            image_dir.glob("*.png")
        )

        masks = sorted(
            mask_dir.glob("*.png")
        )

        rois = sorted(
            roi_dir.glob("*.png")
        )

        print(
            f"\n{split_name.upper()}"
        )

        print(
            f"  SEM patches : {len(images)}"
        )

        print(
            f"  Void masks  : {len(masks)}"
        )

        print(
            f"  Bundle masks: {len(rois)}"
        )

        image_stems = {
            p.stem for p in images
        }

        mask_stems = {
            p.stem for p in masks
        }

        roi_stems = {
            p.stem for p in rois
        }

        if not (
            image_stems
            == mask_stems
            == roi_stems
        ):

            raise RuntimeError(
                f"Image/mask pairing problem "
                f"in {split_name}."
            )

    print(
        "\nPASS: Every SEM patch has its matching "
        "Void/NoGF and bundle ROI mask."
    )

    # ========================================================
    # PATCH-LEVEL LEAKAGE CHECK
    # ========================================================

    print("\n" + "=" * 80)
    print("PATCH DATA LEAKAGE CHECK")
    print("=" * 80)

    patch_sample_splits = (
        manifest_df
        .groupby("sample_id")["split"]
        .nunique()
    )

    leaked_samples = (
        patch_sample_splits[
            patch_sample_splits > 1
        ]
    )

    if not leaked_samples.empty:

        print(leaked_samples)

        raise RuntimeError(
            "A specimen contributed patches "
            "to multiple splits."
        )

    print(
        "\nPASS: No specimen contributes patches "
        "to more than one split."
    )

    # ========================================================
    # VERIFY EXPECTED SPECIMENS
    # ========================================================

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        manifest_ids = set(
            manifest_df.loc[
                manifest_df["split"]
                == split_name,
                "sample_id"
            ]
        )

        expected_ids = set(
            split_df.loc[
                split_df["split"]
                == split_name,
                "sample_id"
            ]
        )

        missing_ids = (
            expected_ids - manifest_ids
        )

        unexpected_ids = (
            manifest_ids - expected_ids
        )

        print(
            f"\n{split_name.upper()} specimen check:"
        )

        print(
            f"  Expected specimens : "
            f"{len(expected_ids)}"
        )

        print(
            f"  Patch specimens    : "
            f"{len(manifest_ids)}"
        )

        print(
            f"  Missing specimens  : "
            f"{sorted(missing_ids)}"
        )

        print(
            f"  Unexpected         : "
            f"{sorted(unexpected_ids)}"
        )

        if missing_ids or unexpected_ids:

            raise RuntimeError(
                f"Specimen mismatch in "
                f"{split_name}."
            )

    # ========================================================
    # VOID DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 80)
    print("VOID / NoGF PATCH DISTRIBUTION")
    print("=" * 80)

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        subset = manifest_df[
            manifest_df["split"]
            == split_name
        ]

        positive = int(
            (
                subset[
                    "void_positive_pixels"
                ] > 0
            ).sum()
        )

        negative = int(
            (
                subset[
                    "void_positive_pixels"
                ] == 0
            ).sum()
        )

        total_void_pixels = int(
            subset[
                "void_positive_pixels"
            ].sum()
        )

        total_roi_pixels = int(
            subset[
                "roi_positive_pixels"
            ].sum()
        )

        if total_roi_pixels > 0:

            overall_void_percent = (
                total_void_pixels
                / total_roi_pixels
                * 100.0
            )

        else:

            overall_void_percent = 0.0

        print(
            f"\n{split_name.upper()}"
        )

        print(
            f"  Void-positive patches : "
            f"{positive}"
        )

        print(
            f"  Zero-void patches     : "
            f"{negative}"
        )

        print(
            f"  Void pixels in ROI    : "
            f"{total_void_pixels:,}"
        )

        print(
            f"  Overall void fraction : "
            f"{overall_void_percent:.2f}%"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("VOID U-NET DATASET PREPARATION COMPLETE")
    print("=" * 80)

    total_candidates = sum(
        split_candidate_counts.values()
    )

    total_removed = sum(
        split_removed_counts.values()
    )

    total_saved = sum(
        split_saved_counts.values()
    )

    print(
        f"\nOriginal specimens: "
        f"{len(split_df)}"
    )

    print(
        f"Patch size: "
        f"{PATCH_SIZE} x {PATCH_SIZE}"
    )

    print(
        f"Stride: {STRIDE}"
    )

    print(
        f"Minimum bundle ROI: "
        f"{MIN_BUNDLE_COVERAGE * 100:.0f}%"
    )

    print(
        f"\nCandidate patches: "
        f"{total_candidates}"
    )

    print(
        f"Removed by ROI: "
        f"{total_removed}"
    )

    print(
        f"Final usable: "
        f"{total_saved}"
    )

    print("\nUsable patches by split:")

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        print(
            f"  {split_name:<10}: "
            f"{split_saved_counts[split_name]}"
        )

    print(
        f"\nManifest:\n"
        f"{manifest_path}"
    )

    print(
        f"\nSpecimen statistics:\n"
        f"{statistics_path}"
    )

    print(
        f"\nVoid dataset:\n"
        f"{OUTPUT_ROOT}"
    )

    print("""
IMPORTANT:
- Target = FLM_Void_NoGF.
- Bundle mask is used as the valid ROI.
- Same original 70/15/15 specimen split is preserved.
- Test specimens remain untouched for model selection.
- No augmentation has been performed.
- No training has been performed.
- Existing GF dataset has NOT been modified.
- Existing GF U-Net checkpoint has NOT been modified.
""")


if __name__ == "__main__":
    main()