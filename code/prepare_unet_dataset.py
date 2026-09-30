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

# NEW 70/15/15 specimen-level split
SPLIT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "final_70_15_15"
    / "sample_split_70_15_15.csv"
)

# NEW output folder.
# This prevents mixing with the old 30/9/9 experiment.
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_70_15_15"
    / "unet_dataset_256"
)

PATCH_SIZE = 256
STRIDE = 256

# Keep patch only if at least 50% of it is inside bundle ROI
MIN_BUNDLE_COVERAGE = 0.50


# ============================================================
# FIND FILES FOR ONE SPECIMEN
# ============================================================

def find_sample_files(sample_name):

    # --------------------------------------------------------
    # Find original SEM image
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
    # Find corresponding mask folder
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
    # Bundle ROI mask
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
    # Glass-fibre ground-truth mask
    #
    # Most specimens:
    # SEM_GF_Mask.tif
    #
    # Special specimen CR0_L1M1:
    # SEM_GF_Green_Large.tif
    # --------------------------------------------------------

    normal_gf = (
        mask_folder
        / "SEM_GF_Mask.tif"
    )

    special_gf = (
        mask_folder
        / "SEM_GF_Green_Large.tif"
    )

    if normal_gf.exists():
        gf_mask_path = normal_gf

    elif special_gf.exists():
        gf_mask_path = special_gf

    else:
        raise FileNotFoundError(
            f"\nGlass-fibre mask missing for "
            f"{sample_name}.\n"
            f"Checked:\n"
            f"{normal_gf}\n"
            f"{special_gf}"
        )

    return (
        image_path,
        gf_mask_path,
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
    print("FINAL 70/15/15 - U-NET PATCH DATASET PREPARATION")
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

    split_df = pd.read_csv(
        SPLIT_CSV
    )

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

    print(
        f"\nOriginal specimens in split file: "
        f"{len(split_df)}"
    )

    if len(split_df) != 48:
        raise ValueError(
            f"\nExpected 48 specimens, "
            f"but split CSV contains {len(split_df)}."
        )

    # ========================================================
    # VERIFY SPECIMEN-LEVEL SPLIT COUNTS
    # ========================================================

    split_df["split"] = (
        split_df["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    split_df["sample_id"] = (
        split_df["sample_id"]
        .astype(str)
        .str.strip()
    )

    specimen_counts = (
        split_df["split"]
        .value_counts()
        .to_dict()
    )

    print("\nSpecimen-level split:")

    for split_name in [
        "train",
        "validation",
        "test"
    ]:
        print(
            f"  {split_name:<10}: "
            f"{specimen_counts.get(split_name, 0)}"
        )

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
    # SPECIMEN LEAKAGE CHECK BEFORE PATCH EXTRACTION
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

    train_val_overlap = (
        train_ids & validation_ids
    )

    train_test_overlap = (
        train_ids & test_ids
    )

    val_test_overlap = (
        validation_ids & test_ids
    )

    print("\n" + "=" * 80)
    print("PRE-PATCH DATA LEAKAGE CHECK")
    print("=" * 80)

    print(
        "TRAIN IDs ∩ VALIDATION IDs =",
        sorted(train_val_overlap)
    )

    print(
        "TRAIN IDs ∩ TEST IDs       =",
        sorted(train_test_overlap)
    )

    print(
        "VALIDATION IDs ∩ TEST IDs  =",
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

    # ========================================================
    # CLEAN ONLY THE NEW OUTPUT DIRECTORY
    #
    # This does NOT touch:
    # dataset/
    # old outputs/
    # old experiment
    # ========================================================

    if OUTPUT_ROOT.exists():

        print(
            "\nRemoving previous contents of NEW "
            "patch output folder:"
        )

        print(OUTPUT_ROOT)

        shutil.rmtree(
            OUTPUT_ROOT
        )

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

    processed_samples = {
        "train": set(),
        "validation": set(),
        "test": set()
    }

    sample_statistics = []

    missing_file_count = 0
    dimension_problem_count = 0

    # ========================================================
    # PROCESS EACH ORIGINAL SPECIMEN
    # ========================================================

    for _, row in split_df.iterrows():

        sample_name = row["sample_id"]
        split_name = row["split"]

        cr = str(
            row["compaction_ratio"]
        )

        layer = int(
            row["layer"]
        )

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
        # FIND FILES
        # ====================================================

        try:

            (
                image_path,
                gf_mask_path,
                bundle_mask_path

            ) = find_sample_files(
                sample_name
            )

        except FileNotFoundError:

            missing_file_count += 1
            raise

        print(
            f"  SEM    : {image_path.name}"
        )

        print(
            f"  GF mask: {gf_mask_path.name}"
        )

        print(
            f"  Bundle : {bundle_mask_path.name}"
        )

        # ====================================================
        # LOAD DATA
        # ====================================================

        with Image.open(
            image_path
        ) as img:

            image_np = np.array(
                img.convert("RGB")
            )

        with Image.open(
            gf_mask_path
        ) as img:

            gf_np = np.array(
                img.convert("L")
            )

        with Image.open(
            bundle_mask_path
        ) as img:

            bundle_np = np.array(
                img.convert("L")
            )

        h, w = image_np.shape[:2]

        # ====================================================
        # VERIFY DIMENSIONS
        # ====================================================

        if gf_np.shape != (h, w):

            dimension_problem_count += 1

            raise ValueError(
                f"\nGF mask dimension mismatch "
                f"for {sample_name}\n"
                f"SEM: {image_np.shape}\n"
                f"GF : {gf_np.shape}"
            )

        if bundle_np.shape != (h, w):

            dimension_problem_count += 1

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

        gf_binary = (
            (gf_np > 0)
            .astype(np.uint8)
            * 255
        )

        bundle_binary = (
            (bundle_np > 0)
            .astype(np.uint8)
            * 255
        )

        # ====================================================
        # PATCH COUNTERS FOR THIS SPECIMEN
        # ====================================================

        candidate_count = 0
        removed_count = 0
        saved_count = 0

        # ====================================================
        # EXTRACT PATCHES
        #
        # IMPORTANT:
        # SEM, GF and bundle ROI all use EXACTLY the same
        # x/y coordinates.
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

                # --------------------------------------------
                # SEM patch
                # --------------------------------------------

                image_patch = image_np[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ]

                # --------------------------------------------
                # GF mask - SAME COORDINATES
                # --------------------------------------------

                gf_patch = gf_binary[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ]

                # --------------------------------------------
                # Bundle mask - SAME COORDINATES
                # --------------------------------------------

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

                if gf_patch.shape != (
                    PATCH_SIZE,
                    PATCH_SIZE
                ):
                    raise RuntimeError(
                        "Unexpected GF patch size."
                    )

                if bundle_patch.shape != (
                    PATCH_SIZE,
                    PATCH_SIZE
                ):
                    raise RuntimeError(
                        "Unexpected bundle patch size."
                    )

                # --------------------------------------------
                # BUNDLE ROI COVERAGE
                # --------------------------------------------

                bundle_coverage = float(
                    np.mean(
                        bundle_patch > 0
                    )
                )

                # --------------------------------------------
                # FILTER PATCHES OUTSIDE USEFUL ROI
                # --------------------------------------------

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
                # GF INFORMATION
                #
                # NOTE:
                # We do NOT remove a patch simply because
                # GF is absent. Valid negative/background
                # patches can be useful for binary
                # segmentation.
                # --------------------------------------------

                gf_positive_pixels = int(
                    np.sum(
                        gf_patch > 0
                    )
                )

                gf_fraction = float(
                    np.mean(
                        gf_patch > 0
                    )
                )

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
                # SAVE EXACT PAIRS
                # --------------------------------------------

                save_patch(
                    image_patch,
                    image_output
                )

                save_patch(
                    gf_patch,
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

                    "gf_positive_pixels":
                        gf_positive_pixels,

                    "gf_fraction":
                        gf_fraction,

                    "image_patch":
                        str(image_output),

                    "gf_mask_patch":
                        str(mask_output),

                    "bundle_mask_patch":
                        str(bundle_output)
                })

                saved_count += 1

                split_saved_counts[
                    split_name
                ] += 1

                processed_samples[
                    split_name
                ].add(
                    sample_name
                )

        # ====================================================
        # PER-SPECIMEN SUMMARY
        # ====================================================

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
                saved_count
        })

        print(
            f"  Candidate patches : "
            f"{candidate_count}"
        )

        print(
            f"  Removed (<50% ROI): "
            f"{removed_count}"
        )

        print(
            f"  Final usable       : "
            f"{saved_count}"
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
        / "unet_patch_manifest.csv"
    )

    manifest_df.to_csv(
        manifest_path,
        index=False
    )

    # ========================================================
    # SAVE PER-SPECIMEN PATCH STATISTICS
    # ========================================================

    statistics_df = pd.DataFrame(
        sample_statistics
    )

    statistics_path = (
        OUTPUT_ROOT
        / "specimen_patch_statistics.csv"
    )

    statistics_df.to_csv(
        statistics_path,
        index=False
    )

    # ========================================================
    # VERIFY SAVED IMAGE/MASK PAIRS
    # ========================================================

    print("\n" + "=" * 80)
    print("VERIFYING SAVED IMAGE / MASK PAIRS")
    print("=" * 80)

    pair_problem = False

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
            f"  GF masks    : {len(masks)}"
        )

        print(
            f"  Bundle masks: {len(rois)}"
        )

        if not (
            len(images)
            == len(masks)
            == len(rois)
        ):

            pair_problem = True

        # Check filenames correspond exactly
        image_stems = {
            p.stem
            for p in images
        }

        mask_stems = {
            p.stem
            for p in masks
        }

        roi_stems = {
            p.stem
            for p in rois
        }

        if not (
            image_stems
            == mask_stems
            == roi_stems
        ):

            pair_problem = True

    if pair_problem:
        raise RuntimeError(
            "Image/mask pairing problem detected."
        )

    print(
        "\nPASS: Every saved SEM patch has "
        "its matching GF and bundle mask."
    )

    # ========================================================
    # PATCH-LEVEL SPECIMEN LEAKAGE CHECK
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

        print(
            "\nLEAKED SPECIMENS:"
        )

        print(
            leaked_samples
        )

        raise RuntimeError(
            "A specimen contributed patches "
            "to multiple splits."
        )

    print(
        "\nPASS: No original specimen contributes "
        "patches to more than one split."
    )

    # ========================================================
    # VERIFY PATCH SPECIMEN IDS MATCH ORIGINAL SPLIT
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
            expected_ids
            - manifest_ids
        )

        unexpected_ids = (
            manifest_ids
            - expected_ids
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

        if unexpected_ids:
            raise RuntimeError(
                f"Unexpected specimen IDs found "
                f"in {split_name} patches."
            )

        if missing_ids:
            raise RuntimeError(
                f"Some {split_name} specimens "
                f"produced zero usable patches."
            )

    # ========================================================
    # FINAL PATCH COUNTS
    # ========================================================

    print("\n" + "=" * 80)
    print("FINAL PATCH QUALITY-CONTROL SUMMARY")
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
        f"\nOriginal independent specimens: "
        f"{len(split_df)}"
    )

    print(
        f"\nPatch size: "
        f"{PATCH_SIZE} x {PATCH_SIZE}"
    )

    print(
        f"Stride: {STRIDE}"
    )

    print(
        f"Minimum bundle ROI coverage: "
        f"{MIN_BUNDLE_COVERAGE * 100:.0f}%"
    )

    print(
        f"\nPatches before ROI filtering: "
        f"{total_candidates}"
    )

    print(
        f"Removed patches: "
        f"{total_removed}"
    )

    print(
        f"Final usable patches: "
        f"{total_saved}"
    )

    print(
        "\nUsable patches by split:"
    )

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
        "\nBefore/removed/final by split:"
    )

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        print(
            f"  {split_name:<10}: "
            f"{split_candidate_counts[split_name]} "
            f"/ {split_removed_counts[split_name]} "
            f"/ {split_saved_counts[split_name]}"
        )

    # ========================================================
    # GF PATCH INFORMATION
    # ========================================================

    print("\n" + "=" * 80)
    print("GLASS-FIBRE PATCH INFORMATION")
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

        zero_gf = int(
            (
                subset["gf_positive_pixels"]
                == 0
            ).sum()
        )

        positive_gf = int(
            (
                subset["gf_positive_pixels"]
                > 0
            ).sum()
        )

        print(
            f"\n{split_name.upper()}"
        )

        print(
            f"  Patches with GF pixels : "
            f"{positive_gf}"
        )

        print(
            f"  Patches with zero GF   : "
            f"{zero_gf}"
        )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n" + "=" * 80)
    print("256 x 256 PATCH EXTRACTION COMPLETE")
    print("=" * 80)

    print(
        f"\nPatch manifest:\n"
        f"{manifest_path}"
    )

    print(
        f"\nSpecimen statistics:\n"
        f"{statistics_path}"
    )

    print(
        f"\nDataset saved to:\n"
        f"{OUTPUT_ROOT}"
    )

    print(
        "\nIMPORTANT:"
        "\n- These are ORIGINAL patches."
        "\n- No augmentation has been performed."
        "\n- Validation has NOT been augmented."
        "\n- Test has NOT been augmented."
        "\n- FLM_Void_NoGF has NOT been used as a training target."
        "\n- GF segmentation is the current supervised task."
        "\n- No model has been trained."
    )


if __name__ == "__main__":
    main()