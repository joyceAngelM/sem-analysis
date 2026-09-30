from pathlib import Path
import random
import shutil

import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_70_15_15"
    / "void_unet_dataset_256"
)

SOURCE_MANIFEST = (
    SOURCE_ROOT
    / "void_unet_patch_manifest.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_70_15_15"
    / "void_unet_dataset_augmented"
)

TARGET_TOTAL_TRAINING_PAIRS = 7000

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ============================================================
# TRANSFORM FUNCTION
# ============================================================

def apply_transform(array, transform_name):

    if transform_name == "hflip":
        return np.fliplr(array).copy()

    elif transform_name == "vflip":
        return np.flipud(array).copy()

    elif transform_name == "rot90":
        return np.rot90(array, k=1).copy()

    elif transform_name == "rot180":
        return np.rot90(array, k=2).copy()

    elif transform_name == "rot270":
        return np.rot90(array, k=3).copy()

    elif transform_name == "hvflip":
        return np.flipud(
            np.fliplr(array)
        ).copy()

    elif transform_name == "transpose":

        if array.ndim == 3:
            return np.transpose(
                array,
                axes=(1, 0, 2)
            ).copy()

        return np.transpose(
            array
        ).copy()

    else:
        raise ValueError(
            f"Unknown transform: {transform_name}"
        )


# ============================================================
# LOAD FUNCTIONS
# ============================================================

def load_rgb(path):

    with Image.open(path) as image:

        return np.array(
            image.convert("RGB")
        )


def load_mask(path):

    with Image.open(path) as image:

        array = np.array(
            image.convert("L")
        )

    # Preserve binary labels strictly as 0/255
    return (
        (array > 0)
        .astype(np.uint8)
        * 255
    )


# ============================================================
# SAVE
# ============================================================

def save_array(array, path):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Image.fromarray(
        array
    ).save(path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("VOID / NoGF - TRAINING DATA AUGMENTATION")
    print("=" * 80)

    print("\nSource dataset:")
    print(SOURCE_ROOT)

    print("\nOutput dataset:")
    print(OUTPUT_ROOT)

    print(
        f"\nTarget total training pairs: "
        f"{TARGET_TOTAL_TRAINING_PAIRS}"
    )

    # ========================================================
    # CHECK SOURCE
    # ========================================================

    if not SOURCE_MANIFEST.exists():

        raise FileNotFoundError(
            f"\nManifest not found:\n"
            f"{SOURCE_MANIFEST}"
        )

    manifest = pd.read_csv(
        SOURCE_MANIFEST
    )

    required_columns = {
        "sample_id",
        "compaction_ratio",
        "layer",
        "split",
        "patch_id",
        "image_patch",
        "void_mask_patch",
        "bundle_mask_patch"
    }

    missing_columns = (
        required_columns
        - set(manifest.columns)
    )

    if missing_columns:

        raise ValueError(
            f"Missing manifest columns: "
            f"{missing_columns}"
        )

    train_df = manifest[
        manifest["split"] == "train"
    ].copy()

    validation_df = manifest[
        manifest["split"] == "validation"
    ].copy()

    test_df = manifest[
        manifest["split"] == "test"
    ].copy()

    print("\nOriginal VOID dataset:")

    print(
        f"  Train      : {len(train_df)}"
    )

    print(
        f"  Validation : {len(validation_df)}"
    )

    print(
        f"  Test       : {len(test_df)}"
    )

    # Verify expected counts

    if len(train_df) != 894:

        raise ValueError(
            f"Expected 894 original training patches, "
            f"found {len(train_df)}."
        )

    if len(validation_df) != 173:

        raise ValueError(
            f"Expected 173 validation patches, "
            f"found {len(validation_df)}."
        )

    if len(test_df) != 179:

        raise ValueError(
            f"Expected 179 test patches, "
            f"found {len(test_df)}."
        )

    # ========================================================
    # CLEAN ONLY PREVIOUS VOID AUGMENTATION OUTPUT
    # ========================================================

    if OUTPUT_ROOT.exists():

        print(
            "\nRemoving previous VOID augmentation output..."
        )

        shutil.rmtree(
            OUTPUT_ROOT
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # OUTPUT DIRECTORIES
    # ========================================================

    train_image_dir = (
        OUTPUT_ROOT
        / "train"
        / "images"
    )

    train_mask_dir = (
        OUTPUT_ROOT
        / "train"
        / "void_masks"
    )

    train_bundle_dir = (
        OUTPUT_ROOT
        / "train"
        / "bundle_masks"
    )

    for folder in [
        train_image_dir,
        train_mask_dir,
        train_bundle_dir
    ]:

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

    # ========================================================
    # COPY ORIGINAL TRAINING PAIRS
    # ========================================================

    print(
        "\nCopying original VOID training pairs..."
    )

    output_rows = []

    for _, row in train_df.iterrows():

        source_image = Path(
            row["image_patch"]
        )

        source_mask = Path(
            row["void_mask_patch"]
        )

        source_bundle = Path(
            row["bundle_mask_patch"]
        )

        patch_id = row["patch_id"]

        destination_image = (
            train_image_dir
            / f"{patch_id}.png"
        )

        destination_mask = (
            train_mask_dir
            / f"{patch_id}.png"
        )

        destination_bundle = (
            train_bundle_dir
            / f"{patch_id}.png"
        )

        shutil.copy2(
            source_image,
            destination_image
        )

        shutil.copy2(
            source_mask,
            destination_mask
        )

        shutil.copy2(
            source_bundle,
            destination_bundle
        )

        output_rows.append({

            "sample_id":
                row["sample_id"],

            "compaction_ratio":
                row["compaction_ratio"],

            "layer":
                row["layer"],

            "patch_id":
                patch_id,

            "source_patch_id":
                patch_id,

            "augmentation":
                "original",

            "image_patch":
                str(destination_image),

            "void_mask_patch":
                str(destination_mask),

            "bundle_mask_patch":
                str(destination_bundle)
        })

    original_count = len(
        output_rows
    )

    print(
        f"Original training pairs copied: "
        f"{original_count}"
    )

    # ========================================================
    # AUGMENTATION
    # ========================================================

    transforms = [
        "hflip",
        "vflip",
        "rot90",
        "rot180",
        "rot270",
        "hvflip",
        "transpose"
    ]

    required_augmented = (
        TARGET_TOTAL_TRAINING_PAIRS
        - original_count
    )

    print(
        f"\nAugmented pairs required: "
        f"{required_augmented}"
    )

    generated = 0
    cycle = 0

    while generated < required_augmented:

        shuffled_indices = list(
            train_df.index
        )

        random.shuffle(
            shuffled_indices
        )

        for index in shuffled_indices:

            if generated >= required_augmented:
                break

            row = train_df.loc[
                index
            ]

            source_image_path = Path(
                row["image_patch"]
            )

            source_mask_path = Path(
                row["void_mask_patch"]
            )

            source_bundle_path = Path(
                row["bundle_mask_patch"]
            )

            # --------------------------------------------
            # LOAD EXACT IMAGE / VOID / ROI TRIPLET
            # --------------------------------------------

            image = load_rgb(
                source_image_path
            )

            mask = load_mask(
                source_mask_path
            )

            bundle = load_mask(
                source_bundle_path
            )

            # --------------------------------------------
            # SAME TRANSFORM SELECTION AS GF EXPERIMENT
            # --------------------------------------------

            transform_name = transforms[
                cycle % len(transforms)
            ]

            # --------------------------------------------
            # APPLY IDENTICAL TRANSFORM TO ALL THREE
            # --------------------------------------------

            image_aug = apply_transform(
                image,
                transform_name
            )

            mask_aug = apply_transform(
                mask,
                transform_name
            )

            bundle_aug = apply_transform(
                bundle,
                transform_name
            )

            # --------------------------------------------
            # VERIFY DIMENSIONS
            # --------------------------------------------

            if image_aug.shape[:2] != (
                256,
                256
            ):

                raise RuntimeError(
                    "Augmented SEM patch has "
                    "incorrect dimensions."
                )

            if mask_aug.shape != (
                256,
                256
            ):

                raise RuntimeError(
                    "Augmented Void mask has "
                    "incorrect dimensions."
                )

            if bundle_aug.shape != (
                256,
                256
            ):

                raise RuntimeError(
                    "Augmented bundle mask has "
                    "incorrect dimensions."
                )

            # --------------------------------------------
            # VERIFY BINARY MASKS
            # --------------------------------------------

            mask_values = set(
                np.unique(
                    mask_aug
                ).tolist()
            )

            if not mask_values.issubset(
                {0, 255}
            ):

                raise RuntimeError(
                    f"Void mask became non-binary: "
                    f"{mask_values}"
                )

            bundle_values = set(
                np.unique(
                    bundle_aug
                ).tolist()
            )

            if not bundle_values.issubset(
                {0, 255}
            ):

                raise RuntimeError(
                    "Bundle mask became non-binary."
                )

            # --------------------------------------------
            # UNIQUE AUGMENTED ID
            # --------------------------------------------

            source_patch_id = row[
                "patch_id"
            ]

            augmented_id = (
                f"{source_patch_id}"
                f"_aug{generated + 1:05d}"
                f"_{transform_name}"
            )

            image_output = (
                train_image_dir
                / f"{augmented_id}.png"
            )

            mask_output = (
                train_mask_dir
                / f"{augmented_id}.png"
            )

            bundle_output = (
                train_bundle_dir
                / f"{augmented_id}.png"
            )

            # --------------------------------------------
            # SAVE
            # --------------------------------------------

            save_array(
                image_aug,
                image_output
            )

            save_array(
                mask_aug,
                mask_output
            )

            save_array(
                bundle_aug,
                bundle_output
            )

            output_rows.append({

                "sample_id":
                    row["sample_id"],

                "compaction_ratio":
                    row["compaction_ratio"],

                "layer":
                    row["layer"],

                "patch_id":
                    augmented_id,

                "source_patch_id":
                    source_patch_id,

                "augmentation":
                    transform_name,

                "image_patch":
                    str(image_output),

                "void_mask_patch":
                    str(mask_output),

                "bundle_mask_patch":
                    str(bundle_output)
            })

            generated += 1

        cycle += 1

        print(
            f"Generated augmented pairs: "
            f"{generated} / "
            f"{required_augmented}"
        )

    # ========================================================
    # SAVE AUGMENTED TRAINING MANIFEST
    # ========================================================

    output_manifest = pd.DataFrame(
        output_rows
    )

    manifest_path = (
        OUTPUT_ROOT
        / "augmented_void_train_manifest.csv"
    )

    output_manifest.to_csv(
        manifest_path,
        index=False
    )

    # ========================================================
    # FINAL FILE COUNTS
    # ========================================================

    image_files = list(
        train_image_dir.glob(
            "*.png"
        )
    )

    mask_files = list(
        train_mask_dir.glob(
            "*.png"
        )
    )

    bundle_files = list(
        train_bundle_dir.glob(
            "*.png"
        )
    )

    print("\n" + "=" * 80)
    print("VOID AUGMENTATION VERIFICATION")
    print("=" * 80)

    print(
        f"\nTraining SEM files       : "
        f"{len(image_files)}"
    )

    print(
        f"Training Void/NoGF masks : "
        f"{len(mask_files)}"
    )

    print(
        f"Training bundle masks    : "
        f"{len(bundle_files)}"
    )

    if not (
        len(image_files)
        == TARGET_TOTAL_TRAINING_PAIRS
        and len(mask_files)
        == TARGET_TOTAL_TRAINING_PAIRS
        and len(bundle_files)
        == TARGET_TOTAL_TRAINING_PAIRS
    ):

        raise RuntimeError(
            "Final VOID augmentation file "
            "counts are incorrect."
        )

    # ========================================================
    # VERIFY MATCHING FILENAMES
    # ========================================================

    image_stems = {
        p.stem
        for p in image_files
    }

    mask_stems = {
        p.stem
        for p in mask_files
    }

    bundle_stems = {
        p.stem
        for p in bundle_files
    }

    if not (
        image_stems
        == mask_stems
        == bundle_stems
    ):

        raise RuntimeError(
            "SEM/Void/bundle filenames "
            "do not match."
        )

    print(
        "\nPASS: Every augmented SEM patch has "
        "a matching Void/NoGF and bundle mask."
    )

    # ========================================================
    # VERIFY VALIDATION / TEST UNCHANGED
    # ========================================================

    current_validation_images = list(
        (
            SOURCE_ROOT
            / "validation"
            / "images"
        ).glob("*.png")
    )

    current_test_images = list(
        (
            SOURCE_ROOT
            / "test"
            / "images"
        ).glob("*.png")
    )

    print("\nValidation/test check:")

    print(
        f"  Validation remains: "
        f"{len(current_validation_images)}"
    )

    print(
        f"  Test remains      : "
        f"{len(current_test_images)}"
    )

    if len(
        current_validation_images
    ) != 173:

        raise RuntimeError(
            "Validation dataset changed."
        )

    if len(
        current_test_images
    ) != 179:

        raise RuntimeError(
            "Test dataset changed."
        )

    # ========================================================
    # AUGMENTATION DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 80)
    print("VOID AUGMENTATION DISTRIBUTION")
    print("=" * 80)

    counts = (
        output_manifest[
            "augmentation"
        ]
        .value_counts()
    )

    print(
        counts.to_string()
    )

    # ========================================================
    # VERIFY SPECIMEN IDs
    # ========================================================

    original_specimens = set(
        train_df[
            "sample_id"
        ].astype(str)
    )

    augmented_specimens = set(
        output_manifest[
            "sample_id"
        ].astype(str)
    )

    if (
        original_specimens
        != augmented_specimens
    ):

        raise RuntimeError(
            "Training specimen IDs changed "
            "during augmentation."
        )

    print(
        "\nPASS: Augmentation introduced "
        "no new specimens."
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("VOID TRAINING AUGMENTATION COMPLETE")
    print("=" * 80)

    print(
        f"\nOriginal training patches : "
        f"{original_count}"
    )

    print(
        f"Augmented patches created : "
        f"{generated}"
    )

    print(
        f"Total training pairs      : "
        f"{len(output_manifest)}"
    )

    print(
        f"\nValidation patches        : "
        f"{len(validation_df)} "
        f"(UNCHANGED)"
    )

    print(
        f"Test patches              : "
        f"{len(test_df)} "
        f"(UNCHANGED)"
    )

    print(
        f"\nManifest saved to:\n"
        f"{manifest_path}"
    )

    print("""
IMPORTANT:
7000 training pairs are NOT 7000 independent SEM specimens.
They originate from the same 34 independent training specimens.
Target = FLM_Void_NoGF.
The bundle mask remains the ROI.
The SAME geometric transform was applied to SEM + Void mask + ROI.
Validation and test data were NOT augmented.
No model was trained by this script.
Existing GF dataset/model were NOT modified.
""")


if __name__ == "__main__":
    main()