from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "Multi_Modal_Microscopy_DataSet_paper"
    / "Multi_Modal_Microscopy_DataSet_paper"
)

BUNDLE_ROOT = DATASET_ROOT / "Bundles_0_degree_SEM_FLM"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mask_alignment_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SPECIMENS TO CHECK
#
# We deliberately inspect samples from CR0, CR30 and CR41.
# CR0_L1M1 is included because it has the special
# SEM_GF_Green_Large.tif filename.
# ============================================================

SAMPLES = {
    "CR_0": [
        "CR0_L1M1",
        "CR0_L3M3",
        "CR0_L5M4",
    ],

    "CR_30": [
        "CR30_L1M1",
        "CR30_L3M3",
        "CR30_L5M5",
    ],

    "CR_41": [
        "CR41_L1M1",
        "CR41_L3M3",
        "CR41_L5M5",
    ],
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_sample_file(folder, sample_id):
    """
    Find the original SEM/bundle image corresponding
    to a specimen ID.
    """

    candidates = []

    for path in folder.iterdir():

        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".tif", ".tiff", ".png", ".jpg", ".jpeg"
        }:
            continue

        stem_lower = path.stem.lower()
        sample_lower = sample_id.lower()

        if (
            stem_lower == sample_lower
            or sample_lower in stem_lower
        ):
            candidates.append(path)

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No original SEM image found for {sample_id} in:\n{folder}"
        )

    if len(candidates) > 1:
        print(f"\nWARNING: Multiple image candidates for {sample_id}:")
        for p in candidates:
            print("   ", p.name)

    return sorted(candidates)[0]


def load_rgb(path):
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def load_mask(path):
    with Image.open(path) as img:
        arr = np.array(img.convert("L"))

    # Convert 0/255 mask to boolean.
    return arr > 0


def mask_overlay(image, mask):
    """
    Create a simple red overlay for visual alignment checking.
    """

    image = image.astype(np.float32)

    overlay = image.copy()

    red = np.zeros_like(overlay)
    red[:, :, 0] = 255

    alpha = 0.40

    overlay[mask] = (
        (1 - alpha) * overlay[mask]
        + alpha * red[mask]
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def get_condition_folders(condition):

    condition_root = BUNDLE_ROOT / condition

    if condition in {"CR_0", "CR_30"}:

        image_dir = (
            condition_root
            / "0deg_Bundles_Stack"
        )

        mask_dir = (
            condition_root
            / "0deg_Binary_Masks_Quantitative"
        )

    elif condition == "CR_41":

        image_dir = (
            condition_root
            / "Bundles_Stack"
        )

        mask_dir = (
            condition_root
            / "Binary_Masks"
        )

    else:
        raise ValueError(
            f"Unknown condition: {condition}"
        )

    return image_dir, mask_dir


def locate_gf_mask(sample_mask_dir):

    normal = sample_mask_dir / "SEM_GF_Mask.tif"

    special = (
        sample_mask_dir
        / "SEM_GF_Green_Large.tif"
    )

    if normal.exists():
        return normal

    if special.exists():
        return special

    raise FileNotFoundError(
        f"No GF mask found in:\n{sample_mask_dir}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("STEP 2 - MASK ALIGNMENT VERIFICATION")
    print("=" * 80)

    print("\nDataset root:")
    print(DATASET_ROOT)

    if not BUNDLE_ROOT.exists():
        raise FileNotFoundError(
            f"\nBundle root not found:\n{BUNDLE_ROOT}"
        )

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    total_created = 0

    for condition, sample_ids in SAMPLES.items():

        print("\n" + "=" * 80)
        print(condition)
        print("=" * 80)

        image_dir, mask_root = get_condition_folders(condition)

        print("\nImage directory:")
        print(image_dir)

        print("\nMask directory:")
        print(mask_root)

        for sample_id in sample_ids:

            print("\n--------------------------------------------------")
            print(f"Checking: {sample_id}")
            print("--------------------------------------------------")

            image_path = find_sample_file(
                image_dir,
                sample_id
            )

            sample_mask_dir = (
                mask_root
                / sample_id
            )

            if not sample_mask_dir.exists():
                raise FileNotFoundError(
                    f"Mask folder not found:\n{sample_mask_dir}"
                )

            gf_path = locate_gf_mask(
                sample_mask_dir
            )

            bundle_path = (
                sample_mask_dir
                / "SEM_Bundle_Mask.tif"
            )

            void_path = (
                sample_mask_dir
                / "FLM_Void_NoGF.tif"
            )

            required = [
                bundle_path,
                void_path,
            ]

            for path in required:
                if not path.exists():
                    raise FileNotFoundError(
                        f"Required mask missing:\n{path}"
                    )

            # --------------------------------------------
            # Load files
            # --------------------------------------------

            sem = load_rgb(image_path)

            gf = load_mask(gf_path)

            bundle = load_mask(bundle_path)

            void = load_mask(void_path)

            print(f"SEM file    : {image_path.name}")
            print(f"GF file     : {gf_path.name}")
            print(f"Bundle file : {bundle_path.name}")
            print(f"Void file   : {void_path.name}")

            print(f"\nSEM shape    : {sem.shape}")
            print(f"GF shape     : {gf.shape}")
            print(f"Bundle shape : {bundle.shape}")
            print(f"Void shape   : {void.shape}")

            # --------------------------------------------
            # Exact dimension check
            # --------------------------------------------

            sem_hw = sem.shape[:2]

            dimension_match = (
                gf.shape == sem_hw
                and bundle.shape == sem_hw
                and void.shape == sem_hw
            )

            print(
                "All dimensions match SEM:",
                dimension_match
            )

            if not dimension_match:

                print(
                    "\nWARNING:"
                    " Dimensions do not match."
                )

                print(
                    "No resizing will be performed."
                )

                print(
                    "This sample will NOT be overlaid."
                )

                continue

            # --------------------------------------------
            # Pixel statistics
            # --------------------------------------------

            total_pixels = sem_hw[0] * sem_hw[1]

            gf_pixels = int(gf.sum())
            bundle_pixels = int(bundle.sum())
            void_pixels = int(void.sum())

            print(
                f"GF positive pixels     : {gf_pixels}"
            )

            print(
                f"Bundle positive pixels : {bundle_pixels}"
            )

            print(
                f"Void positive pixels   : {void_pixels}"
            )

            print(
                f"GF image fraction      : "
                f"{100 * gf_pixels / total_pixels:.3f}%"
            )

            print(
                f"Void image fraction    : "
                f"{100 * void_pixels / total_pixels:.3f}%"
            )

            # --------------------------------------------
            # Create overlays
            # --------------------------------------------

            gf_overlay = mask_overlay(
                sem,
                gf
            )

            bundle_overlay = mask_overlay(
                sem,
                bundle
            )

            void_overlay = mask_overlay(
                sem,
                void
            )

            # --------------------------------------------
            # Visualization
            # --------------------------------------------

            fig, axes = plt.subplots(
                2,
                4,
                figsize=(20, 9)
            )

            # Row 1
            axes[0, 0].imshow(sem)
            axes[0, 0].set_title(
                "Original SEM"
            )

            axes[0, 1].imshow(
                gf,
                cmap="gray"
            )
            axes[0, 1].set_title(
                "GF Mask"
            )

            axes[0, 2].imshow(
                gf_overlay
            )
            axes[0, 2].set_title(
                "SEM + GF Overlay"
            )

            axes[0, 3].imshow(
                bundle,
                cmap="gray"
            )
            axes[0, 3].set_title(
                "Bundle Mask"
            )

            # Row 2
            axes[1, 0].imshow(
                bundle_overlay
            )
            axes[1, 0].set_title(
                "SEM + Bundle Overlay"
            )

            axes[1, 1].imshow(
                void,
                cmap="gray"
            )
            axes[1, 1].set_title(
                "FLM Void / No-GF"
            )

            axes[1, 2].imshow(
                void_overlay
            )
            axes[1, 2].set_title(
                "SEM + Void Overlay"
            )

            # Combined GF + void
            combined = sem.astype(
                np.float32
            ).copy()

            gf_colour = np.zeros_like(
                combined
            )
            gf_colour[:, :, 1] = 255

            void_colour = np.zeros_like(
                combined
            )
            void_colour[:, :, 0] = 255

            alpha = 0.40

            combined[gf] = (
                (1 - alpha) * combined[gf]
                + alpha * gf_colour[gf]
            )

            combined[void] = (
                (1 - alpha) * combined[void]
                + alpha * void_colour[void]
            )

            combined = np.clip(
                combined,
                0,
                255
            ).astype(np.uint8)

            axes[1, 3].imshow(
                combined
            )

            axes[1, 3].set_title(
                "Combined: GF=Green, Void=Red"
            )

            for ax in axes.ravel():
                ax.axis("off")

            fig.suptitle(
                f"{condition} - {sample_id}",
                fontsize=16
            )

            plt.tight_layout()

            output_path = (
                OUTPUT_DIR
                / f"{sample_id}_alignment.png"
            )

            plt.savefig(
                output_path,
                dpi=150,
                bbox_inches="tight"
            )

            plt.close(fig)

            total_created += 1

            print(
                f"\nSaved visualization:\n{output_path}"
            )

    print("\n" + "=" * 80)
    print("ALIGNMENT VISUALIZATION COMPLETE")
    print("=" * 80)

    print(
        f"\nVisualizations created: {total_created}"
    )

    print(
        "\nIMPORTANT:"
        "\nThese figures are for manual spatial verification."
        "\nMatching dimensions alone are not proof of alignment."
    )


if __name__ == "__main__":
    main()