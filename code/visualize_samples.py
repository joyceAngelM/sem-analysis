from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================
# RQ2GEI SAMPLE VISUALIZATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "Multi_Modal_Microscopy_DataSet_paper"
    / "Multi_Modal_Microscopy_DataSet_paper"
)

BUNDLE_ROOT = DATASET_ROOT / "Bundles_0_degree_SEM_FLM"

# One representative sample from each compaction ratio
SAMPLES = {
    "CR_0": {
        "sample": "CR0_L1M2",
        "image_folder": "0deg_Bundles_Stack",
        "mask_folder": "0deg_Binary_Masks_Quantitative",
    },

    "CR_30": {
        "sample": "CR30_L1M2",
        "image_folder": "0deg_Bundles_Stack",
        "mask_folder": "0deg_Binary_Masks_Quantitative",
    },

    "CR_41": {
        "sample": "CR41_L1M2",
        "image_folder": "Bundles_Stack",
        "mask_folder": "Binary_Masks",
    },
}


def load_image(path):
    return Image.open(path)


for cr_name, config in SAMPLES.items():

    sample_name = config["sample"]

    cr_root = BUNDLE_ROOT / cr_name

    original_path = (
        cr_root
        / config["image_folder"]
        / f"{sample_name}.tif"
    )

    mask_root = (
        cr_root
        / config["mask_folder"]
        / sample_name
    )

    bundle_mask_path = mask_root / "SEM_Bundle_Mask.tif"
    gf_mask_path = mask_root / "SEM_GF_Mask.tif"
    void_mask_path = mask_root / "FLM_Void_NoGF.tif"

    # Handle the special filename if encountered
    if not gf_mask_path.exists():
        alternative = mask_root / "SEM_GF_Green_Large.tif"

        if alternative.exists():
            gf_mask_path = alternative

    print("\n" + "=" * 60)
    print(f"Sample: {sample_name}")
    print("=" * 60)

    paths = {
        "Original Bundle": original_path,
        "SEM Bundle Mask": bundle_mask_path,
        "SEM GF Mask": gf_mask_path,
        "FLM Void Mask": void_mask_path,
    }

    # Verify files
    missing = False

    for label, path in paths.items():
        print(f"{label}: {path.name}")

        if not path.exists():
            print("   ERROR: File not found!")
            missing = True

    if missing:
        print("Skipping this sample because a file is missing.")
        continue

    # Load images
    original = load_image(original_path)
    bundle_mask = load_image(bundle_mask_path)
    gf_mask = load_image(gf_mask_path)
    void_mask = load_image(void_mask_path)

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))

    axes[0, 0].imshow(original)
    axes[0, 0].set_title(
        f"{sample_name} - Original Bundle"
    )

    axes[0, 1].imshow(bundle_mask, cmap="gray")
    axes[0, 1].set_title("SEM Bundle Mask")

    axes[1, 0].imshow(gf_mask, cmap="gray")
    axes[1, 0].set_title("SEM Glass Fibre Mask")

    axes[1, 1].imshow(void_mask, cmap="gray")
    axes[1, 1].set_title("FLM Void / No-GF Mask")

    for ax in axes.flat:
        ax.axis("off")

    plt.suptitle(
        f"RQ2GEI - {cr_name} - {sample_name}",
        fontsize=16
    )

    plt.tight_layout()

    # Save a copy so we can inspect it later
    output_dir = PROJECT_ROOT / "outputs" / "sample_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{sample_name}_visualization.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(f"Saved visualization: {output_path}")

    plt.show()

    # Close before moving to next sample
    plt.close(fig)


print("\nVisualization complete.")