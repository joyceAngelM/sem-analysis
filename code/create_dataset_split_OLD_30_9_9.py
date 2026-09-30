from pathlib import Path
import csv
import random
from collections import defaultdict, Counter


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = PROJECT_ROOT / "outputs" / "mask_quantification.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dataset_split"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "sample_split.csv"

# Fixed seed = same split every time
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ============================================================
# LOAD THE 48 ORIGINAL SAMPLES
# ============================================================

samples = []

with open(INPUT_CSV, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        samples.append({
            "sample_id": row["sample_id"],
            "compaction_ratio": row["compaction_ratio"],
            "layer": int(row["layer"]),
        })


print("=" * 70)
print("ORIGINAL SAMPLE-LEVEL DATASET SPLIT")
print("=" * 70)

print(f"\nTotal original samples found: {len(samples)}")

if len(samples) != 48:
    print(
        "\nWARNING: Expected 48 original samples, "
        f"but found {len(samples)}."
    )


# ============================================================
# GROUP BY CR + LAYER
#
# Example groups:
# CR_0 + Layer 1
# CR_0 + Layer 3
# CR_0 + Layer 5
# CR_30 + Layer 1
# ...
#
# This helps preserve both experimental condition
# and through-thickness layer.
# ============================================================

groups = defaultdict(list)

for sample in samples:
    key = (
        sample["compaction_ratio"],
        sample["layer"]
    )

    groups[key].append(sample)


# ============================================================
# SPLIT EACH SMALL GROUP
#
# Most groups contain 4-6 samples.
#
# For each CR + Layer group:
#   1 sample -> validation
#   1 sample -> test
#   remaining -> train
#
# This guarantees representation of every CR/layer
# combination in validation and test.
# ============================================================

split_rows = []

for key in sorted(groups.keys()):

    group = groups[key].copy()

    random.shuffle(group)

    cr, layer = key

    if len(group) < 3:
        raise ValueError(
            f"Not enough samples in {cr}, Layer {layer}"
        )

    validation_sample = group[0]
    test_sample = group[1]
    train_samples = group[2:]

    # Validation
    split_rows.append({
        **validation_sample,
        "split": "validation"
    })

    # Test
    split_rows.append({
        **test_sample,
        "split": "test"
    })

    # Train
    for sample in train_samples:
        split_rows.append({
            **sample,
            "split": "train"
        })


# ============================================================
# SORT FOR READABILITY
# ============================================================

cr_order = {
    "CR_0": 0,
    "CR_30": 1,
    "CR_41": 2
}

split_order = {
    "train": 0,
    "validation": 1,
    "test": 2
}

split_rows.sort(
    key=lambda x: (
        split_order[x["split"]],
        cr_order[x["compaction_ratio"]],
        x["layer"],
        x["sample_id"]
    )
)


# ============================================================
# SAVE CSV
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "sample_id",
        "compaction_ratio",
        "layer",
        "split"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(split_rows)


# ============================================================
# VALIDATION CHECKS
# ============================================================

sample_ids = [
    row["sample_id"]
    for row in split_rows
]

if len(sample_ids) != len(set(sample_ids)):
    raise ValueError(
        "ERROR: Duplicate samples detected across splits!"
    )

if len(sample_ids) != len(samples):
    raise ValueError(
        "ERROR: Some samples are missing!"
    )


# ============================================================
# PRINT OVERALL SPLIT
# ============================================================

print("\n" + "=" * 70)
print("OVERALL SPLIT")
print("=" * 70)

split_counts = Counter(
    row["split"]
    for row in split_rows
)

for split in ["train", "validation", "test"]:

    count = split_counts[split]

    percentage = (
        count / len(split_rows)
    ) * 100

    print(
        f"{split.capitalize():<12}: "
        f"{count:>2} samples "
        f"({percentage:.1f}%)"
    )


# ============================================================
# PRINT DISTRIBUTION BY CR
# ============================================================

print("\n" + "=" * 70)
print("DISTRIBUTION BY COMPACTION RATIO")
print("=" * 70)

for split in ["train", "validation", "test"]:

    print(f"\n{split.upper()}")

    for cr in ["CR_0", "CR_30", "CR_41"]:

        count = sum(
            1
            for row in split_rows
            if row["split"] == split
            and row["compaction_ratio"] == cr
        )

        print(f"  {cr}: {count}")


# ============================================================
# PRINT DISTRIBUTION BY CR + LAYER
# ============================================================

print("\n" + "=" * 70)
print("DISTRIBUTION BY CR + LAYER")
print("=" * 70)

for split in ["train", "validation", "test"]:

    print(f"\n{split.upper()}")

    for cr in ["CR_0", "CR_30", "CR_41"]:

        for layer in [1, 3, 5]:

            count = sum(
                1
                for row in split_rows
                if row["split"] == split
                and row["compaction_ratio"] == cr
                and row["layer"] == layer
            )

            print(
                f"  {cr} | Layer {layer}: {count}"
            )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("SPLIT COMPLETE")
print("=" * 70)

print(f"\nSplit file saved to:\n{OUTPUT_CSV}")

print("""
IMPORTANT:
The original images have NOT been moved or modified.

From this point onward, every patch generated from an original
sample MUST inherit that original sample's split.

A patch from a training sample can only be used for training.
A patch from a validation sample can only be used for validation.
A patch from a test sample can only be used for final testing.
""")