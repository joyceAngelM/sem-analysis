from pathlib import Path
import csv
import random
from collections import Counter, defaultdict


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = PROJECT_ROOT / "outputs" / "mask_quantification.csv"

# NEW experiment output folder.
# This prevents mixing the new 70/15/15 experiment
# with the old 30/9/9 experiment.
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_70_15_15"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "sample_split_70_15_15.csv"

RANDOM_SEED = 42

TARGET_TRAIN = 34
TARGET_VALIDATION = 7
TARGET_TEST = 7


# ============================================================
# LOAD ORIGINAL SPECIMENS
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


print("=" * 75)
print("FINAL 70 / 15 / 15 SPECIMEN-LEVEL SPLIT")
print("=" * 75)

print(f"\nInput file:\n{INPUT_CSV}")

print(f"\nTotal original specimens found: {len(samples)}")


# ============================================================
# BASIC DATASET CHECKS
# ============================================================

if len(samples) != 48:
    raise ValueError(
        f"Expected exactly 48 original specimens, "
        f"but found {len(samples)}."
    )


sample_ids = [
    sample["sample_id"]
    for sample in samples
]


if len(sample_ids) != len(set(sample_ids)):
    raise ValueError(
        "Duplicate original specimen IDs detected."
    )


# ============================================================
# SHOW ORIGINAL DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("ORIGINAL DATASET DISTRIBUTION")
print("=" * 75)

for cr in ["CR_0", "CR_30", "CR_41"]:

    cr_samples = [
        s for s in samples
        if s["compaction_ratio"] == cr
    ]

    print(f"\n{cr}: {len(cr_samples)} specimens")

    for layer in [1, 3, 5]:

        count = sum(
            1
            for s in cr_samples
            if s["layer"] == layer
        )

        print(f"  Layer {layer}: {count}")


# ============================================================
# TARGET DISTRIBUTION
#
# Total:
# CR_0  = 15
# CR_30 = 16
# CR_41 = 17
#
# We need:
# Train = 34
# Val   = 7
# Test  = 7
#
# Target CR representation:
#
#             Train   Val   Test
# CR_0          11     2      2
# CR_30         11     2      3
# CR_41         12     3      2
#
# Totals         34     7      7
#
# This keeps every CR represented in every split.
# ============================================================

TARGETS = {

    "CR_0": {
        "train": 11,
        "validation": 2,
        "test": 2,
    },

    "CR_30": {
        "train": 11,
        "validation": 2,
        "test": 3,
    },

    "CR_41": {
        "train": 12,
        "validation": 3,
        "test": 2,
    },
}


# ============================================================
# STRATIFIED SPLIT BY CR
#
# We use a fixed random seed so the result is reproducible.
# Layer representation is checked afterwards.
# ============================================================

rng = random.Random(RANDOM_SEED)

split_rows = []

for cr in ["CR_0", "CR_30", "CR_41"]:

    cr_samples = [
        sample.copy()
        for sample in samples
        if sample["compaction_ratio"] == cr
    ]

    # Sort before shuffling so the result is reproducible
    # even if CSV row order changes.
    cr_samples.sort(
        key=lambda x: x["sample_id"]
    )

    rng.shuffle(cr_samples)

    n_val = TARGETS[cr]["validation"]
    n_test = TARGETS[cr]["test"]
    n_train = TARGETS[cr]["train"]

    validation_samples = cr_samples[:n_val]

    test_samples = cr_samples[
        n_val:n_val + n_test
    ]

    train_samples = cr_samples[
        n_val + n_test:
    ]

    if len(train_samples) != n_train:
        raise ValueError(
            f"Unexpected training count for {cr}: "
            f"{len(train_samples)} instead of {n_train}"
        )

    for sample in train_samples:
        split_rows.append({
            **sample,
            "split": "train"
        })

    for sample in validation_samples:
        split_rows.append({
            **sample,
            "split": "validation"
        })

    for sample in test_samples:
        split_rows.append({
            **sample,
            "split": "test"
        })


# ============================================================
# CREATE ID SETS
# ============================================================

train_ids = {
    row["sample_id"]
    for row in split_rows
    if row["split"] == "train"
}

validation_ids = {
    row["sample_id"]
    for row in split_rows
    if row["split"] == "validation"
}

test_ids = {
    row["sample_id"]
    for row in split_rows
    if row["split"] == "test"
}


# ============================================================
# DATA LEAKAGE CHECK
# ============================================================

train_val_overlap = train_ids & validation_ids
train_test_overlap = train_ids & test_ids
val_test_overlap = validation_ids & test_ids


print("\n" + "=" * 75)
print("DATA LEAKAGE CHECK")
print("=" * 75)

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


if train_val_overlap:
    raise ValueError(
        "DATA LEAKAGE: Train/validation overlap detected."
    )

if train_test_overlap:
    raise ValueError(
        "DATA LEAKAGE: Train/test overlap detected."
    )

if val_test_overlap:
    raise ValueError(
        "DATA LEAKAGE: Validation/test overlap detected."
    )


# ============================================================
# CHECK ALL SPECIMENS APPEAR EXACTLY ONCE
# ============================================================

all_split_ids = (
    train_ids
    | validation_ids
    | test_ids
)

original_ids = set(sample_ids)


if all_split_ids != original_ids:

    missing = original_ids - all_split_ids
    unexpected = all_split_ids - original_ids

    raise ValueError(
        f"Split mismatch.\n"
        f"Missing IDs: {sorted(missing)}\n"
        f"Unexpected IDs: {sorted(unexpected)}"
    )


if (
    len(train_ids) != TARGET_TRAIN
    or len(validation_ids) != TARGET_VALIDATION
    or len(test_ids) != TARGET_TEST
):

    raise ValueError(
        "Final split counts do not match 34/7/7."
    )


# ============================================================
# SORT FOR READABILITY
# ============================================================

split_order = {
    "train": 0,
    "validation": 1,
    "test": 2,
}

cr_order = {
    "CR_0": 0,
    "CR_30": 1,
    "CR_41": 2,
}

split_rows.sort(
    key=lambda x: (
        split_order[x["split"]],
        cr_order[x["compaction_ratio"]],
        x["layer"],
        x["sample_id"],
    )
)


# ============================================================
# SAVE NEW SPLIT CSV
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    fieldnames = [
        "sample_id",
        "compaction_ratio",
        "layer",
        "split",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(split_rows)


# ============================================================
# PRINT OVERALL SPLIT
# ============================================================

print("\n" + "=" * 75)
print("FINAL SPLIT COUNTS")
print("=" * 75)

split_counts = Counter(
    row["split"]
    for row in split_rows
)

for split in [
    "train",
    "validation",
    "test"
]:

    count = split_counts[split]

    percentage = (
        count / len(samples)
    ) * 100

    print(
        f"{split.upper():<12}: "
        f"{count:>2} specimens "
        f"({percentage:.2f}%)"
    )


# ============================================================
# PRINT CR DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("DISTRIBUTION BY COMPACTION RATIO")
print("=" * 75)

for split in [
    "train",
    "validation",
    "test"
]:

    print(f"\n{split.upper()}")

    for cr in [
        "CR_0",
        "CR_30",
        "CR_41"
    ]:

        count = sum(
            1
            for row in split_rows
            if row["split"] == split
            and row["compaction_ratio"] == cr
        )

        print(
            f"  {cr}: {count}"
        )


# ============================================================
# PRINT CR + LAYER DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("DISTRIBUTION BY CR + LAYER")
print("=" * 75)

for split in [
    "train",
    "validation",
    "test"
]:

    print(f"\n{split.upper()}")

    for cr in [
        "CR_0",
        "CR_30",
        "CR_41"
    ]:

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
# PRINT EVERY SPECIMEN ID
# ============================================================

print("\n" + "=" * 75)
print("SPECIMEN IDs")
print("=" * 75)

for split in [
    "train",
    "validation",
    "test"
]:

    rows = [
        row
        for row in split_rows
        if row["split"] == split
    ]

    print(
        f"\n{split.upper()} "
        f"({len(rows)} specimens)"
    )

    for row in rows:

        print(
            f"  {row['sample_id']:<12} "
            f"{row['compaction_ratio']:<6} "
            f"Layer {row['layer']}"
        )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("FINAL 70 / 15 / 15 SPLIT COMPLETE")
print("=" * 75)

print(
    f"\nSplit CSV saved to:\n{OUTPUT_CSV}"
)

print(
    "\nIMPORTANT:"
    "\n48 independent original specimens were split BEFORE "
    "patch extraction."
    "\nNo patches have been generated."
    "\nNo augmentation has been performed."
    "\nNo model has been trained."
)

print(
    "\nRequired final counts:"
    "\nTrain      = 34 / 48 = 70.83%"
    "\nValidation =  7 / 48 = 14.58%"
    "\nTest       =  7 / 48 = 14.58%"
)