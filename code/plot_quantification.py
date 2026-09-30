from pathlib import Path
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "mask_quantification.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "quantification_plots"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD CSV
# ============================================================

rows = []

with open(CSV_PATH, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        rows.append({
            "sample_id": row["sample_id"],
            "cr": row["compaction_ratio"],
            "layer": int(row["layer"]),
            "gf": float(row["glass_fibre_percent"]),
            "void": float(row["void_noGF_percent"]),
        })

print(f"Loaded {len(rows)} samples.")


CR_ORDER = ["CR_0", "CR_30", "CR_41"]
LAYERS = [1, 3, 5]


# ============================================================
# GRAPH 1
# GLASS FIBRE % BY COMPACTION RATIO
# ============================================================

gf_by_cr = []

for cr in CR_ORDER:
    values = [
        row["gf"]
        for row in rows
        if row["cr"] == cr
    ]

    gf_by_cr.append(values)


plt.figure(figsize=(8, 6))

plt.boxplot(
    gf_by_cr,
    tick_labels=["CR0", "CR30", "CR41"]
)

plt.ylabel("Glass Fibre Area (%)")
plt.xlabel("Compaction Ratio")
plt.title("Glass Fibre Area by Compaction Ratio")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

path = OUTPUT_DIR / "01_glass_fibre_by_CR.png"

plt.savefig(path, dpi=200)
plt.close()

print(f"Saved: {path.name}")


# ============================================================
# GRAPH 2
# VOID/NoGF % BY COMPACTION RATIO
# ============================================================

void_by_cr = []

for cr in CR_ORDER:
    values = [
        row["void"]
        for row in rows
        if row["cr"] == cr
    ]

    void_by_cr.append(values)


plt.figure(figsize=(8, 6))

plt.boxplot(
    void_by_cr,
    tick_labels=["CR0", "CR30", "CR41"]
)

plt.ylabel("Void / No-GF Area (%)")
plt.xlabel("Compaction Ratio")
plt.title("Void / No-GF Area by Compaction Ratio")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

path = OUTPUT_DIR / "02_void_by_CR.png"

plt.savefig(path, dpi=200)
plt.close()

print(f"Saved: {path.name}")


# ============================================================
# GRAPH 3
# GLASS FIBRE % BY CR AND LAYER
# ============================================================

gf_grouped = defaultdict(list)

for row in rows:
    gf_grouped[(row["cr"], row["layer"])].append(
        row["gf"]
    )


x = np.arange(len(LAYERS))
width = 0.25


plt.figure(figsize=(10, 6))

for i, cr in enumerate(CR_ORDER):

    means = []

    for layer in LAYERS:

        values = gf_grouped[(cr, layer)]

        if values:
            means.append(np.mean(values))
        else:
            means.append(np.nan)

    plt.bar(
        x + (i - 1) * width,
        means,
        width,
        label=cr
    )


plt.xticks(x, ["Layer 1", "Layer 3", "Layer 5"])

plt.xlabel("Through-Thickness Layer")
plt.ylabel("Mean Glass Fibre Area (%)")

plt.title(
    "Mean Glass Fibre Area by "
    "Compaction Ratio and Layer"
)

plt.legend()

plt.tight_layout()

path = OUTPUT_DIR / "03_glass_fibre_CR_layer.png"

plt.savefig(path, dpi=200)
plt.close()

print(f"Saved: {path.name}")


# ============================================================
# GRAPH 4
# VOID/NoGF % BY CR AND LAYER
# ============================================================

void_grouped = defaultdict(list)

for row in rows:
    void_grouped[(row["cr"], row["layer"])].append(
        row["void"]
    )


plt.figure(figsize=(10, 6))

for i, cr in enumerate(CR_ORDER):

    means = []

    for layer in LAYERS:

        values = void_grouped[(cr, layer)]

        if values:
            means.append(np.mean(values))
        else:
            means.append(np.nan)

    plt.bar(
        x + (i - 1) * width,
        means,
        width,
        label=cr
    )


plt.xticks(x, ["Layer 1", "Layer 3", "Layer 5"])

plt.xlabel("Through-Thickness Layer")
plt.ylabel("Mean Void / No-GF Area (%)")

plt.title(
    "Mean Void / No-GF Area by "
    "Compaction Ratio and Layer"
)

plt.legend()

plt.tight_layout()

path = OUTPUT_DIR / "04_void_CR_layer.png"

plt.savefig(path, dpi=200)
plt.close()

print(f"Saved: {path.name}")


# ============================================================
# GRAPH 5
# GF % vs VOID/NoGF %
# ============================================================

plt.figure(figsize=(8, 6))

for cr in CR_ORDER:

    group = [
        row for row in rows
        if row["cr"] == cr
    ]

    x_values = [
        row["gf"]
        for row in group
    ]

    y_values = [
        row["void"]
        for row in group
    ]

    plt.scatter(
        x_values,
        y_values,
        label=cr,
        s=60
    )


plt.xlabel("Glass Fibre Area (%)")
plt.ylabel("Void / No-GF Area (%)")

plt.title(
    "Relationship Between Glass Fibre "
    "and Void / No-GF Area"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

path = OUTPUT_DIR / "05_GF_vs_void.png"

plt.savefig(path, dpi=200)
plt.close()

print(f"Saved: {path.name}")


# ============================================================
# NUMERICAL LAYER SUMMARY
# ============================================================

print("\n" + "=" * 72)
print("MEAN VALUES BY COMPACTION RATIO AND LAYER")
print("=" * 72)


for cr in CR_ORDER:

    print(f"\n{cr}")

    for layer in LAYERS:

        group = [
            row
            for row in rows
            if row["cr"] == cr
            and row["layer"] == layer
        ]

        if not group:
            continue

        gf_values = [
            row["gf"]
            for row in group
        ]

        void_values = [
            row["void"]
            for row in group
        ]

        print(
            f"Layer {layer}: "
            f"n={len(group)} | "
            f"GF={np.mean(gf_values):.2f}% | "
            f"Void/NoGF={np.mean(void_values):.2f}%"
        )


print("\n" + "=" * 72)
print("PLOTTING COMPLETE")
print("=" * 72)

print(f"\nPlots saved in:\n{OUTPUT_DIR}")