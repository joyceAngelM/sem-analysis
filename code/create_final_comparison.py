from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# FINAL VERIFIED UNTOUCHED TEST RESULTS
# Same 179 test patches / same 7 independent specimens
# ---------------------------------------------------------

results = {
    "Metric": [
        "Dice",
        "IoU",
        "Precision",
        "Recall",
        "Pixel Accuracy"
    ],
    "U-Net": [
        93.5440,
        87.8711,
        92.4813,
        94.6315,
        93.0294
    ],
    "SVM": [
        89.9384,
        81.7165,
        89.9978,
        89.8792,
        89.2683
    ]
}

df = pd.DataFrame(results)

output_dir = Path(
    "outputs/final_70_15_15/final_comparison"
)
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# SAVE FINAL TABLE
# ---------------------------------------------------------

csv_path = output_dir / "unet_vs_svm_final_metrics.csv"
df.to_csv(csv_path, index=False)

print("=" * 70)
print("FINAL U-NET VS SVM COMPARISON")
print("=" * 70)
print()
print(df.to_string(index=False))
print()

# ---------------------------------------------------------
# BAR GRAPH
# ---------------------------------------------------------

metrics = df["Metric"]
unet = df["U-Net"]
svm = df["SVM"]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))

bars1 = ax.bar(
    x - width/2,
    unet,
    width,
    label="U-Net"
)

bars2 = ax.bar(
    x + width/2,
    svm,
    width,
    label="SVM"
)

ax.set_ylabel("Performance (%)")
ax.set_xlabel("Evaluation Metric")
ax.set_title(
    "U-Net vs SVM on Untouched Test Set"
)

ax.set_xticks(x)
ax.set_xticklabels(metrics)

ax.set_ylim(75, 100)
ax.legend()

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

# Add values above bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.25,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.tight_layout()

graph_path = output_dir / "unet_vs_svm_final_comparison.png"

plt.savefig(
    graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------------------------
# GF AREA-FRACTION ERROR GRAPH
# ---------------------------------------------------------

models = ["U-Net", "SVM"]
gf_errors = [1.2408, 0.0703]

fig, ax = plt.subplots(figsize=(7, 5))

bars = ax.bar(models, gf_errors)

ax.set_ylabel("Absolute Error (percentage points)")
ax.set_title("GF Area-Fraction Error on Untouched Test Set")
ax.grid(axis="y", linestyle="--", alpha=0.4)

for bar, value in zip(bars, gf_errors):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        value + 0.03,
        f"{value:.4f} pp",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

error_graph_path = (
    output_dir / "unet_vs_svm_gf_area_error.png"
)

plt.savefig(
    error_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------------------------
# SUMMARY TEXT
# ---------------------------------------------------------

summary_path = output_dir / "final_comparison_summary.txt"

with open(summary_path, "w", encoding="utf-8") as f:

    f.write("FINAL U-NET VS SVM COMPARISON\n")
    f.write("=" * 60 + "\n\n")

    f.write("Evaluation dataset:\n")
    f.write("Untouched test patches: 179\n")
    f.write("Independent test specimens: 7\n\n")

    f.write("U-Net:\n")
    f.write("Dice: 93.54%\n")
    f.write("IoU: 87.87%\n")
    f.write("Precision: 92.48%\n")
    f.write("Recall: 94.63%\n")
    f.write("Pixel Accuracy: 93.03%\n")
    f.write("GF area-fraction error: 1.2408 percentage points\n\n")

    f.write("SVM:\n")
    f.write("Dice: 89.94%\n")
    f.write("IoU: 81.72%\n")
    f.write("Precision: 90.00%\n")
    f.write("Recall: 89.88%\n")
    f.write("Pixel Accuracy: 89.27%\n")
    f.write("GF area-fraction error: 0.0703 percentage points\n\n")

    f.write(
        "Both models were evaluated on the same untouched "
        "test set and same bundle ROI.\n"
    )

print("Saved:")
print(csv_path)
print(graph_path)
print(error_graph_path)
print(summary_path)

print()
print("=" * 70)
print("FINAL COMPARISON FILES CREATED SUCCESSFULLY")
print("=" * 70)