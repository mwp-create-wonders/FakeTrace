import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Data: attribution distribution (%)
# -----------------------------
seen_generators = [
    "BigGAN", "CycleGAN", "DALLE", "Flux", "Glide",
    "LatentDM", "Midjourney", "SD1.4", "SD1.5"
]

data = {
    "ADM":    [3.2, 0.7, 5.8, 8.6, 43.5, 20.4, 3.6, 7.1, 7.1],
    "VQDM":   [1.8, 0.6, 26.7, 5.4, 13.8, 32.6, 4.3, 6.9, 7.9],
    "Wukong": [0.4, 0.3, 1.8, 3.7, 2.6, 7.9, 7.2, 36.8, 39.3],
}

# -----------------------------
# Style settings
# -----------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.0

fig, axes = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(12.5, 3),
    sharey=True
)

y = np.arange(len(seen_generators))

for ax, (unseen_name, values) in zip(axes, data.items()):
    values = np.array(values)

    bars = ax.barh(
        y,
        values,
        height=0.62,
        edgecolor="black",
        linewidth=0.6
    )

    # Mark the top-2 attributed seen generators
    top2_idx = np.argsort(values)[-2:]
    for idx in top2_idx:
        bars[idx].set_hatch("//")

    # Add value labels
    for i, v in enumerate(values):
        ax.text(
            v + 0.8,
            i,
            f"{v:.1f}",
            va="center",
            ha="left",
            fontsize=10
        )

    ax.set_title(f"Unseen: {unseen_name}", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 50)
    ax.set_xlabel("Attribution rate (%)")
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.invert_yaxis()

axes[0].set_yticks(y)
axes[0].set_yticklabels(seen_generators)
axes[0].set_ylabel("Predicted seen generator")

fig.suptitle(
    "Open-set Attribution Distribution for Unseen Generators",
    fontsize=15,
    fontweight="bold",
    y=1.03
)

fig.text(
    0.5,
    -0.02,
    "Each subfigure reports the prediction distribution of an unseen generator over seen generator classes. "
    "Hatched bars indicate the top-2 attributed seen generators.",
    ha="center",
    fontsize=10
)

plt.tight_layout()

# Save figures
plt.savefig("open_set_attribution_bar.png", dpi=600, bbox_inches="tight")
# plt.savefig("open_set_attribution_bar.pdf", bbox_inches="tight")

plt.show()