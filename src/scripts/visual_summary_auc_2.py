import matplotlib.pyplot as plt
import numpy as np

models = (
    "cnn2l",
    "cnn2l_rsa",
    "cnn2l_linear",
    "cnn2l_weighted",
    "cnn2l_bn",
    "cnn2l_weighted_bn",
    "cnn2l_weighted_bn sep",
    "PdPS",
    "PSPire",
)

aucs = {
    "ROCAUC IDR": (0.74, 0.73, 0.75, 0.76, 0.79, 0.74, 0.82, 0.84, 0.86),
    "ROCAUC nIDR": (0.64, 0.66, 0.64, 0.70, 0.66, 0.75, 0.86, 0.68, 0.84),
    "PRAUC IDR": (0.26, 0.29, 0.34, 0.38, 0.37, 0.33, 0.42, 0.42, 0.51),
    "PRAUC nIDR": (0.05, 0.05, 0.06, 0.06, 0.08, 0.13, 0.17, 0.08, 0.24),
}

x = np.arange(len(models))
width = 0.2
multiplier = 1

fig, ax = plt.subplots(figsize=(20, 5), layout="constrained")

for attribute, measurement in aucs.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xticks(x + width, models)
ax.legend(loc="upper left", ncols=1)
ax.set_ylim(0, 1)

plt.savefig("./results/plots/visual_summary_auc_pspire_later_models")
plt.close()
