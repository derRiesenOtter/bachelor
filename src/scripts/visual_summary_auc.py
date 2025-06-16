import matplotlib.pyplot as plt
import numpy as np

models = (
    "cnn1l_bd",
    "cnn2l_bd",
    "xgb_bd",
    "cnn1l",
    "cnn2l",
    "cnn3l",
    "blstm",
    "transformer",
    "cnn2l_blstm",
    "PdPS",
    "PSPire",
)

aucs = {
    "ROCAUC IDR": (0.50, 0.70, 0.74, 0.70, 0.74, 0.73, 0.78, 0.80, 0.75, 0.84, 0.86),
    "ROCAUC nIDR": (0.61, 0.62, 0.54, 0.64, 0.64, 0.66, 0.49, 0.55, 0.63, 0.68, 0.84),
    "PRAUC IDR": (0.09, 0.22, 0.25, 0.20, 0.26, 0.29, 0.25, 0.29, 0.24, 0.42, 0.51),
    "PRAUC nIDR": (0.06, 0.05, 0.03, 0.06, 0.05, 0.05, 0.03, 0.03, 0.06, 0.08, 0.24),
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

plt.savefig("./results/plots/visual_summary_auc_pspire_first_models")
plt.close()
