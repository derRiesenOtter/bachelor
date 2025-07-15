import pickle

import matplotlib.pyplot as plt
import seaborn as sns

with open("./data/intermediate_data/ppmclab.pkl", "rb") as f:
    ppmclab = pickle.load(f)

with open("./data/intermediate_data/pspire.pkl", "rb") as f:
    pspire = pickle.load(f)

with open("./data/intermediate_data/catgranule2.pkl", "rb") as f:
    catgranule2 = pickle.load(f)

catgranule2 = catgranule2[(catgranule2["test"] == 1) | (catgranule2["training"] == 1)]

ppmclab_seq_lengths = ppmclab["Full.seq"].str.len()
pspire_seq_lengths = pspire["seq"].str.len()
catgranule2_seq_lengths = catgranule2["seq"].str.len()

bins = range(0, 3000, 250)
plt.hist(
    pspire_seq_lengths,
    label="PSPire",
    color="orange",
    bins=bins,
    histtype="step",
    linewidth=2,
)
plt.hist(
    catgranule2_seq_lengths,
    label="catGranule2.0",
    color="green",
    bins=bins,
    histtype="step",
    linewidth=2,
)
plt.hist(ppmclab_seq_lengths, label="PPMC-lab", bins=bins, histtype="step", linewidth=2)
plt.legend()
plt.xlim(left=0)
plt.xlabel("Sequence Length")
plt.ylabel("Number of Sequences")

plt.savefig("./results/plots/data_preparation_length_distribution.png")
