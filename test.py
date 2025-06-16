import pickle
import matplotlib.pyplot as plt
from numpy import arange

with open("./data/intermediate_data/ppmclab_rsa.pkl", "rb") as f:
    df = pickle.load(f)

test = df.loc[df["UniProt.Acc"] == "P08047", "rsa"].values[0]

plt.plot(list(range(len(test))), test)
