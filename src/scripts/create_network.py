import pickle

import networkx as nx
import pandas as pd
from node2vec import Node2Vec

df = pd.read_csv("./data/raw_data/BIOGRID-ALL-4.4.246.tab3.txt", sep="\t")

df = df[df["Experimental System Type"] == "physical"]
df[(df["Organism ID Interactor A"] == 9606) & (df["Organism ID Interactor B"] == 9606)]

G = nx.Graph()

for _, row in df.iterrows():
    protein_a = row['SWISS-PROT Accessions Interactor A']
    protein_b = row['SWISS-PROT Accessions Interactor B']
    G.add_edge(protein_a, protein_b)

node2vec = Node2Vec(G, dimensions=64, walk_length=20, num_walks=100, workers=4)
model = node2vec.fit(window=5, min_count=1, batch_words=4)

with open("./data/intermediate_data/network.pkl", "wb") as f:
    pickle.dump(model, f)
