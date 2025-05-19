# Lab book 

<!--toc:start-->
- [Lab book](#lab-book)
  - [Before](#before)
  - [2025-05-11](#2025-05-11)
  - [2025-05-12](#2025-05-12)
  - [2025-05-13](#2025-05-13)
  - [2025-05-15](#2025-05-15)
  - [2025-05-19](#2025-05-19)
<!--toc:end-->


## Before

To be able to replicate this project, one would need to install the block
decomposition module written by Martin Girard.

The project can be found here: [Words](https://gitlab.mpcdf.mpg.de/mgirard/Words)
**Important**: The dev branch needs to be installed.


## 2025-05-11

A python module was written. It is responsible for converting the original
output of the block decomposition algorithm from `list[tuple[int, int]]` to a
`np.ndarry` that also labels the blocks based on their main component.

The module can be found here:
`./src/modules/block_decomposition_modifier.py`

## 2025-05-12

A data set of proteins was downloaded.
It contains over 600 sequences of phase separating proteins and over 2000 
proteins of non phase separating proteins.
Source: [llpsdatasets](https://llpsdatasets.ppmclab.com)

The file `datasets.tsv` was downloaded and moved to
`./data/raw_data/llps_data_ppmclab.tsv`.

---

The Words module (containing the block decomposition algorithm) was copied into
the folder `./src/modules/` folder: `./src/modules/Words/`.

Additional mappings (beside the APNAMApping - Aliphatic, Positive, Negative,
Aromatic) were added to the file `./src/modules/Words/Mappings.py`:
- Amino Acids associated with IDRs (IDRMapping) [article](https://pmc.ncbi.nlm.nih.gov/articles/PMC2676888/)
- Most meaningful grouping with five groups (MM5Mapping) [article](https://www.academia.edu/14913388/Simplifying_amino_acid_alphabets_by_means_of_a_branch_and_bound_algorithm_and_substitution_matrices)
- PiPi interactions per group (PIPIGMapping) and per frequency (PIPIFMapping) [article](https://elifesciences.org/articles/31486)
- RG Motifs (RGMapping)

---

The script `./src/scripts/prepare_raw_data.py` was created and run with: 
```sh 
python src/scripts/prepare_raw_data.py
```
Creating:
`./data/intermediate_data/llps_data_ppmclab.pkl`

This script read the `.csv` file and filtered out sequences containing letters
that are not in the mapped amino acid alphabet (X and U). It also added one
column named `PS` that contains a `0` as negative label and a `1` as positive
label. It saved the data as a pickle.

---

The script `./src/scripts/run_block_decomposition.py` was created and run using:
```sh 
python src/scripts/run_block_decomposition.py
```
Creating:
`./data/intermediate_data/llps_data_ppmclab_bd.pkl`

This script ran the block decomposition algorithm with all mappings a

## 2025-05-13 

Added another Mapping to `./src/modules/Words/Mappings.py`:

- Mapping that shows enriched amino acids in proteins containing the RG-Motif
and take part in PS

---

The script `./src/scripts/run_block_decomposition.py` was run again with the new
mapping:
```sh 
python src/scripts/run_block_decomposition.py
```
Creating:
`./data/intermediate_data/llps_data_ppmclab_bd.pkl`

---

Created a file `./src/scripts/prepare_training_data.py` and started to implement
a model. The max length of a sequence was limited to 2700, as this is the length
of the largest possible sequence that can be analyzed with PSPire. This improved
the performance of the model slightly.
The model was run several times with varying parameters, just to get a feeling
if any of them change the result a lot. 

```sh
python ./src/scripts/prepare_training_data.py
```

The performance of the model was okay. It plateaued after just a few epochs.
AUROC was around 0.84 and PRAUC around 0.7. Only using the balance threshold of
4 instead of 3 and 4 did not impact the balance.

## 2025-05-15

Renamed the following files to keep things clean:
```sh 
mv data/raw_data/llps_data_ppmclab.tsv data/raw_data/ppmclab.tsv
mv data/raw_data/ps_pire_data.csv data/raw_data/pspire.csv
mv data/intermediate_data/llps_data_ppmclab.pkl data/intermediate_data/ppmclab.pkl
mv data/intermediate_data/llps_data_ppmclab_bd.pkl data/intermediate_data/ppmclab_bd.pkl
mv src/scripts/prepare_raw_data.py src/scripts/prepare_ppmclab.py
mv src/scripts/get_ps_pire_sequences.py src/scripts/prepare_pspire.py
mv src/scripts/run_block_decomposition.py src/scripts/run_bd_ppmclab.py
mv src/scripts/prepare_training_data.py src/scripts/run_cnn_ppmclab.py

```

## 2025-05-19

The data of the PS-Pire article was downloaded and saved as
`./data/raw_data/pspire.csv`. A script was written to download the sequences
for the proteins from UniProt and filter them as well as create some
visualizations and also map the sequences. (`./src/scripts/prepare_pspire.py`)
This script was run with: 

```sh 
python ./src/scripts/prepare_pspire.py
```

The script for preparing the ppmclab data set was also modified to yield some
graphics (`./src/scripts/prepare_ppmclab.py`).

```sh 
python ./src/scripts/prepare_ppmclab.py
```

Started to modularize code. Datasets and models will now get a script each that
resides in `./src/modules/`.
Following files have been created there:
`./src/modules/bd_cnn_1l.py`
`./src/modules/bd_sequence_dataset.py`
`./src/modules/sequence_dataset.py`
`./src/modules/mappings.py`

The module `./src/modules/block_decomposition_modifier.py` was renamed to
`./src/modules/bd_tools.py`.


