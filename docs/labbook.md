# Lab book 

<!--toc:start-->
- [Lab book](#lab-book)
  - [Before](#before)
  - [2025-05-11](#2025-05-11)
  - [2025-05-12](#2025-05-12)
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

Created a file `./src/scripts/prepare_raw_data.py` and started to implement a
1dcnn
