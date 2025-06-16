# Lab book 

<!--toc:start-->
- [Lab book](#lab-book)
  - [Before](#before)
  - [2025-05-11](#2025-05-11)
  - [2025-05-12](#2025-05-12)
  - [2025-05-13](#2025-05-13)
  - [2025-05-15](#2025-05-15)
  - [2025-05-19](#2025-05-19)
  - [2025-05-20](#2025-05-20)
  - [2025-05-21](#2025-05-21)
  - [2025-05-22](#2025-05-22)
  - [2025-05-23](#2025-05-23)
  - [2025-05-26 to 2025-05-28](#2025-05-26-to-2025-05-28)
  - [2025-06-02](#2025-06-02)
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

## 2025-05-20

Created the block decomposition for the pspire data set with:
```sh 
python ./src/scripts/run_bd_pspire.py
```

Continued to modularize and rewrite the model. Today the focus was on improving
the training loop and creating additional graphics.

The following was run:
```sh 
python ./src/scripts/run_cnn1l_bd_ppmclab.py
```

## 2025-05-21 

Ran and adjusted multiple models. Changed the optimizer to Adam as it is faster
and tunes itself.
All models were run today: 
```sh 
python ./src/scripts/run_cnn1l_ppmclab.py
python ./src/scripts/run_cnn1l_bd_ppmclab.py
python ./src/scripts/run_cnn1l_pspire.py
python ./src/scripts/run_cnn1l_bd_pspire.py
python ./src/scripts/run_cnn2l_ppmclab.py
python ./src/scripts/run_cnn2l_bd_ppmclab.py
python ./src/scripts/run_cnn2l_pspire.py
python ./src/scripts/run_cnn2l_bd_pspire.py
```
The pspire data set seems to be a challange. For now the best model was the `cnn2l_pspire` model. 

## 2025-05-22 
Build models that train on ppmclab and pspire data to test the pspire data set: 
```sh 
python ./src/scripts/run_cnn2l_ppmclab_pspire.py
python ./src/scripts/run_cnn2l_att_ppmclab_pspire.py
python ./src/scripts/run_transformer_ppmclab_pspire.py
```
To see if it helps to train the model on an extra label for ps proteins with idr
content a model was trained with this label:
```sh
python ./src/scripts/run_cnn2l_ppmclab_pspire_multi.py 
```

To analyze the results a train test loop was created for multiclass cases:
`./src/modules/train_eval_multi.py`


## 2025-05-23

Downloaded Dataset S01 and S06 from the [PhaSepDB article](https://www.pnas.org/doi/10.1073/pnas.2115369119#supplementary-materials)
and placed them as follows:
S02 saps: `./data/raw_data/phasepdb_saps.csv`
S02 pdps: `./data/raw_data/phasepdb_pdps.csv`
S02 nops: `./data/raw_data/phasepdb_nops.csv`
S03 saps test: `./data/raw_data/phasepdb_saps_test.csv`
S03 pdps test: `./data/raw_data/phasepdb_pdps_test.csv`
S03 nops test: `./data/raw_data/phasepdb_nops_test.csv`
S03 ps test: `./data/raw_data/phasepdb_ps_test.csv`
S06: `./data/raw_data/phasepdb_mlo.csv`

Downloaded the supplementary data 5 from the [PSPire article](https://www.nature.com/articles/s41467-024-46445-y#MOESM9) 
and placed it as follows: `./data/raw_data/pspire_mlo.csv`

Wrote a script to prepare the data from phasepbd and ran it:
```sh
python ./src/scripts/prepare_phasepdb.py

```
## 2025-05-26 to 2025-05-28

Created scripts to prepare the data from the mlo data sets (phasepdb and pspire)
and ran them: 

```sh 
python prepare_phasepdb_mlo.py 
python prepare_pspire_mlo.py 
```
Created a script to test a simpler model (xgb) on the data of the block
decomposition and ran it on the phasepdb data: 

```sh 
python run_xgb_phasepdb.py
```

Started to evaluate the currently best model (one trained on the phasepdb data,
the other on the union of pspire and ppmclab) on the MLO data:
```sh
python eval_phasepdb_G3BP1.py
python eval_phasepdb_dact1.py
python eval_phasepdb_opencell.py
python eval_phasepdb_phasepdb_high.py
python eval_pspire_dact1.py
python eval_pspire_drllps.py
python eval_pspire_g3bp1.py
python eval_pspire_psdbht.py
```

Created a script that downloads the data of alphafold and calculates the surface
availability for the pspire data:
```sh 
python prepare_pspire_alpha.py
```

## 2025-06-02

Created scripts to download the alphafold data for all datasets and maybe run
them, if the internet service guy comes around.
```sh 
python prepare_pspire_alpha.py
python prepare_pspire_mlo_alpha.py
python prepare_ppmclab_alpha.py
python prepare_phasepdb_alpha.py
python prepare_phasepdb_mlo_alpha.py
```

## 2025-06-03 - 2025-06-14

Created models that take the rsa values into consideration. 
One that uses it as separate feature. 
```sh 
python ./src/modules/cnn_2l_rsa.py
```
One that uses a linear layer for the rsa values. 
```sh 
python ./src/modules/cnn_2l_rsa_linear.py
```
One that uses the rsa values as weights for the amino acids. 
```sh 
python ./src/modules/cnn_2l_rsa_weight.py
```

Other optimizations were tried: 
Again, using a third convolutional layer. 
```sh
python ./src/modules/cnn_3l_rsa.py
```

Adding batch normalization: 
```sh
python ./src/scripts/run_cnn2l_pspire_bn.py
```

Adding layers with multiple kernel sizes: 
```sh 
python ./src/scripts/run_cnn2l_msf.py
```

Adding layers with an attention like mechanism: 
```sh 
python ./src/scripts/run_cnn2l_att_pspire.py
```

The batch normalization and the rsa as weights was combined, as they were the
most affective. 

```sh
python ./src/scripts/run_cnn2l_pspire_rsa_weight_bn.py
```

Posttranslational Modification Sites were added as they do affect the ability of
proteins to undergo phase separation.

```sh 
python ./src/scripts/run_cnn2l_pspire_rsa_weight_bn_idr_ptm.py
```

To see if it helps the model if it learns idrs and non idrs separately, two
separate models were created and run: 

```sh
python ./src/scripts/run_cnn2l_pspire_rsa_weight_bn_idr_ptm.py
python ./src/scripts/run_cnn2l_pspire_rsa_weight_bn_nidr_ptm.py
```

## 2025-06-16

Created a model that integrates a transformer into the already strong two layer
cnn. 

```sh 
python src/scripts/run_cnn2l_trans_pspire_rsa_weight_bn_idr_ptm.py
```
