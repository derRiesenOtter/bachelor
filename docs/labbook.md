# Lab book 

## How to Reproduce 

This section is supposed to help reproducing the results that are found 
in the thesis that resulted from this work. To reproduce follow the 
instructions below:

1. Install the [Words](https://gitlab.mpcdf.mpg.de/mgirard/Words) module of
   Martin Girard if you want to reproduce the block decomopsition experiments. 
   **Important**: The dev branch needs to be installed.
1. Clone this repository and `cd` into it: 
    ```sh 
    git clone https://github.com/derRiesenOtter/bachelor.git
    cd bachelor
    ```
1. Create a virtual python environment and install the packages from the 
    `./requirements.txt` file. 
1. Acquire the datasets from the following sources: 
    - [PPMC-lab](https://llpsdatasets.ppmclab.com) (datasets.tsv)
    - [PSPire](https://www.nature.com/articles/s41467-024-46445-y) (Supplementary Data 4, Supplementary Data 5)
    - [catGranule 2.0](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03497-7) (13059_2025_3497_MOESM4_ESM.xlsx)
1. Rename these datasets to:
    - `ppmclab.tsv`
    - `pspire.csv` and `pspire_mlo.csv`
    - `catgranule2.csv`

    and place them into the `./data/raw_data/` folder
1. Run the preparation scripts:  
    ```sh 
    python ./src/scripts/prepare_ppmclab.py
    python ./src/scripts/prepare_pspire.py
    python ./src/scripts/prepare_pspire_mlo.py
    python ./src/scripts/prepare_catrgranule2.py
    ```
1. Run the scripts that download the RSA data: 
    ```sh 
    python ./src/scripts/prepare_ppmclab_alpha.py
    python ./src/scripts/prepare_pspire_alpha.py
    python ./src/scripts/prepare_pspire_mlo_alpha.py
    python ./src/scripts/prepare_catgranule2_alpha.py
    ```
1. Run the scripts that download the PTM data: 
    ```sh 
    python ./src/scripts/get_ptms_ppmclab.py
    python ./src/scripts/get_ptms_pspire.py
    python ./src/scripts/get_ptms_catgranule2.py
    ```
1. Run the block decomposition on the PPMC-lab and PSPire dataset: 
    ```sh 
    python ./src/scripts/run_bd_ppmclab.py
    python ./src/scripts/run_bd_pspire.py
    ```
1. Run the models you would like to rerun (only the scripts used to produce data that was used in the thesis are listed below): 
    - The models that used the block decompositon:
        ```sh 
        python ./src/scripts/run_cnn1l_bd_ppmclab.py
        python ./src/scripts/run_cnn1l_bd_pspire.py
        python ./src/scripts/run_cnn2l_bd_ppmclab.py
        python ./src/scripts/run_cnn2l_bd_pspire.py
        python ./src/scripts/run_xgb_pspire.py
        ```
    - The models that used the raw protein sequence: 
        ```sh 
        python ./src/scripts/run_cnn1l_ppmclab.py
        python ./src/scripts/run_cnn1l_pspire.py
        python ./src/scripts/run_cnn2l_ppmclab.py
        python ./src/scripts/run_cnn2l_pspire.py
        python ./src/scripts/run_cnn2l_pspire_idr.py
        python ./src/scripts/run_cnn2l_pspire_nidr.py
        python ./src/scripts/run_cnn2l_pspire_bn_idr.py
        python ./src/scripts/run_cnn2l_pspire_bn_nidr.py
        python ./src/scripts/run_cnn2l_pspire_rsa_idr.py
        python ./src/scripts/run_cnn2l_pspire_rsa_nidr.py
        python ./src/scripts/run_cnn2l_pspire_rsa_weight_idr.py
        python ./src/scripts/run_cnn2l_pspire_rsa_weight_nidr.py
        python ./src/scripts/run_cnn2l_pspire_idr_ptm.py
        python ./src/scripts/run_cnn2l_pspire_nidr_ptm.py
        python ./src/scripts/run_cnn2l_ppmclab_bn_idr.py
        python ./src/scripts/run_cnn2l_ppmclab_bn_nidr.py
        python ./src/scripts/run_cnn2l_ppmclab_rsa_idr.py
        python ./src/scripts/run_cnn2l_ppmclab_rsa_nidr.py
        python ./src/scripts/run_cnn2l_ppmclab_rsa_weight_idr.py
        python ./src/scripts/run_cnn2l_ppmclab_rsa_weight_nidr.py
        python ./src/scripts/run_cnn2l_ppmclab_ptm_idr.py
        python ./src/scripts/run_cnn2l_ppmclab_ptm_nidr.py
        python ./src/scripts/run_cnn2l_pspire_rsa_weight_bn_idr.py
        python ./src/scripts/run_cnn2l_pspire_rsa_weight_bn_nidr.py
        python ./src/scripts/run_cnn2l_pspire_rsa_weight_ptm_idr.py
        python ./src/scripts/run_cnn2l_pspire_rsa_weight_ptm_nidr.py
        python ./src/scripts/run_cnn2l_pspire_rsa_weight_bn_idr_ptm.py
        python ./src/scripts/run_cnn2l_pspire_rsa_weight_bn_nidr_ptm.py
        python ./src/scripts/run_cnn2l_catgranule2_rsa_weight.py
        python ./src/scripts/run_cnn2l_catgranule2_rsa_weight_bn_ptm.py
        ```
    - The evaluation scripts for the MLO datasets:
        ```sh 
        python ./src/scripts/eval_ppmclab_dact1_idr.py
        python ./src/scripts/eval_ppmclab_dact1_idr_self.py
        python ./src/scripts/eval_ppmclab_dact1_nidr.py
        python ./src/scripts/eval_ppmclab_dact1_nidr_self.py
        python ./src/scripts/eval_ppmclab_drllps_idr.py
        python ./src/scripts/eval_ppmclab_drllps_nidr.py
        python ./src/scripts/eval_ppmclab_g3bp1_idr.py
        python ./src/scripts/eval_ppmclab_g3bp1_nidr.py
        python ./src/scripts/eval_ppmclab_phasep_idr.py
        python ./src/scripts/eval_ppmclab_phasep_nidr.py
        python ./src/scripts/eval_ppmclab_rnagranule_idr.py
        python ./src/scripts/eval_ppmclab_rnagranule_nidr.py
        python ./src/scripts/eval_pspire_dact1_w_idr.py
        python ./src/scripts/eval_pspire_dact1_wn_nidr_ptm.py
        python ./src/scripts/eval_pspire_drllps_w_idr.py
        python ./src/scripts/eval_pspire_drllps_wn_nidr_ptm.py
        python ./src/scripts/eval_pspire_g3bp1_w_idr.py
        python ./src/scripts/eval_pspire_g3bp1_wn_nidr_ptm.py
        python ./src/scripts/eval_pspire_phasep_w_idr.py
        python ./src/scripts/eval_pspire_phasep_wn_nidr_ptm.py
        python ./src/scripts/eval_pspire_rnagranule_w_idr.py
        python ./src/scripts/eval_pspire_rnagranule_wn_nidr_ptm.py
        ```
    - The Captum Evaluation: 
        ```sh 
        python ./src/scripts/eval_test_captum_idr.py
        python ./src/scripts/eval_test_captum_nidr.py
        ```
1. The evaluation metrics are all visible inside the plots 
   that were created by running the scripts at `./results/plots/`.

## Timeline 

In the following section, the actions conducted in this work are 
chronologically listed.

### 2025-05-12

*git-tag 0.1.1*

The PPMC-lab dataset was downloaded.
Source: [llpsdatasets](https://llpsdatasets.ppmclab.com)

The file `datasets.tsv` was downloaded and moved to
`./data/raw_data/llps_data_ppmclab.tsv`.

---

The data of the PPMC-lab dataset was prepared for further usage using  
the script `./src/scripts/prepare_raw_data.py` and the command:

```sh 
python src/scripts/prepare_raw_data.py
```
Creating:
`./data/intermediate_data/llps_data_ppmclab.pkl`

---

The block decomposition algorithm was run on the PPMC-lab dataset using 
the script `./src/scripts/run_block_decomposition.py` and the command:
```sh 
python src/scripts/run_block_decomposition.py
```
Creating:
`./data/intermediate_data/llps_data_ppmclab_bd.pkl`

### 2025-05-13 

*git-tag 0.1.2*

As new mappings were added to the block decomposition the
script `./src/scripts/run_block_decomposition.py` was run again:

```sh 
python src/scripts/run_block_decomposition.py
```
Creating:
`./data/intermediate_data/llps_data_ppmclab_bd.pkl`

---

A simple neural network was created and run with the command:

```sh
python ./src/scripts/prepare_training_data.py
```

### 2025-05-14 

The PSPire dataset was downloaded from the article ([source](https://www.nature.com/articles/s41467-024-46445-y)). 

It was then moved to `data/raw_data/ps_pire_data.csv`.

### 2025-05-15

*git-tag 0.1.3*

As the file names were not ideal, many files were renamed using the 
commands below:

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

### 2025-05-19

_git-tag 0.1.4_

To download the protein sequences of PSPire and prepare the 
data for further analysis the script `./src/scripts/prepare_pspire.py` 
was run with the command: 

```sh 
python ./src/scripts/prepare_pspire.py
```

---

The script for preparing the PPMC-lab dataset was modified to yield some
graphics and rerun: 

```sh 
python ./src/scripts/prepare_ppmclab.py
```

---

Started to modularize code. Datasets and models will now get a script each that
resides in `./src/modules/`.
Following files have been created there:
`./src/modules/bd_cnn_1l.py`
`./src/modules/bd_sequence_dataset.py`
`./src/modules/sequence_dataset.py`
`./src/modules/mappings.py`

The module `./src/modules/block_decomposition_modifier.py` was renamed to
`./src/modules/bd_tools.py` using `mv`.

### 2025-05-20

*git-tag 0.1.4*

The block decomposition algorithm was run on the PSPire dataset with 
the following command:
```sh 
python ./src/scripts/run_bd_pspire.py
```
---

The parameters in the one layer block decomposition cnn were modified 
as well as the name of the script: 
```sh 
mv .src/scripts/run_cnn_ppmclab.py ./src/scripts/run_cnn1l_bd_ppmclab.py
python ./src/scripts/run_cnn1l_bd_ppmclab.py
```

### 2025-05-21 

*git-tag 0.1.5*

Ran and adjusted multiple models. Changed the optimizer to Adam as it is faster.
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

### 2025-05-22 

*git-tag 0.1.6*

Experimented with models that use both the PSPire and PPMC-lab 
dataset and different architectures: 
```sh 
python ./src/scripts/run_cnn2l_ppmclab_pspire.py
python ./src/scripts/run_cnn2l_att_ppmclab_pspire.py
python ./src/scripts/run_transformer_ppmclab_pspire.py
```
---

Tested a model that used three labels. One for negatives, 
one for positives with idr and one for positives without idr.

```sh
python ./src/scripts/run_cnn2l_ppmclab_pspire_multi.py 
```

### 2025-05-23

Downloaded the PhasePred data from the [PhaSePred article](https://www.pnas.org/doi/10.1073/pnas.2115369119#supplementary-materials)
and placed them as follows:
S02 saps: `./data/raw_data/phasepdb_saps.csv`
S02 pdps: `./data/raw_data/phasepdb_pdps.csv`
S02 nops: `./data/raw_data/phasepdb_nops.csv`
S03 saps test: `./data/raw_data/phasepdb_saps_test.csv`
S03 pdps test: `./data/raw_data/phasepdb_pdps_test.csv`
S03 nops test: `./data/raw_data/phasepdb_nops_test.csv`
S03 ps test: `./data/raw_data/phasepdb_ps_test.csv`
S06: `./data/raw_data/phasepdb_mlo.csv`

---

Downloaded the mlo data from the [PSPire article](https://www.nature.com/articles/s41467-024-46445-y#MOESM9) 
and placed it as follows: `./data/raw_data/pspire_mlo.csv`

---

Prepared the data from the PhasPred article using:
```sh
python ./src/scripts/prepare_phasepdb.py

```
### 2025-05-26 to 2025-05-28

Created scripts to prepare the data from the mlo data sets (phasepred and pspire)
and ran them: 

```sh 
python prepare_phasepdb_mlo.py 
python prepare_pspire_mlo.py 
```

---
Created a script to test a xgb model on the data of the block
decomposition and ran it on the phasepdb data: 

```sh 
python run_xgb_phasepdb.py
```

---
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
---

Created a script that downloads the data of alphafold and calculates the surface
availability for the pspire data:
```sh 
python prepare_pspire_alpha.py
```

### 2025-06-02

*git-tag 0.1.7*
*git-tag 0.1.8*

Created scripts to download the alphafold data for all datasets and 
ran them over the next days:
```sh 
python prepare_pspire_alpha.py
python prepare_pspire_mlo_alpha.py
python prepare_ppmclab_alpha.py
python prepare_phasepdb_alpha.py
python prepare_phasepdb_mlo_alpha.py
```

### 2025-06-03 

Created models that take the rsa values into consideration. 
One that uses it as separate feature. 
```sh 
python ./src/modules/cnn_2l_rsa.py
python ./src/modules/cnn_2l_rsa_linear.py
python ./src/modules/cnn_2l_rsa_weight.py
```
---

More modifications to the architecture were tested.

```sh
python ./src/modules/cnn_3l_rsa.py
python ./src/scripts/run_cnn2l_pspire_bn.py
python ./src/scripts/run_cnn2l_msf.py
python ./src/scripts/run_cnn2l_att_pspire.py
```

---

The batch normalization and the RSA as weights was combined, as they were the
most affective. 

```sh
python ./src/scripts/run_cnn2l_pspire_rsa_weight_bn.py
```

### 2025-06-12

PTMs were downloaded from uniprot for the PSPire dataset: 

```sh 
python get_ptms_pspire.py
```
---

Posttranslational Modification Sites were added as they do affect the ability of
proteins to undergo phase separation.

```sh 
python ./src/scripts/run_cnn2l_pspire_rsa_weight_bn_idr_ptm.py
```

---

To see if it helps the model if it learns idrs and non idrs separately, two
separate models were created and run: 

```sh
python ./src/scripts/run_cnn2l_pspire_rsa_weight_bn_idr_ptm.py
python ./src/scripts/run_cnn2l_pspire_rsa_weight_bn_nidr_ptm.py
```

### 2025-06-16

PTMs were downloaded from UniProt for the PPMC-lab dataset and the PhasPred dataset:

```sh 
python get_ptms_ppmclab.py
python get_ptms_phasepdb.py
```

---

Tested combining cnn and transformer:

```sh 
python src/scripts/run_cnn2l_trans_pspire_rsa_weight_bn_idr_ptm.py
```

### 2025-06-17

*git-tag 0.1.9*

Tested around with three layer cnns. All scripts starting with `run_cnn3l_pspire`
were run with the python command.

### 2025-06-24 

Downloaded the dataset from [catGranule 2.0](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03497-7) and moved it to `./data/raw_data/catgranule2.csv`

---

Prepared the catGranule 2.0 dataset for further analysis: 

```sh 
python ./src/scripts/prepare_catrgranule2.py
```

### 2025-06-25

Downloaded the RSA and PTM values for the catGranule 2.0 dataset: 

```sh 
python ./src/scripts/prepare_catgranule2_alpha.py
python ./src/scripts/get_ptms_catgranule2.py
```

--- 

Run some models on the catGranule 2.0 dataset: 

```sh 
python ./src/scripts/run_cnn2l_catgranule2_rsa_weight_bn_ptm.py
```

## 2025-07-07 

After checking that all model values are comparable 
all important scripts that run models or evaluations were 
rerun. They are listed in the section "Reproducibility".
