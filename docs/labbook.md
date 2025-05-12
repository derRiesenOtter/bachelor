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

The file `datasets.tsv` was downloaded and moved and renamed to
`./data/raw_data/llps_data_ppmclab.tsv`.
