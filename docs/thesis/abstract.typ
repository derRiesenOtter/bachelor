#heading(outlined: false, level: 2)[Abstract]

This work presents a novel sequence-based predictor for @llps proteins using
@nn::pl. In addition to raw amino acid sequences, a block decomposition of
these sequences was explored as input. Various @nn architectures were
evaluated, including @cnn::pl, @bilstm::pl, and Transformers. The model was
further enhanced with additional features such as @rsa and @ptm::pl. Unlike
existing @llps predictors that rely on conventional @ml models with tabular
inputs, this study investigates the direct use of @nn on sequence
data.

The results demonstrate that even relatively simple @nn architectures can match
the performance of current state-of-the-art predictors. Raw sequences
outperformed block decomposition in most cases. While @llps prediction remains
challenging due to limited experimental data and complex biological mechanisms,
this study suggests that sequence-based @nn models hold strong potential. As
more data becomes available, such models may define the next generation of
@llps predictors.

#pagebreak()
