#heading(outlined: false, level: 2)[Abstract]

Liquid-Liquid Phase Separation of proteins is a fundamental cellular mechanism
responsible for compartmentalization without membrane boundaries. It also plays
a role in the development of certain diseases. Computational prediction of
Liquid-Liquid Phase Separation propensity helps in understanding protein
interactions in cellular contexts where experimental data is lacking.

This work presents a novel sequence-based predictor for Liquid-Liquid Phase
Separation proteins using Neural Networks. In addition to raw amino acid
sequences, a block decomposition of these sequences was explored as main input.
Various Neural Network architectures were evaluated, including Convolutional
Neural Networks, Bidirectional Long-Short Term Memory models, and Transformers.
From these models the best performing was chosen and further enhanced with
additional features such as Relative Solvent Accessibility and Post
Translational Modifications. Unlike existing Liquid-Liquid Phase Separation
predictors that rely on conventional Machine Learning models and therefore use
tabular inputs, this approach enables the direct use of sequence data.

The results demonstrate that even relatively simple Neural Network
architectures can match the performance of current state-of-the-art predictors.
The usage of the raw sequences proved to be better than using the block
decomposition. While Liquid-Liquid Phase Separation prediction remains
challenging due to limited experimental data and complex biological mechanisms,
this study suggests that sequence-based Neural Network models hold strong
potential. As more data becomes available, such models may define the next
generation of Liquid-Liquid Phase Separation predictors.

#pagebreak()
