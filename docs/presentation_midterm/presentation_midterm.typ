#import "@preview/touying:0.6.1": *
#import themes.simple: *
#show: simple-theme.with(aspect-ratio: "16-9")
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge, shapes.triangle, shapes.pill, shapes.cylinder

= Midterm meeting: Bachelor thesis of Robin Ender
#place(top + left, image("figures/cbdm_logo.png", width: 40%))
#place(top + right, image("figures/th_bingen_logo.jpg", width: 20%))
Date: _2025-06-12_ \
Attendees: _Robin Ender, Asis Hallab, Eric Schumbera_

== Current status of phase separation prediction
- Many Phase Separation Predictors have emerged, yet the performance is still
  not optimal
  - the best ones use scalar features (e.g. percentage of IDR)
- Problems are:
  - Lacking Training Data
  - Bias towards proteins containing IDRs
- Idea:
  - Using the sequence (block decomposition) to predict PS and try a newly
    published data set

== Repetition - Block Decomposition
- The block decomposition algorithm takes a protein sequence and a mapping
- It outputs a list of blocks that all have a certain word balance ("uniformity")
- Steps to use it for neural networks:
  - Adjusting the output
  - Finding relevant mappings

== Adjusting the output of the block decomposition algorithm

The block decomposition algorithm was modified to yield an output that can be
used in a neural network:

#place(left, block(width: 50%)[*Old Output*: \
[(5, 14),(15, 22)...]])
#place(right, block(width: 50%)[*New Output*: \
[0,0,0,0,1,1,1,...,2,2,2...]])

#v(25%)
- The list of tuples was converted to one list representing the sequence.
- Labels were created, representing the most common group(s).

== Finding meaningful mappings
Seven Mappings were found (in literature) that relate to phase separation or are generally
meaningful:
#align(center, table(
  columns: 1,
  "Aliphatic - Aromatic - Positive - Negative",
  "RG-Mappings (two separate)",
  "IDR-Mapping",
  "Most meaningful 5 mapping",
  "PiPi-Mapping (two separate)"
))
Those were used to generate one block decomposition for each mapping per
protein.

== Data sets I
For now two different data sets were used:
- PPMCLABs llps data set
  - created in an effort to create a dataset with an appropriate negative data
    set (many studies use RCSB Protein Data Bank, which are not guaranteed to be
    non phase separating)
  // - this model also classifies the positive proteins into _driver_ (proteins
  //   that can undergo PS on their own) and _client_
  //   (proteins recruited to the preexisting site)
  - contains 746 positive and 2077 negative entries
#place(diagram())

// == Data sets II
// - PhaSePred data set
//   - used to train the two tools SaPS and PdPS, addressed the difficulty of
//     predicting phase separating proteins that are _partner dependent_
//   - uses proteins that are not annotated as PS as negatives
//   - contains 567 positive and around 60,000 negative entries

== Data sets II
- PSPires data set
  - used to train PSPire, a recent well performing PS predictor addressing the
    difficulty of predicting phase separating proteins with no IDRs
  // - similar to PhaSePred data set, but less negative entries
  - contains 517 positive and around 10,000 negative entries

// == Main Goals
// #align(center, diagram(
//   spacing: (2mm, 4mm),
//   node((0, 0), [Check if deep learning models are able to learn PS prediction], shape: pill, fill: rgb(150, 200, 200)),
//   node((0, 1), [Check if the block decomposition is beneficial over the sequence alone], shape: pill, fill: rgb(150, 200, 200)),
//   node((0, 2), [See which architecture is the best], shape: pill, fill: rgb(150, 200, 200)),
//   node((0, 3), [Consider Driver Client], shape: pill, fill: rgb(150, 200, 200)),
//   node((0, 4), [Consider IDRs], shape: pill, fill: rgb(150, 200, 200)),
// ))
//

== Test if a simple cnn model can learn from the data
To see if a cnn model was able to learn a basic 1 layer model
was created and run on the PPMCLAB data set:
#place(left, image("figures/cnn1l_bd_ppmclab_cm.png", width: 35%))
#place(center, image("figures/cnn1l_bd_ppmclab_rocauc.png", width: 33%))
#place(right, image("figures/cnn1l_bd_ppmclab_prauc.png", width: 33%))

== Testing different models
As the CNN2l model performed relatively good and was relatively quick in training it
was used for further tests:
#align(center, diagram(
  spacing: (3mm, 10mm),
  node((0, 0), [Start], shape: triangle, fill: rgb(150, 200, 200), stroke: black),

  edge((0, 0), (-2, 1), "-"),
  node((-2, 1), [BD], shape: pill, fill: red, stroke: black),

  edge((-2, 1), (-1, 2), "-"),
  node((-1, 2), [CNN2L], shape: pill, fill: red, stroke: black),

  edge((-2, 1), (-2, 2), "-"),
  node((-2, 2), [CNN1L], shape: pill, fill: red, stroke: black),

  edge((-2, 1), (-3, 2), "-"),
  node((-3, 2), [XGBoost], shape: pill, fill: red, stroke: black),

  edge((0, 0), (2, 1), "-"),
  node((2, 1), [Seq], shape: pill, fill: red, stroke: black),

  edge((2, 1), (1, 2), "-"),
  node((1, 2), [CNN1L], shape: pill, fill: red, stroke: black),

  edge((2, 1), (3, 2), "-"),
  node((3, 2), [CNN2L], shape: pill, fill: green, stroke: black),

  edge((2, 1), (1, 3), "-"),
  node((1, 3), [BiLSTM], shape: pill, fill: red, stroke: black),

  edge((2, 1), (3, 3), "-"),
  node((3, 3), [Transformer], shape: pill, fill: red, stroke: black),

  edge((2, 1), (2, 3), "-"),
  node((2, 3), [CNN2L + BiLSTM], shape: pill, fill: red, stroke: black),
))

// == Block Decomposition vs Sequence
// The models based on the sequence instead of the block decomposition outperformed
// their counterparts, eg. CNN2L trained and tested on PSPire data set:
//
// #align(center, table(
//   columns: 5,
//   "",      table.cell(colspan: 2)[#text(red)[Block Decomposition]], table.cell(colspan: 2)[#text(green)[Sequence]],
//   "",      "IDR",           "nIDR",                                 "IDR",             "nIDR",
//   "AUROC", text(red)[0.70], text(red)[0.62],                        text(green)[0.74], text(green)[0.64],
//   "PRAUC", text(red)[0.22], "0.05",                                 text(green)[0.26], "0.05",
// ))
//
// == Machine Learning vs. Deep Learning model
// To see if a model like XGBoost is able to use the data created by the block
// decomposition, a simple model was created and compared to the deep learning
// model. It compared very similar to the block decomposition model (PSPire data
// set):
//
// #align(center, table(
//   columns: 4,
//   "",      "XGBoost", "CNN2L BD", "CNN2L SEQ",
//   "AUROC", "0.69",    "0.68",     "0.71",
//   "PRAUC", "0.24",    "0.24",     "0.28"
// ))
//
// == BiLSTM and Transformer
// Both models were not able to outperform the CNN2L model.

== Performance of the initial models on PSPire data
#image("figures/visual_summary_auc_pspire_first_models.png")

== Enhancing the CNN2L models
#align(center, diagram(spacing: (3mm, 5mm), node((0, 0), [CNN2L], shape: triangle, fill: rgb(150, 200, 200), stroke: black), edge((0, 0), (-2, 1), "-"), node((-2, 1), [RSA], shape: pill, fill: rgb(150, 200, 200), stroke: black), edge((-2, 1), (-1, 2), "-"), node((-3, 2), [RSA], shape: pill, fill: red, stroke: black), edge((-2, 1), (-2, 2), "-"), node((-2, 2), [RSA linear], shape: pill, fill: red, stroke: black), edge((-2, 1), (-3, 2), "-"), node((-1, 2), [RSA weights], shape: pill, fill: green, stroke: black), edge((0, 0), (2, 1), "-"), node((2, 1), [Opt.], shape: pill, fill: rgb(150, 200, 200), stroke: black), edge((2, 1), (1, 2), "-"), node((1, 2), [norm.], shape: pill, fill: green, stroke: black), edge((2, 1), (3, 2), "-"), node((3, 2), [Attention], shape: pill, fill: red, stroke: black), edge((0, 3), (1, 2), "-"), edge((0, 3), (-1, 2), "-"), node((0, 3), [both], shape: pill, fill: green, stroke: black), edge((0, 3), (0, 4), "-"), node((0, 4), [split training \
in idr and nidr], inset: 5mm, shape: pill, fill: green, stroke: black), edge((0, 4), (0, 5), "-"), node((0, 5), [Added PTMs], shape: pill, fill: green, stroke: black)))

== Performance of the new models
#image("figures/visual_summary_auc_pspire_later_models.png")

== Performance on MLO data sets
#align(center, table(
  columns: 6,
  "PSPs",                             "Dataset",                          "Parameter", "My model", "PdPS", "PSPire",
  table.cell(rowspan: 10)[noID-PSPs], table.cell(rowspan: 2)[G3BP1],      "ROCAUC",    [*0.96*],   "0.81", "0.93",
                                                                          "PRAUC",     "0.51",     "0.18", [*0.66*],
                                      table.cell(rowspan: 2)[DACT1],      "ROCAUC",    "0.90",     "0.81", [*0.93*],
                                                                          "PRAUC",     "0.49",     "0.18", [*0.60*],
                                      table.cell(rowspan: 2)[RNAGranule], "ROCAUC",    "0.88",     "0.68", [*0.90*],
                                                                          "PRAUC",     "0.18",     "0.08", [*0.28*],
                                      table.cell(rowspan: 2)[PhaSep],     "ROCAUC",    [*0.85*],   "0.65", "0.80",
                                                                          "PRAUC",     [*0.73*],   "0.47", "0.71",
                                      table.cell(rowspan: 2)[DRLLPS],     "ROCAUC",    "0.80",     "0.68", [*0.85*],
                                                                          "PRAUC",     [*0.77*],   "0.45", "0.74",
))

== Performance on MLO data sets
#align(center, table(
  columns: 6,
  "PSPs",                           "Dataset",                          "",       "My model", "PdPS",   "PSPire",
  table.cell(rowspan: 10)[ID-PSPs], table.cell(rowspan: 2)[G3BP1],      "ROCAUC", "0.74",     "0.86",   [*0.91*],
                                                                        "PRAUC",  "0.29",     "0.41",   [*0.58*],
                                    table.cell(rowspan: 2)[DACT1],      "ROCAUC", "0.72",     "0.85",   [*0.88*],
                                                                        "PRAUC",  "0.22",     "0.33",   [*0.35*],
                                    table.cell(rowspan: 2)[RNAGranule], "ROCAUC", "0.80",     "0.82",   [*0.84*],
                                                                        "PRAUC",  "0.39",     "0.42",   [*0.48*],
                                    table.cell(rowspan: 2)[PhaSep],     "ROCAUC", "0.71",     [*0.74*], "0.72",
                                                                        "PRAUC",  "0.70",     [*0.80*], "0.79",
                                    table.cell(rowspan: 2)[DRLLPS],     "ROCAUC", "0.69",     [*0.76*], "0.75",
                                                                        "PRAUC",  "0.71",     "0.77",   [*0.78*],
))

== Results in context of the initial idea of this work
- block decomposition can be used to predict phase separation
  - but using the raw sequence yields better results
  - therefore the focus has shifted to using the sequence and additional data to
    build a good phase separation predictor
- while the ppmclab data set should be a beeter training set, it failed to
  predict the MLOs

== Visualizing the results
- ROC AUC and PR AUC
- bar plots / tables for comparison with other tools
- Saliency (shows which input positions the model “cares about most” when making its prediction)
  probably only for some visualizations
https://bbb.rlp.net/rooms/hal-cta-mfd-wzm/join

== Saliency
#image("figures/saliency_test.png")
#place(image("figures/mobi.png", width: 80%))

== What can / should be done in the remaining time?
- Cross validation of the final model
- writing!
- next week, report will be send
- last meeting: 10.07.
- end of this thesis: 04.08.
- personal deadline: 25.07.

== Structure of the thesis
- Introduction
  - Liquid-Liquid Phase Separation
  - Block decomposition of protein sequences
  - Current predictors and the difficulties
  - Machine Learning in Bioinformatics
    - CNN
- Material
  - Data (explain and visualize data sets)
- Methods
  - Tools
  - data preparation
  - model architectures
  - model optimizations
- Results
  - comparison of my own models
    - most important block decompositions
  - comparison with other models
    - on their test data
    - on mlo data
  - visualizations of important sequence segments
- Discussion
  - Usefulness of this model
  - What should / could still be done

//
// == Data Preparation
// The datasets were filtered for:
// - sequence length: only sequences smaller than 2700 residues
// - unknown amino acid annotations: sequences containing letters that did not fit
//   into the mappings were filtered out
//
// == Most basic model I
// To see if a convolutional neural network is able to learn, a very basic model was
// created. It had the following structure (80/20 train/test split):
// + Embedding Layer, creating an embedding for each of the block decompositions
// + Convolutional Layer
// + ReLu Activation Layer
// + MaxPoolingLayer
// + GlobalPoolingLayer
// + Dropout layer
// + Linear Layer (Fully Connected)
//
// == Most basic model II
// The first run showed, that the model is able to learn from the data.
// #place(left, image("figures/cnn1l_bd_ppmclab_cm.png", width: 33%))
// #place(center, image("figures/cnn1l_bd_ppmclab_rocauc.png", width: 33%))
// #place(right, image("figures/cnn1l_bd_ppmclab_prauc.png", width: 33%))
//
// == Comparison with only the sequence
// To see if the block decomposition yielded a benefit over feeding the model with
// the raw sequence as input, a model using only the sequence was created and run:
//
// #place(left, image("figures/run_cnn1l_ppmclab_cm.png", width: 33%))
// #place(center, image("figures/run_cnn1l_ppmclab_rocauc.png", width: 33%))
// #place(right, image("figures/run_cnn1l_ppmclab_prauc.png", width: 33%))
//
// == Comparison to PSPire
// To compare the performance of this simple model to the performance of PSPire,
// the model was trained and validated on the same data.
//
// #align(center, table(
//   columns: 7,
//   "",       "cnn1l IDR (bd)", "cnn1l nIDR (bd)", "cnn1l IDR (seq)", "cnn1l nIDR (seq)", "PSPire IDR", "PSPire nIDR",
//   "AUCROC", "0.50",           "0.61",            "0.70",            "0.64",             "0.86",       "0.84",
//   "PRAUC",  "0.09",           "0.06",            "0.20",            "0.06",             "0.51",       "0.24"
// ))
//
// Second best predictor (PdPS): \
// *IDR* AUROC:0.84 PRAUC:0.42, *nIDR* AUROC: 0.68,
// PRAUC: 0.08
//
// == Conclusion for the simple model
// While both models did not perform too well, they already outperformed many older
// predictors for the nIDR proteins. The model using the block decomposition
// however performed significantly worse than the model that used only the
// sequence.
//
// A second model was created with an additional convolutional layer (+activation,
// +pooling).
//
// == PPMCLAB 1 layer vs 2 layers
// Block Decomposition:
// #align(center, table(
//   columns: 5,
//   "",      "cnn1l IDR", "cnn1l nIDR", "cnn2l IDR", "cnn2l nIDR",
//   "AUROC", "0.50",      "0.61",       "0.86",      "0.70",
//   "PRAUC", "0.09",      "0.06",       "0.64",      "0.78"
// ))
//
// == PPMCLAB 1 layer vs 2 layers
// Sequence:
// #align(center, table(
//   columns: 5,
//   "",      "cnn1l IDR", "cnn1l nIDR", "cnn2l IDR", "cnn2l nIDR",
//   "AUROC", "0.87",      "0.65",       "0.88",      "0.71",
//   "PRAUC", "0.66",      "0.71",       "0.68",      "0.74"
// ))
//
// == 2 layer cnn comparison to PSPire
// To compare the performance of this simple model to the performance of PSPire,
// the model was trained and validated on the same data.
//
// #align(center, table(
//   columns: 7,
//   "",       "cnn2l IDR (bd)", "cnn2l nIDR (bd)", "cnn2l IDR (seq)", "cnn2l nIDR (seq)", "PSPire IDR", "PSPire nIDR",
//   "AUCROC", "0.70",           "0.62",            "0.74",            "0.64",             "0.86",       "0.84",
//   "PRAUC",  "0.22",           "0.05",            "0.26",            "0.05",             "0.51",       "0.24"
// ))
//
// Second best predictor (PdPS): \
// *IDR* AUROC:0.84 PRAUC:0.42, *nIDR* AUROC: 0.68, PRAUC: 0.08
//
// == Conclusion for the 2 convolutional layer model
// Both models benefited from the additional layer, especially the model using the
// block decomposition. In the end the sequence model still outperformed it.
// As a next step the introduction of other layers was tested (bilstm and
// transformer).
//
// == BiLSTM TODO
// Sequence:
// #align(center, table(
//   columns: 5,
//   "",      "BiLSTM IDR", "BiLSTM nIDR", "cnn2l IDR", "cnn2l nIDR",
//   "AUROC", [0.87],       [0.65],        "0.74",      "0.64",
//   "PRAUC", [0.66],       [0.71],        "0.26",      "0.05"
// ))
//
// == Cnn2l + BiLSTM TODO
// Sequence:
// #align(center, table(
//   columns: 5,
//   "",      "BiLSTM + Cnn2l IDR", "BiLSTM + Cnn2l nIDR", "cnn2l IDR", "cnn2l nIDR",
//   "AUROC", [0.87],               [0.65],                "0.74",      "0.64",
//   "PRAUC", [0.66],               [0.71],                "0.26",      "0.05"
// ))
//
// == Transformer TODO
// Sequence:
// #align(center, table(
//   columns: 5,
//   "",      "Transformer IDR", "Transformer nIDR", "cnn2l IDR", "cnn2l nIDR",
//   "AUROC", [0.87],            [0.65],             "0.74",      "0.64",
//   "PRAUC", [0.66],            [0.71],             "0.26",      "0.05"
// ))
//
// == Introducing RSA
// What helped PSPire to outperform the other predictors was the inclusion of
// structural predictions of the proteins. They divided did calculate the features
// for the IDRs and the ordered regions separately. For the ordered regions they
// only used the amino acids with an *Relative Surface Availability* greater than
// 0.25.
//
// As a cnn should be able to learn what the RSA is, it was supplied to the model
// as an additional feature.
//
// == Cnn2l + RSA
// #align(center, table(
//   columns: 5,
//   "",      "Cnn2l + RSA IDR", "Cnn2l + RSA nIDR", "cnn2l IDR", "cnn2l nIDR",
//   "AUROC", text(green)[0.79], text(red)[0.62],    "0.74",      "0.64",
//   "PRAUC", text(green)[0.37], [0.05],             "0.26",      "0.05"
// ))
//
// The performance gain was small. Other methods to feed the RSA value were
// evaluated.
//
// == Cnn2l + RSA linear
// #align(center, table(
//   columns: 5,
//   "",      "Cnn2l + RSA IDR", "Cnn2l + RSA nIDR", "cnn2l IDR", "cnn2l nIDR",
//   "AUROC", text(green)[0.75], [0.64],             "0.74",      "0.64",
//   "PRAUC", text(green)[0.34], text(green)[0.06],  "0.26",      "0.05"
// ))
//
// The performance gain was smaller than without the linear layer.
//
// == Cnn2l + RSA weighted
// #align(center, table(
//   columns: 5,
//   "",      "Cnn2l + RSA IDR", "Cnn2l + RSA nIDR", "cnn2l IDR", "cnn2l nIDR",
//   "AUROC", text(green)[0.76], text(green)[0.70],  "0.74",      "0.64",
//   "PRAUC", text(green)[0.38], text(green)[0.06],  "0.26",      "0.05"
// ))
//
// The method with the RSA values as weights outperformed the other methods. This
// model was subjected to further improvements.
//
// == Cnn2l + RSA weighted +
// #align(center, table(
//   columns: 5,
//   "",      "Cnn2l + RSA IDR", "Cnn2l + RSA nIDR", "cnn2l IDR", "cnn2l nIDR",
//   "AUROC", text(green)[0.76], text(green)[0.70],  "0.74",      "0.64",
//   "PRAUC", text(green)[0.38], text(green)[0.06],  "0.26",      "0.05"
// ))
//
// == Results in the context of the topic
//
// == Presentation of the results
//
// == What can be improved?
//
// == Structure of the thesis
//
// == Data Preparation
// The datasets were filtered for:
// - sequence length: only sequences smaller than 2700 residues
// - unknown amino acid annotations: sequences containing letters that did not fit
//   into the mappings were filtered out
//
// == Most basic model I
// To see if a convolutional neural network is able to learn, a very basic model was
// created. It had the following structure (80/20 train/test split):
// + Embedding Layer, creating an embedding for each of the block decompositions
// + Convolutional Layer
// + ReLu Activation Layer
// + MaxPoolingLayer
// + GlobalPoolingLayer
// + Dropout layer
// + Linear Layer (Fully Connected)
//
// == Most basic model II
// The first run showed, that the model is able to learn from the data.
// #place(left, image("figures/cnn1l_bd_ppmclab_cm.png", width: 33%))
// #place(center, image("figures/cnn1l_bd_ppmclab_rocauc.png", width: 33%))
// #place(right, image("figures/cnn1l_bd_ppmclab_prauc.png", width: 33%))
//
// == Comparison with only the sequence
// To see if the block decomposition yielded a benefit over feeding the model with
// the raw sequence as input, a model using only the sequence was created and run:
//
// #place(left, image("figures/run_cnn1l_ppmclab_cm.png", width: 33%))
// #place(center, image("figures/run_cnn1l_ppmclab_rocauc.png", width: 33%))
// #place(right, image("figures/run_cnn1l_ppmclab_prauc.png", width: 33%))
//
// == Comparison to PSPire
// To compare the performance of this simple model to the performance of PSPire,
// the model was trained and validated on the same data.
//
// #align(center, table(
//   columns: 7,
//   "",       "cnn1l IDR (bd)", "cnn1l nIDR (bd)", "cnn1l IDR (seq)", "cnn1l nIDR (seq)", "PSPire IDR", "PSPire nIDR",
//   "AUCROC", "0.50",           "0.61",            "0.70",            "0.64",             "0.86",       "0.84",
//   "PRAUC",  "0.09",           "0.06",            "0.20",            "0.06",             "0.51",       "0.24"
// ))
//
// Second best predictor (PdPS): \
// *IDR* AUROC:0.84 PRAUC:0.42, *nIDR* AUROC: 0.68,
// PRAUC: 0.08
//
// == Conclusion for the simple model
// While both models did not perform too well, they already outperformed many older
// predictors for the nIDR proteins. The model using the block decomposition
// however performed significantly worse than the model that used only the
// sequence.
//
// A second model was created with an additional convolutional layer (+activation,
// +pooling).
//
// == PPMCLAB 1 layer vs 2 layers
// Block Decomposition:
// #align(center, table(
//   columns: 5,
//   "",      "cnn1l IDR", "cnn1l nIDR", "cnn2l IDR", "cnn2l nIDR",
//   "AUROC", "0.50",      "0.61",       "0.86",      "0.70",
//   "PRAUC", "0.09",      "0.06",       "0.64",      "0.78"
// ))
//
// == PPMCLAB 1 layer vs 2 layers
// Sequence:
// #align(center, table(
//   columns: 5,
//   "",      "cnn1l IDR", "cnn1l nIDR", "cnn2l IDR", "cnn2l nIDR",
//   "AUROC", "0.87",      "0.65",       "0.88",      "0.71",
//   "PRAUC", "0.66",      "0.71",       "0.68",      "0.74"
// ))
//
// == 2 layer cnn comparison to PSPire
// To compare the performance of this simple model to the performance of PSPire,
// the model was trained and validated on the same data.
//
// #align(center, table(
//   columns: 7,
//   "",       "cnn2l IDR (bd)", "cnn2l nIDR (bd)", "cnn2l IDR (seq)", "cnn2l nIDR (seq)", "PSPire IDR", "PSPire nIDR",
//   "AUCROC", "0.70",           "0.62",            "0.74",            "0.64",             "0.86",       "0.84",
//   "PRAUC",  "0.22",           "0.05",            "0.26",            "0.05",             "0.51",       "0.24"
// ))
//
// Second best predictor (PdPS): \
// *IDR* AUROC:0.84 PRAUC:0.42, *nIDR* AUROC: 0.68, PRAUC: 0.08
//
// == Conclusion for the 2 convolutional layer model
// Both models benefited from the additional layer, especially the model using the
// block decomposition. In the end the sequence model still outperformed it.
// As a next step the introduction of other layers was tested (bilstm and
// transformer).
//
// == BiLSTM TODO
// Sequence:
// #align(center, table(
//   columns: 5,
//   "",      "BiLSTM IDR", "BiLSTM nIDR", "cnn2l IDR", "cnn2l nIDR",
//   "AUROC", [0.87],       [0.65],        "0.74",      "0.64",
//   "PRAUC", [0.66],       [0.71],        "0.26",      "0.05"
// ))
//
// == Cnn2l + BiLSTM TODO
// Sequence:
// #align(center, table(
//   columns: 5,
//   "",      "BiLSTM + Cnn2l IDR", "BiLSTM + Cnn2l nIDR", "cnn2l IDR", "cnn2l nIDR",
//   "AUROC", [0.87],               [0.65],                "0.74",      "0.64",
//   "PRAUC", [0.66],               [0.71],                "0.26",      "0.05"
// ))
//
// == Transformer TODO
// Sequence:
// #align(center, table(
//   columns: 5,
//   "",      "Transformer IDR", "Transformer nIDR", "cnn2l IDR", "cnn2l nIDR",
//   "AUROC", [0.87],            [0.65],             "0.74",      "0.64",
//   "PRAUC", [0.66],            [0.71],             "0.26",      "0.05"
// ))
//
// == Introducing RSA
// What helped PSPire to outperform the other predictors was the inclusion of
// structural predictions of the proteins. They divided did calculate the features
// for the IDRs and the ordered regions separately. For the ordered regions they
// only used the amino acids with an *Relative Surface Availability* greater than
// 0.25.
//
// As a cnn should be able to learn what the RSA is, it was supplied to the model
// as an additional feature.
//
// == Cnn2l + RSA
// #align(center, table(
//   columns: 5,
//   "",      "Cnn2l + RSA IDR", "Cnn2l + RSA nIDR", "cnn2l IDR", "cnn2l nIDR",
//   "AUROC", text(green)[0.79], text(red)[0.62],    "0.74",      "0.64",
//   "PRAUC", text(green)[0.37], [0.05],             "0.26",      "0.05"
// ))
//
// The performance gain was small. Other methods to feed the RSA value were
// evaluated.
//
// == Cnn2l + RSA linear
// #align(center, table(
//   columns: 5,
//   "",      "Cnn2l + RSA IDR", "Cnn2l + RSA nIDR", "cnn2l IDR", "cnn2l nIDR",
//   "AUROC", text(green)[0.75], [0.64],             "0.74",      "0.64",
//   "PRAUC", text(green)[0.34], text(green)[0.06],  "0.26",      "0.05"
// ))
//
// The performance gain was smaller than without the linear layer.
//
// == Cnn2l + RSA weighted
// #align(center, table(
//   columns: 5,
//   "",      "Cnn2l + RSA IDR", "Cnn2l + RSA nIDR", "cnn2l IDR", "cnn2l nIDR",
//   "AUROC", text(green)[0.76], text(green)[0.70],  "0.74",      "0.64",
//   "PRAUC", text(green)[0.38], text(green)[0.06],  "0.26",      "0.05"
// ))
//
// The method with the RSA values as weights outperformed the other methods. This
// model was subjected to further improvements.
//
// == Cnn2l + RSA weighted +
// #align(center, table(
//   columns: 5,
//   "",      "Cnn2l + RSA IDR", "Cnn2l + RSA nIDR", "cnn2l IDR", "cnn2l nIDR",
//   "AUROC", text(green)[0.76], text(green)[0.70],  "0.74",      "0.64",
//   "PRAUC", text(green)[0.38], text(green)[0.06],  "0.26",      "0.05"
// ))
//
// == Results in the context of the topic
//
// == Presentation of the results
//
// == What can be improved?
//
// == Structure of the thesis
