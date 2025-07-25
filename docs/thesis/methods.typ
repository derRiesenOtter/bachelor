#import "@preview/glossy:0.8.0": *
#import "state.typ": bib_state
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import "@preview/cetz:0.4.0"
#context bib_state.get()
#show: init-glossary.with(yaml("glossary.yaml"))
#let grey = luma(90%)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

= Methods

== Use of Artificial Intelligence Tools

During the course of this work, AI assistance was employed to support various
aspects of the writing and development process. Specifically, ChatGPT was used
for translating text passages, debugging code, and generating suggestions for
rewriting or rephrasing sentences to improve clarity and structure.
Additionally, it was consulted to retrieve standard parameter values and
practices commonly used in machine learning models. All outputs were carefully
reviewed and adapted to ensure accuracy, relevance, and alignment with the
goals of this thesis.

== Programs

A variety of programs and packages were used during this work. They are listed
in @programs. All computations were conducted on an Apple MacBook Pro with an
ARM-based, unbinned M1 Pro processor and 16 GB of RAM.
#figure(table(
  columns: 3,
  "Program",                                     "Package",                                   "Version",
  table.cell(rowspan: 11, "Python", fill: none), "-",                                         "3.13.3",
                                                 [numpy @harris_array_2020],                  "2.2.6",
                                                 [bio @cock_biopython_2009],                  "1.8.0",
                                                 [pandas @mckinney_data_2010],                "2.2.3",
                                                 [matplotlib @hunter_matplotlib_2007],        "3.10.3",
                                                 [requests],                                  "2.32.3",
                                                 [scikit-learn @pedregosa_scikit-learn_2011], "1.6.1",
                                                 [seaborn @waskom_seaborn_2021],              "0.13.2",
                                                 [torch @noauthor_pytorch_nodate],            "2.7.0",
                                                 [xgboost @chen_xgboost_2016],                "3.0.2",
                                                 [captum @kokhlikyan_captum_2020],            "0.8.0",
  [DSSP @noauthor_pdb-redodssp_nodate],          "-",                                         "4.5.3",
  table.hline()
), caption: "Programs used in this work.") <programs>

== Data Preparation

Protein sequences, protein structures and @ptm::pl were all downloaded using
the Python requests module.

As the datasets that were used already undergone data preparation, only three
additional preparation steps were conducted. The first was to filter out all
entries that had a letter in the amino acid sequence that was not one of the
twenty amino acids that were mapped. The second step filtered out all sequences
that were longer than 2700 residues. The third step was to pad all sequences
to a length of 2700, as @cnn::pl require input vectors of the same length. This
length was chosen due to comparability as PSPire uses the same maximum length.

The split used for the PPMC-lab dataset was 80% of the data for the training
set and 20% of the data for the testing set. For all models that incorporate
@rsa values the datasets were cleansed of entries for which no structure data
was available on AlphaFold. The length distribution of the filtered datasets
was visualized to assess the impact of padding. The @idr labels were inherited
for the PSPire dataset from their source. For the PPMC-lab dataset all proteins
that had a disorder fraction of zero were labeled as non-@idr proteins.

=== Block Decomposition

The block decomposition algorithm by Martin Girard @noauthor_files_2024 was
used to factorize the protein sequences into blocks of a certain uniformity.
To be able to use the block decomposition as a feature for @nn::pl
the output was transformed from a list of tuples containing the start and
end of a block to a sequence, where each position represents the block that
it is in. As the original algorithm did not provide information about the
content of a block, a label for each block was generated, that represented
the most common mapping or mappings in the block.

The label was derived as follows. After creating the block decomposition every
block was checked if it consisted to more than 66% of the same group. If it did
the groups mapping number was assigned as label to this block. If it did not, it
was tested if more than 66% of the block were from two groups. In that case
the two group numbers were concatenated, with the first one corresponding to the
larger group and used as the label for the block. If neither was the case, the
block was given a number that represented no group as a label.

The mappings that were used as well as the source and a short description are
shown in @mappings.

#figure(table(
  columns: 4,
  "Name",        "Description",                                                                       "Source",                     "Mapping",
  "APNA",        "Categorization into aliphatic, positive, negative and aromatic",                    [General],                    [0: A, C, G, H, I, L, M, N, P, Q, S, T, V \
  1: D, E \
  2: K, R \
  3: F, W, Y],
  "RG",          "RG-Motif",                                                                          [@chong_rggrg_2018],          [0: Other \
  1: R, G],
  "RG2",         "Significantly more abundant in phase separating proteins that contain an RG-Motif", [Internal],                   [0: Other \
  1: D, E, F, G, I, K, M, N, R, Y],
  "IDR",         "Amino acids that are more common in IDRs",                                          [@campen_top-idp-scale_2008], [0: F, I, L, M, V, W, Y \
  1: A, C, N, T \
  2: D, E, G, H, K, P, Q, R, S],
  "MM5",         "Most Meaningful grouping with five groups",                                         [@cannata_simplifying_2002],  [0: A, G, P, S, T \
  1: C \
  2: D, E, H, K, N, Q, R \
  3: F, L, M, V, Y \
  4: W],
  "PiPi G",      "Categorization after PiPi",                                                         [@vernon_pi-pi_2018],         [0: A, C, G, I, K, L, M, P, S, T, V \
  1: D, E, N, Q, R \
  2: F, H, W, Y],
  "PiPi F",      "Pi Pi ",                                                                            [@vernon_pi-pi_2018],         [0: C, I, K, L, M, P, T, V \
  1: A, D, E, H, N, Q, S, W \
  2: F, G, R, Y ],
  table.hline()
), caption: "Mappings used by the block decomposition algorithm.") <mappings>

== Evaluation Metrics

In statistics there are several evaluation metrics for binary classification.
The most common will be briefly covered in this section. Most of them use
the four terms @tp, @tn, @fp and @fn for their calculation. The @tp and @tn
values are the values the predictor got right in each category. The @fp and
@fn values describe the values that are predicted to be in the wrong category.
@rainio_evaluation_2024

The Accuracy describes the fraction of all correctly predicted samples, see @acc.
$ "Accuracy" = frac("TP" + "TN", "TP" + "TN" + "FP" + "FN") $ <acc>

The Recall describes the fraction of data with the positive label that was identified
as such, see @rec.

$ "Recall" = frac("TP", "TP" + "FN") $ <rec>

The Specificity describes the fraction of data with the negative label that was
identified as such, see @spe.

$ "Specificity" = frac("TN", "TN" + "FP") $ <spe>

The Precision describes the fraction of the data that has a correctly
predicted positive label in all predicted positive samples, see @pre.

$ "Precision" = frac("TP", "TP" + "FP") $ <pre>

While these values are all dependent on the threshold, a value that decides the
probability level above a value gets the positive label, there are also two
important metrics that do not depend on it. They are both curves that can be
evaluated by visual inspection as well as the @auc value. The @roc plots the
recall against the false positive rate, which is equal to one minus
specificity. The @prc plots the precision against the recall. @auc values
lie between zero and one. A higher @auc value is better.
While a @auc of 0.5 for the @roc means that the predictions of the model are
not better than random guessing, the interpretation of the @prc@auc depends on
the positive class frequency. A @prc@auc value of the positive class frequency
would equal random guessing. The usage of @prc@auc values is especially important
if imbalanced datasets are used, where there are more negative samples than positives.
Both metrics combined are a sufficient and threshold independent way to compare
model performance. @saito_precision-recall_2015

== Model Selection and Optimization

The process of selecting and testing different models during this work can be
divided into three phases. The first phase was about testing if @nn::pl,
especially @cnn::pl are capable of predicting @llps and comparing the block
decomposition to the raw sequence as input. The second phase revolved around
testing other @nn architectures that are more complex than the models from the
previous phase and testing the block decomposition as input for a @ml model.
The third phase focused on enhancing the best model with additional features
and optimizations. \
As the PSPire dataset proved to be challenging and
provided comparability towards many other @llps predictors it was chosen as the
main dataset for deciding which model or optimization to keep. Therefore, not
all models were run on all datasets. While the PhasePred dataset was tested in
the beginning of this work, it was abandoned due to the similarity to the
PSPire dataset and lack of time. Therefore, no results of this dataset are part
of this work. \
During all runs a random seed of 13 was set to achieve reproducibility. The
training data loaders were set to shuffle, while the validation data loaders
were not. The batch size was set depending on the complexity of the model and
the size of the data set, these values can be found in the raw data, see
appendix. All models, excluding the XGBoost model, used cross entropy loss for
the loss function and considered the distribution of positive and negative
values using class weights. The ADAM optimizer @kingma_adam_2017 was used in
all models except the XGBoost, as it requires less manual optimization than
stochastic gradient descent. The learning rate and decay was adjusted per model
and can be found in the raw data as well, see appendix. The models will be
evaluated with the @auc values from the @roc and @prc. \
The evaluation will
mostly be split into proteins that contain @idr::pl and proteins that do not
contain @idr::pl for comparability with the results of the PSPire. Testing other
splits like the driver and client split from the PPMC-lab dataset as well as
the split into partner-dependent and self-assembling proteins was only touched
briefly and was not included in this work.

During this development @bn was used @noauthor_batchnorm1d_nodate for
smoother training as well as @do for reducing overfitting. An overview
of the tested models and optimizations is shown in @journey. The following
sections will cover each phase and go into detail about what models were used,
the inputs as well as the parameters.

#let g = rgb(80, 150, 200, 100)
#let y = rgb(180, 230, 0, 100)
#let f = rgb(240, 100, 0, 100)
#figure(diagram(node-corner-radius: 2pt, spacing: (2.6em, 2.0em), node((0, 0), [Start], fill: g, name: <a>), edge(<a>, <b>, "-"), node((-1.5, 1), [Block Decomposition], fill: g, name: <b>), edge(<b>, <d>, "-"), node((-2.5, 2), [XGBoost], fill: y, name: <d>), edge(<b>, <e>, "-"), node((-1.5, 2), [1L CNN], fill: g, name: <e>), edge(<b>, <f>, "-"), node((-0.5, 2), [2L CNN], fill: g, name: <f>), edge(<a>, <c>, "=>"), node((1.5, 1), [Sequence], fill: g, name: <c>), edge(<c>, <g>, "-"), node((0.5, 2), [1L CNN], fill: g, name: <g>), edge(<c>, <h>, "-"), node((1.5, 3), [3L CNN], fill: y, name: <h>), edge(<c>, <i>, "-"), node((2.5, 2), [@bilstm], fill: y, name: <i>), edge(<c>, <j>, "=>"), node((0.5, 3), [2L CNN], fill: g, name: <j>), edge(<c>, <k>, "-"), node((2.5, 3), [Transformer], fill: y, name: <k>), node((0.5, 5), [@bn], fill: f, name: <m>), node((-0.5, 5), [RSA Weights], fill: f, name: <o>), node((-1.5, 5), [RSA], fill: f, name: <p>), node((0, 4), [Split @idr
non-@idr \
\+ \
higher @do], fill: f, name: <q>), node((1.5, 5), [@ptm], fill: f, name: <r>), node((0, 6), [Final Model], fill: f, name: <s>), edge(<j>, <q>, "=>"), edge(<q>, <o>, "=>"), edge(<q>, <p>), edge(<q>, <o>, "=>"), edge(<q>, <m>, "=>"), edge(<q>, <r>, "=>"), edge(<r>, <s>, "=>"), edge(<m>, <s>, "=>"), edge(<o>, <s>, "=>")), caption: [Overview of the models
tested during this work. The path to the final model is shown via the double arrows.
The phases are represented by the fill of the nodes. Phase one is colored blue, phase two is colored green and phase three is colored orange.]) <journey>

=== Phase One

As there have been no previous @llps predictors that relied on @nn::pl, the first
step was to create simple models to see if @cnn::pl are capable of this task.
Throughout this work the 1-dimensional versions of @cl::pl, @mpl::pl and
@ampl::pl were used. Another goal of these simple models was to test if the
block decomposition proves to be beneficial for the prediction over using only
the sequence as input. Two models were created for each input. A one Layer @cnn
and a two layer @cnn. The basic models only consisted of an embedding layer, @cl::pl
followed by the @relu activation functions and @mpl::pl and ended in an @ampl,
@do and a @fcl.

==== 1 Layer Convolutional Neural Network

The architecture of the one layer @cnn for the sequence based approach is shown in
@1lcnn. The values for the @oc, @ed, @ks, @st and @pd are given in @par_1lcnn.

#figure(table(
  align: (center, center, center, center),
  columns: 9,
  table.cell(rowspan: 2)[Input], table.vline(stroke: 0.5pt),  table.cell(rowspan: 2)[@ed], table.vline(stroke: 0.5pt), table.cell(colspan: 4)[@cl],                                                                                        table.vline(stroke: 0.5pt),
                                 table.cell(colspan: 2)[@mpl],                             table.vline(stroke: 0.5pt), table.cell(rowspan: 2)[@do], table.cell(fill: none)[@oc], table.cell(fill: none)[@ks], table.cell(fill: none)[@pd], table.cell(fill: none)[@st],
  table.cell(fill: none)[@ks],   table.cell(fill: none)[@st], table.hline(stroke: .5pt),   [Sequence],                                              [10],                        [70],                        [10],                        [2],
  [2],                           [2],                         [2],                         [0.3],                      [Block Decomposition],       [3],                         [70],                        [10],                        [2],
  [2],                           [2],                         [2],                         [0.3],                      table.hline()
), caption: [Parameters for 1 Layer CNNs.]) <par_1lcnn>

The model created for the Block Decomposition input already had 14 channels, two for
every mapping. Each channel got its own embedding dimension. The architecture of the
one layer @cnn is visualized in @1lcnn.

#figure(cetz.canvas({
  import cetz.draw: *
  let heading = 4.5
  let dim = -4.5
  let dist = 2.9
  let start = 0

  content((0, heading), [Input])
  content((0, 0), text(size: 7pt)[MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTD...], angle: 270deg)
  content((0, dim), [2700 x 1])
  content((start + 0.55 * dist, 0), [$==>$])

  let start = start + dist
  content((start, heading), [Embedding])
  let bottom = -3.5
  let top = 3.5
  let width = 0.3

  for z in (0, 0.1, 0.2) {
    rect((start - z, bottom - z), (start - z + width, top - z), fill: white)
    let x = bottom
    while x < top {
      line((start - z, x - z), (start - z + width, x - z))
      x = x + 0.1
    }
  }
  content((start, dim), [2700 x 10])
  content((start + 0.55 * dist, 0), [$==>$])

  let start = start + dist
  content((start, heading), [@cl])
  let bottom = -3.3
  let top = 3.3
  let width = 0.3

  for z in (0, 0.1, 0.2, 0.3, 0.4) {
    rect((start - z, bottom - z), (start - z + width, top - z), fill: white)
    let x = bottom
    while x < top {
      line((start - z, x - z), (start - z + width, x - z))
      x = x + 0.1
    }
  }
  content((start, dim), [2691 x 70])
  content((start + 0.5 * dist, 0.8), [@relu])
  content((start + 0.55 * dist, 0), [$==>$])

  let start = start + dist
  content((start, heading), [@mpl])
  let bottom = -1.5
  let top = 1.5
  let width = 0.3

  for z in (0, 0.1, 0.2, 0.3, 0.4) {
    rect((start - z, bottom - z), (start - z + width, top - z), fill: white)
    let x = bottom
    while x < top {
      line((start - z, x - z), (start - z + width, x - z))
      x = x + 0.1
    }
  }
  content((start, dim), [1347 x 70])

  content((start + 0.55 * dist, 0), [$==>$])
  let start = start + dist
  content((start, heading), [@ampl])
  let bottom = 0
  let top = 0.1
  let width = 0.3

  for z in (0, 0.1, 0.2, 0.3, 0.4) {
    rect((start - z, bottom - z), (start - z + width, top - z), fill: white)
    let x = bottom
    while x < top {
      line((start - z, x - z), (start - z + width, x - z))
      x = x + 0.1
    }
  }
  content((start, dim), [1 x 70])

  content((start + 0.55 * dist, 0), [$==>$])
  content((start + 0.5 * dist, 0.8), [@do])

  let start = start + dist
  content((start, heading), [@fcl])
  let bottom = -0.1
  let top = 0.1
  let width = 0.3

  rect((start, bottom), (start + width, top), fill: white)
  let x = bottom
  while x < top {
    line((start, x), (start + width, x))
    x = x + 0.1
  }
  content((start, dim), [2])

}), caption: [Visualization of the 1 Layer CNN used with the sequence as input.]) <1lcnn>

==== 2 Layer Convolutional Neural Network
The parameters of the 2 layer models are summarized in @par_2lcnn.

#figure(table(
  align: (center, center, center, center),
  columns: 13,
  table.cell(rowspan: 2)[Input], table.vline(stroke: 0.5pt), table.cell(rowspan: 2)[@ed], table.vline(stroke: 0.5pt),  table.cell(colspan: 4)[@cl 1],                                                                                      table.vline(stroke: 0.5pt),  table.cell(colspan: 2)[@mpl],                             table.vline(stroke: 0.5pt),  table.cell(colspan: 4)[@cl 2],
                                 table.vline(stroke: 0.5pt),                              table.cell(rowspan: 2)[@do], table.cell(fill: none)[@oc], table.cell(fill: none)[@ks], table.cell(fill: none)[@pd], table.cell(fill: none)[@st], table.cell(fill: none)[@ks], table.cell(fill: none)[@st], table.cell(fill: none)[@oc], table.cell(fill: none)[@ks], table.cell(fill: none)[@pd],
  table.cell(fill: none)[@st],   table.hline(stroke: .5pt),  [Sequence],                                               [10],                        [70],                        [10],                        [2],                         [2],                         [2],                         [2],                         [140],                       [10],
  [2],                           [2],                        [0.3],                       [Block Decomposition],       [3],                         [70],                        [10],                        [2],                         [2],                         [2],                         [2],                         [140],                       [10],
  [2],                           [2],                        [0.3],                       table.hline()
), caption: [Parameters for 2 Layer CNNs.]) <par_2lcnn>

It only added one additional @cl to see if it benefits the model, see the visualization in @2lcnn.

#figure(cetz.canvas({
  import cetz.draw: *
  let heading = 4.5
  let dim = -4.5
  let dist = 2.5
  let start = 0

  content((0, heading), [Input])
  content((0, 0), text(size: 7pt)[MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTD...], angle: 270deg)
  content((0, dim), [2700 x 1])
  content((start + 0.55 * dist, 0), [$==>$])

  let start = start + dist
  content((start, heading), [Embedding])
  let bottom = -3.5
  let top = 3.5
  let width = 0.3

  for z in (0, 0.1, 0.2) {
    rect((start - z, bottom - z), (start - z + width, top - z), fill: white)
    let x = bottom
    while x < top {
      line((start - z, x - z), (start - z + width, x - z))
      x = x + 0.1
    }
  }
  content((start, dim), [2700 x 10])
  content((start + 0.55 * dist, 0), [$==>$])

  let start = start + dist
  content((start, heading), [@cl])
  let bottom = -3.3
  let top = 3.3
  let width = 0.3

  for z in (0, 0.1, 0.2, 0.3, 0.4) {
    rect((start - z, bottom - z), (start - z + width, top - z), fill: white)
    let x = bottom
    while x < top {
      line((start - z, x - z), (start - z + width, x - z))
      x = x + 0.1
    }
  }
  content((start, dim), [2691 x 70])
  content((start + 0.55 * dist, 0), [$==>$])
  content((start + 0.5 * dist, 0.8), [@relu])

  let start = start + dist
  content((start, heading), [@mpl])
  let bottom = -1.5
  let top = 1.5
  let width = 0.3

  for z in (0, 0.1, 0.2, 0.3, 0.4) {
    rect((start - z, bottom - z), (start - z + width, top - z), fill: white)
    let x = bottom
    while x < top {
      line((start - z, x - z), (start - z + width, x - z))
      x = x + 0.1
    }
  }
  content((start, dim), [1347 x 70])

  content((start + 0.55 * dist, 0), [$==>$])
  let start = start + dist
  content((start, heading), [@cl 2])
  let bottom = -1.5
  let top = 1.5
  let width = 0.3

  for z in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6) {
    rect((start - z, bottom - z), (start - z + width, top - z), fill: white)
    let x = bottom
    while x < top {
      line((start - z, x - z), (start - z + width, x - z))
      x = x + 0.1
    }
  }
  content((start, dim), [1338 x 140])
  content((start + 0.5 * dist, 0.8), [@relu])
  content((start + 0.55 * dist, 0), [$==>$])
  let start = start + dist
  content((start, heading), [@ampl])
  let bottom = 0
  let top = 0.1
  let width = 0.3

  for z in (0, 0.1, 0.2, 0.3, 0.4) {
    rect((start - z, bottom - z), (start - z + width, top - z), fill: white)
    let x = bottom
    while x < top {
      line((start - z, x - z), (start - z + width, x - z))
      x = x + 0.1
    }
  }
  content((start, dim), [1 x 70])

  content((start + 0.55 * dist, 0), [$==>$])
  content((start + 0.5 * dist, 0.8), [@do])

  let start = start + dist
  content((start, heading), [@fcl])
  let bottom = -0.1
  let top = 0.1
  let width = 0.3

  rect((start, bottom), (start + width, top), fill: white)
  let x = bottom
  while x < top {
    line((start, x), (start + width, x))
    x = x + 0.1
  }
  content((start, dim), [2])

}), caption: [Visualization of the 2 Layer CNN used with the sequence as input.]) <2lcnn>

=== The second set of Models

After the first set of models was tested, it was decided to discard the idea to
use the block decomposition with @nn::pl as they were outperformed by the models
that used the raw sequence as input, see @first_set. Instead the block
decomposition was used to create a model based on the XGBoost algorithm while the
@nn models based on the raw sequence were focused.

==== XGBoost

To be able to use the block decomposition output for a model like XGBoost, the output
had to be adapted again. For every channel representing a mapping the fraction of each
label was calculated and used as an input. A simple XGBoost model was created with the
parameters in @par_xgboost, for all parameters see appendix.

#figure(table(
  columns: 5,
  align: (left, center, center, center, left),
  [Tree Method],     [Learning Rate], [Estimators], [Max Depth], [Evaluation Metric],
  [Histogram based], [0.05],          [1000],       [4],         [LogLoss],
  table.hline()
), caption: [Parameters of the XGBoost model.]) <par_xgboost>

==== 3 Layer Convolutional Neural Network

This model added one additional @cl, @relu and @mpl to the two layer @cnn.
The parameters are shown in @par_3lcnn.

#figure(table(
  align: (left, center, center, center),
  columns: 17,
  table.cell(rowspan: 2)[Input], table.vline(stroke: 0.5pt),  table.cell(rowspan: 2)[@ed], table.vline(stroke: 0.5pt), table.cell(colspan: 4)[@cl 1],                                                                                    table.vline(stroke: 0.5pt),  table.cell(colspan: 2)[@mpl 1 / 2],                       table.vline(stroke: 0.5pt),  table.cell(colspan: 4)[@cl 2],                                                                                      table.vline(stroke: 0.5pt),
                                 table.cell(colspan: 4)[@cl 3],                                                                                   table.cell(rowspan: 2)[@do], table.cell(fill: none)[@oc], table.cell(fill: none)[@ks], table.cell(fill: none)[@pd], table.cell(fill: none)[@st], table.cell(fill: none)[@ks], table.cell(fill: none)[@st], table.cell(fill: none)[@oc], table.cell(fill: none)[@ks], table.cell(fill: none)[@pd], table.cell(fill: none)[@st], table.cell(fill: none)[@oc],
  table.cell(fill: none)[@ks],   table.cell(fill: none)[@pd], table.cell(fill: none)[@st], table.vline(stroke: 0.5pt), table.hline(stroke: .5pt),                              [Sequence],                  [10],                        [70],                        [10],                        [2],                         [2],                         [2],                         [2],                         [140],                       [10],                        [2],
  [2],                           [210],                       [10],                        [2],                        [2],                       [0.3],                       table.hline()
), caption: [Parameters for 3 Layer CNNs.]) <par_3lcnn>

==== Bidirectional Long Short Term Memory Model

A very basic @bilstm model was created with the parameters shown in @par_bilstm.
The input was embedded and subjected to the @lstm layer. @do and a @fcl followed.

#figure(table(
  align: (center, center, center, center),
  columns: 4,
  [@ed],         [Hidden Dimensions], [Layers], [@do],
  [12],          [3],                 [4],      [0.3],
  table.hline()
), caption: [Parameters of the BILSTM model. ]) <par_bilstm>

==== Transformer

A basic transformer model was created with the parameters shown in @par_transformer.
The model did integrate a positional encoding.

#figure(table(
  align: (center, center, center, center),
  columns: 6,
  [@ed],         [Hidden Dimensions], [Heads], [Feed Forward Dimensions], [Layers], [@do],
  [12],          [3],                 [4],     [256],                     [2],      [0.3],
  table.hline()
), caption: [Parameters of the transformer model.]) <par_transformer>

=== Optimizing the Two Layer Convolutional Neural Network <optimize>

As the second set of models did not provide a better model than the two layer
@cnn, see @secon_set, they were discarded. The work was then focused on
improving the two layer @cnn. The first step was to double the dropout rate and
split the model into one that is responsible for learning @llps prediction for
@idr proteins and one for non-@idr proteins. The dropout rate was doubled due
to the models tendency of overfitting. The dataset was split to reduce the bias
towards @idr proteins and enable the model to learn important features for predicting
non-@idr proteins. \
In the second step different optimizations were tested independently. @bn was
added to smoothen the training process. @rsa values were added as additional
information for the model, both as an additional feature sequence and as a weight
vector for the embedded sequence. To obtain the @rsa values the structure files
of all proteins were downloaded from AlphaFold. Using the tool DSSP
@noauthor_pdb-redodssp_nodate the @rsa per amino acid was calculated. \
Lastly the integration of @ptm::pl was tested. In contrast to previous
studies, the inclusion of @ptm::pl was not limited to phosphorylation sites. While
phosphorylation was the focus of early studies on @llps other @ptm::pl like
acetylation or ubiquitination could also regulate @llps @li_post-translational_2022.
They were downloaded from UniProt. As there are many different @ptm::pl, a
mapping for similar @ptm::pl was created. It searched the @ptm description for
specific strings. @ptm_prep describes this mapping. \
These optimizations were performed on both the PSPire dataset and the PPMC-lab
dataset. To test how these optimizations interact with each other, they were
combined in every meaningful way. This last test was only conducted on the
PSPire dataset.

#figure(table(
  columns: 2,
  [Group],            [Search Patterns],
  [Phosphorylation],  [phospho],
  [Acetylation],      [acetyl],
  [Methylation],      [methyl],
  [Ubiquitin-like],   [ubiquitin, sumo, nedd8, isg15],
  [Adp-Ribosylation], [adp-ribosyl, polyadp],
  [glycosylation],    [glcnac, galnac, glycos, xyl, fuc, man],
  [Oxidation],        [sulfoxide, hydroxy, oxid],
  [Other],            [-],
  table.hline()
), caption: [Mapping of PTMs.]) <ptm_prep>

The parameters for the final models are shown in @par_final_t.

#figure(table(
  align: (left, center, center, center, center, center, center, center, center, center, center, center, center),
  columns: 13,
  table.cell(rowspan: 2)[Input], table.vline(stroke: 0.5pt), table.cell(rowspan: 2)[@ed], table.vline(stroke: 0.5pt),  table.cell(colspan: 4)[@cl 1],                                                                                      table.vline(stroke: 0.5pt),  table.cell(colspan: 2)[@mpl],                             table.vline(stroke: 0.5pt),  table.cell(colspan: 4)[@cl 2],
                                 table.vline(stroke: 0.5pt),                              table.cell(rowspan: 2)[@do], table.cell(fill: none)[@oc], table.cell(fill: none)[@ks], table.cell(fill: none)[@pd], table.cell(fill: none)[@st], table.cell(fill: none)[@ks], table.cell(fill: none)[@st], table.cell(fill: none)[@oc], table.cell(fill: none)[@ks], table.cell(fill: none)[@pd],
  table.cell(fill: none)[@st],   table.hline(stroke: .5pt),  [Sequence],                                               [10],                        [70],                        [10],                        [2],                         [2],                         [2],                         [2],                         [140],                       [10],
  [2],                           [2],                        [0.6],                       table.hline()
), caption: [Parameters for the final two layer CNNs.]) <par_final_t>

As the optimizations affected both models differently, two different architectures
were the result. The architecture for the non-@idr model is shown in @final_model.
Due to visual reasons the step of embedding the @ptm values is not shown here.
In comparison the model for the @idr is missing the concatenation of the @ptm layers and the
@bn::pl.

#figure(cetz.canvas({
  import cetz.draw: *
  let heading = 4.5
  let dim = -4.5
  let dist = 1.9
  let start = 0

  content((0, heading), [Input])
  content((0, 0), text(size: 7pt)[MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTD...], angle: 270deg)
  content((0, dim), [2700 x 1])
  content((start + 0.45 * dist, 0), [$==>$])

  let start = start + dist
  content((start, heading), [Embedding])
  let bottom = -3.5
  let top = 3.5
  let width = 0.3

  for z in (0, 0.1, 0.2) {
    rect((start - z, bottom - z), (start - z + width, top - z), fill: white)
    let x = bottom
    while x < top {
      line((start - z, x - z), (start - z + width, x - z))
      x = x + 0.1
    }
  }
  content((start, dim), [2700 x 10])
  content((start + 0.45 * dist, 0), [$==>$])

  let start = start + dist
  content((start, heading), [@rsa])
  let bottom = -3.5
  let top = 3.5
  let width = 0.3

  for z in (0, 0.1, 0.2) {
    rect((start - z - .3, bottom - z), (start - z - .3 + width, top - z), fill: white)
    let x = bottom
    while x < top {
      line((start - z - .3, x - z), (start - z - .3 + width, x - z))
      x = x + 0.1
    }
  }

  content((start + .15, 0), [$dot$])

  rect((start + .3, bottom), (start + .3 + width, top), fill: white)
  let x = bottom
  while x < top {
    line((start + .3, x), (start + .3 + width, x))
    x = x + 0.1
  }

  content((start, dim), [2700 x 10])

  content((start + 0.5 * dist, 0), [$==>$])

  let start = start + dist
  content((start, heading), [@ptm])
  let bottom = -3.5
  let top = 3.5
  let width = 0.3

  for z in (0, 0.1, 0.2) {
    rect((start - z - .35, bottom - z), (start - z - .35 + width, top - z), fill: white)
    let x = bottom
    while x < top {
      line((start - z - .35, x - z), (start - z - .35 + width, x - z))
      x = x + 0.1
    }
  }

  content((start + .15, 0), [$+$])

  for z in (0, 0.1) {
    rect((start + .45 - z, bottom - z), (start + .45 - z + width, top - z), fill: white)
    let x = bottom
    while x < top {
      line((start + .45 - z, x - z), (start + .45 - z + width, x - z))
      x = x + 0.1
    }
  }
  content((start + 0.6 * dist, 0), [$==>$])

  content((start, dim), [2700 x 18])

  let start = start + dist
  content((start, heading), [@cl 1])
  let bottom = -3.3
  let top = 3.3
  let width = 0.3

  for z in (0, 0.1, 0.2, 0.3, 0.4) {
    rect((start - z, bottom - z), (start - z + width, top - z), fill: white)
    let x = bottom
    while x < top {
      line((start - z, x - z), (start - z + width, x - z))
      x = x + 0.1
    }
  }
  content((start, dim), [2691 x 70])

  content((start + 0.5 * dist, 0), [$==>$])
  content((start + 0.5 * dist, .8), [@bn, \
  @relu])

  let start = start + dist
  content((start, heading), [@mpl])
  let bottom = -1.5
  let top = 1.5
  let width = 0.3

  for z in (0, 0.1, 0.2, 0.3, 0.4) {
    rect((start - z, bottom - z), (start - z + width, top - z), fill: white)
    let x = bottom
    while x < top {
      line((start - z, x - z), (start - z + width, x - z))
      x = x + 0.1
    }
  }
  content((start, dim), [1347 x 70])

  content((start + 0.45 * dist, 0), [$==>$])

  let start = start + dist
  content((start, heading), [@cl 2])
  let bottom = -1.5
  let top = 1.5
  let width = 0.3

  for z in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6) {
    rect((start - z, bottom - z), (start - z + width, top - z), fill: white)
    let x = bottom
    while x < top {
      line((start - z, x - z), (start - z + width, x - z))
      x = x + 0.1
    }
  }
  content((start, dim), [1338 x 140])

  content((start + 0.45 * dist, 0), [$==>$])
  content((start + 0.5 * dist, 0.8), [@bn, \
  @relu])

  let start = start + dist
  content((start, heading), [@ampl])
  let bottom = 0
  let top = 0.1
  let width = 0.3

  for z in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6) {
    rect((start - z, bottom - z), (start - z + width, top - z), fill: white)
    let x = bottom
    while x < top {
      line((start - z, x - z), (start - z + width, x - z))
      x = x + 0.1
    }
  }
  content((start, dim), [1 x 140])

  content((start + 0.55 * dist, 0), [$==>$])
  content((start + 0.5 * dist, 0.55), [@do])

  let start = start + dist
  content((start, heading), [@fcl])
  let bottom = -0.1
  let top = 0.1
  let width = 0.3

  rect((start, bottom), (start + width, top), fill: white)
  let x = bottom
  while x < top {
    line((start, x), (start + width, x))
    x = x + 0.1
  }
  content((start, dim), [2])

}), caption: [Visualization of the final non-IDR model.]) <final_model>

== Evaluation of the models

The final models described in @optimize were used for evaluation. This means
one model for non-@idr proteins, see @final_model, and the sightly modified
version without @ptm::pl and @bn for @idr proteins.

The evaluation of the models was conducted on the PSPire dataset, the PPMC-lab
dataset as well as the catGranule 2.0 dataset. For the PSPire dataset and the
catGranule 2.0 dataset the models were trained and tested on the same data splits
as the models in the respective paper. This way the results are as comparable as possible.
As there are no other models that were trained on the PPMC-lab dataset and
no other @llps predictors were run on the test set used here, the evaluation
will only show the @auc values obtained by the final model. Only the two best scoring @llps
predictors from the PSPire paper were shown in the evaluation on the PSPire dataset.
For the catGranule 2.0 dataset all @llps predictors compared in its paper were
used.

Like in the PSPire paper, the @mlo datasets were used as additional evaluation
datasets. As the proteins in these are all potential @llps proteins they were used
as the positive testing set. The negative testing set of the PSPire paper was used
as the negative testing set for the @mlo::pl. The models trained on the PSPire
and PPMC-lab dataset were evaluated on the @mlo testing sets. One @mlo dataset
was also evaluated with the models trained on the PPMC-lab dataset and The negative
PPMC-lab test data set, to see if the results differ.

As the catGranule 2.0 dataset as well as the results in its paper do not differ
between @idr proteins and non-@idr proteins both in training and testing, a
modified version of the final non-@idr model was used for its evaluation. In
this modified version the split of @idr and non-@idr proteins was removed.

==== Visualization of Input Features

To investigate the influence of the sequence composition as input feature on
the model's predictions, the Python package Captum @kokhlikyan_captum_2020 was
used. Captum is a model interpretability library developed for PyTorch,
providing a variety of attribution methods that help identify which features
contribute most significantly to a prediction. For this work, selected proteins
were analyzed using saliency maps, which highlight important
positions or patterns in the input sequence that influenced the model's output.
These analyses were conducted to gain insights into the model's
decision-making process and to assess the biological plausibility of the
learned representations.

From the Dr.LLPS @mlo dataset twenty random relatively high scoring @llps
proteins were chosen. Ten @idr proteins and ten non-@idr proteins. The salinity
map was created for them using both final models trained on the PSPire dataset.
One of the @idr proteins and one of the non-@idr proteins was chosen for
further comparison. Their saliency maps were compared to their protein features
using the UniProt feature viewer. The two proteins were also submitted to the
catGranule 2.0 web tool, as it provides a @llps propensity score over the
sequence. These visualizations were also compared to the saliency maps. At last
the saliency maps from both the final @idr model and the final non-@idr model
of the chosen non-@idr protein were compared to each other, to see how the
regions that were most important for their predictions differed.

#pagebreak()
