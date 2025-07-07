#import "@preview/glossy:0.8.0": *
#import "state.typ": bib_state
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import "@preview/cetz:0.4.0"
#context bib_state.get()
#show: init-glossary.with(yaml("glossary.yaml"))
#let grey = luma(90%)

= Methods

== Programs

The programs used during this work are listed in @programs.
#figure(table(
  columns: 3,
  "Program",                                     "Package",                                   "Version",
  table.cell(rowspan: 10, "Python", fill: none), "-",                                         "3.13.3",
                                                 [numpy @harris_array_2020],                  "2.2.6",
                                                 [bio @cock_biopython_2009],                  "1.8.0",
                                                 [pandas @mckinney_data_2010],                "2.2.3",
                                                 [matplotlib @hunter_matplotlib_2007],        "3.10.3",
                                                 [requests ],                                 "2.32.3",
                                                 [scikit-learn @pedregosa_scikit-learn_2011], "1.6.1",
                                                 [seaborn @waskom_seaborn_2021],              "0.13.2",
                                                 [torch @noauthor_pytorch_nodate],            "2.7.0",
                                                 [xgboost @chen_xgboost_2016],                "3.0.2",
  [DSSP @noauthor_pdb-redodssp_nodate],          "-",                                         "4.5.3",
  table.hline()
), caption: "Programs used in this work") <programs>

== Data Preparation

As the datasets that were used already undergone data preparation, only three
additional preparation steps were conducted. The first was to filter out all
entries that had a letter in the amino acid sequence that was not one of the
twenty amino acids that were mapped. The second step filtered out all sequences
that were longer than 2700 residues. The third step was to pad all sequences
to a length of 2700, as @cnn::pl require input vectors of the same length. This
length was as it is the same length PSPire set as maximum length.

=== Block Decomposition

The block decomposition algorithm by Martin Girard @noauthor_files_2024 was
used to factorize the protein sequences into blocks of a certain uniformity.
To be able to use the block decomposition as a feature for @nn
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
block was given a number that represented a no group as a label.

The mappings in @mappings were used for the block decompositions.

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
  "MM5",         "Most Meaningful grouping with five groups",                                         [],                           [0: A, G, P, S, T \
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
), caption: "Mappings used by the block decomposition algorithm") <mappings>

== General Informations on running the models
All models were run on the PSPire dataset, as this was the main dataset for comparing
the models between each other. As there was no defined Test or Training set in the
PPMC-lab dataset a split of 80/20 was then used for training. A seed of 13 was used for
all models to make the results reproducible. The training data loaders were set to shuffle,
while the validation data loaders were not. The batch size was set depending on the complexity
of the model and the size of the data set, these values can be found in the appendix. All
models, excluding the XGBoost model, used cross entropy loss for the loss
function and considered the distribution of positive and negative values using
weights. The adam optimizer was used in all models except the XGBoost, as it requires
less manual optimization than stochastic gradient descent. TODO. The learning rate and
decay was adjusted per model and can be found in the appendix.

== The first set of Models
As there have been no previous @llps predictors that relied on @nn, the first step
was to create simple models to see if @cnn::pl are capable of this task. Throughout
this work the 1-dimensional versions of these @cl, the @mpl and the @ampl were used. A one layer
and a two layer @cnn were created. Another goal of these simple models was to test
if the block decomposition proves to be beneficial for the prediction over using only
the sequence as input. Two models were created for each input. A one Layer @cnn and a
two layer @cnn. The basic models only consisted of an embedding, @cl::pl followed by
the @relu activation function, a @mpl and an at the end an @ampl, @do and @fcl.

=== 1 Layer @cnn

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
), caption: [Parameters for 1 Layer @cnn::pl.]) <par_1lcnn>

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

}), caption: [Visualization of the 1 Layer @cnn used with the sequence as input.]) <1lcnn>

=== 2 Layer @cnn
The parameters of the 2 layer models are summarized in @par_2lcnn.

#figure(table(
  align: (center, center, center, center),
  columns: 13,
  table.cell(rowspan: 2)[Input], table.vline(stroke: 0.5pt), table.cell(rowspan: 2)[@ed], table.vline(stroke: 0.5pt),  table.cell(colspan: 4)[@cl 1],                                                                                      table.vline(stroke: 0.5pt),  table.cell(colspan: 2)[@mpl],                             table.vline(stroke: 0.5pt),  table.cell(colspan: 4)[@cl 2],
                                 table.vline(stroke: 0.5pt),                              table.cell(rowspan: 2)[@do], table.cell(fill: none)[@oc], table.cell(fill: none)[@ks], table.cell(fill: none)[@pd], table.cell(fill: none)[@st], table.cell(fill: none)[@ks], table.cell(fill: none)[@st], table.cell(fill: none)[@oc], table.cell(fill: none)[@ks], table.cell(fill: none)[@pd],
  table.cell(fill: none)[@st],   table.hline(stroke: .5pt),  [Sequence],                                               [10],                        [70],                        [10],                        [2],                         [2],                         [2],                         [2],                         [140],                       [10],
  [2],                           [2],                        [0.3],                       [Block Decomposition],       [3],                         [70],                        [10],                        [2],                         [2],                         [2],                         [2],                         [140],                       [10],
  [2],                           [2],                        [0.3],                       table.hline()
), caption: [Parameters for 2 Layer @cnn::pl.]) <par_2lcnn>

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

}), caption: [Visualization of the 2 Layer @cnn used with the sequence as input.]) <2lcnn>

== The second set of Models

After the first set of models was tested, it was decided to discard the idea to use
the block decomposition with @nn as they were outperformed by the models that used
the raw sequence as input, see . Instead the block decomposition was used to create
a model based on the XGBoost algorithm and the @nn models based on the raw sequence
were focused.

=== XGBoost

To be able to use the block decomposition output for a model like XGBoost, the output
had to be adapted again. For every channel representing a mapping the fraction of each
label was calculated and used as an input. A simple XGBoost model was created with the
parameters in @par_xgboost, for all parameters see appendix.

#figure(table(
  columns: 5,
  [Tree Method],     [Learning Rate], [Estimators], [Max Depth], [Evaluation Metric],
  [Histogram based], [0.05],          [1000],       [4],         [LogLoss],
  table.hline()
)) <par_xgboost>

=== 3 Layer @cnn

=== BLSTM

=== Transformer

== The final model
The final model also added @bn.

#figure(table(
  align: (center, center, center, center),
  columns: 13,
  table.cell(rowspan: 2)[Input], table.vline(stroke: 0.5pt), table.cell(rowspan: 2)[@ed], table.vline(stroke: 0.5pt),  table.cell(colspan: 4)[@cl 1],                                                                                      table.vline(stroke: 0.5pt),  table.cell(colspan: 2)[@mpl],                             table.vline(stroke: 0.5pt),  table.cell(colspan: 4)[@cl 2],
                                 table.vline(stroke: 0.5pt),                              table.cell(rowspan: 2)[@do], table.cell(fill: none)[@oc], table.cell(fill: none)[@ks], table.cell(fill: none)[@pd], table.cell(fill: none)[@st], table.cell(fill: none)[@ks], table.cell(fill: none)[@st], table.cell(fill: none)[@oc], table.cell(fill: none)[@ks], table.cell(fill: none)[@pd],
  table.cell(fill: none)[@st],   table.hline(stroke: .5pt),  [Sequence],                                               [10],                        [70],                        [10],                        [2],                         [2],                         [2],                         [2],                         [140],                       [10],
  [2],                           [2],                        [0.6],                       table.hline()
), caption: [Parameters for 2 Layer @cnn::pl.]) <par_final>

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

}), caption: [Visualization of the final model.]) <final_model>

The process of choosing the final model for this work is illustrated in @journey.

#figure(diagram(node-fill: grey, spacing: (3.5em, 2.5em), node((0, 0), [Start], name: <a>), edge(<a>, <b>, "-"), node((-1.5, 1), [Block Decomposition], name: <b>), edge(<b>, <d>, "-"), node((-2.5, 2), [XGBoost], name: <d>), edge(<b>, <e>, "-"), node((-1.5, 2), [1L CNN], name: <e>), edge(<b>, <f>, "-"), node((-0.5, 2), [2L CNN], name: <f>), edge(<a>, <c>, "=>"), node((1.5, 1), [Sequence], name: <c>), edge(<c>, <g>, "-"), node((0.5, 2), [1L CNN], name: <g>), edge(<c>, <h>, "-"), node((1.5, 3), [3L CNN], name: <h>), edge(<c>, <i>, "-"), node((2.5, 2), [BLSTM], name: <i>), edge(<c>, <j>, "=>"), node((0.5, 3), [2L CNN], name: <j>), edge(<c>, <k>, "-"), node((2.5, 3), [Transformer], name: <k>), edge(<d>, <l>, "-"), node((-2.5, 3), [RSA], name: <l>), node((0.5, 4), [Batch Normalization], name: <m>), node((1.5, 4), [Attention], name: <n>), node((-0.5, 4), [RSA Weights], name: <o>), node((-1.5, 4), [RSA], name: <p>), node((0, 5), [Split @idr\
and non-@idr], name: <q>), node((0, 6), [@ptm], name: <r>), edge(<j>, <m>, "=>"), edge(<j>, <n>), edge(<j>, <o>, "=>"), edge(<j>, <p>), edge(<o>, <q>, "=>"), edge(<m>, <q>, "=>"), edge(<q>, <r>, "=>")), caption: [Overview of the models
created during this work. The path to the final model is shown via the double arrows.]) <journey>

#pagebreak()
