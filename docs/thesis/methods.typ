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
                                                 [xgboost @noauthor_xgboost_nodate],          "3.0.2",
  [DSSP @noauthor_pdb-redodssp_nodate],          "-",                                         "4.5.3",
  table.hline()
), caption: "Programs used in this work") <programs>

== Data Preparation

As the data sets that were used already undergone data preparation, only two additional
preparation steps were conducted. The first was to filter out all entries that had a letter
in the amino acid sequence that was not one of the twenty amino acids that were mapped. The second
step filtered out all sequences that were longer than 2700 residues long.

== Block Decomposition

The block decomposition algorithm by Martin Girard @noauthor_files_2024 was
used to factorize the protein sequences into blocks of a certain uniformity.
To be able to use the block decomposition as feature for neural networks
the output was transformed from a list of tuples containing the start and
end of a block to a sequence, where each position represents the block that
it is in. As the original algorithm did not provide information about the
content of a block, a label for each block was generated, that represented
the most common mapping or mappings in the block.

The mappings in @mappings were used for the block decompositions.

#figure(table(
  columns: 4,
  "Name",        "Description",                                                                       "Source", "Mapping",
  "APNA",        "Categorization into aliphatic, positive, negative and aromatic",                    "",       [0: A, C, G, H, I, L, M, N, P, Q, S, T, V \
  1: D, E \
  2: K, R \
  3: F, W, Y],
  "RG",          "RG-Motif",                                                                          "",       [0: Other \
  1: R, G],
  "RG2",         "Significantly more abundant in phase separating proteins that contain an RG-Motif", "",       [0: Other \
  1: D, E, F, G, I, K, M, N, R, Y],
  "IDR",         "Amino acids that are more common in IDRs",                                          "",       [0: F, I, L, M, V, W, Y \
  1: A, C, N, T \
  2: D, E, G, H, K, P, Q, R, S],
  "MM5",         "Most Meaningful grouping with five groups",                                         "",       [0: A, G, P, S, T \
  1: C \
  2: D, E, H, K, N, Q, R \
  3: F, L, M, V, Y \
  4: W],
  "PiPi G",      "Categorization after PiPi",                                                         "",       [0: A, C, G, I, K, L, M, P, S, T, V \
  1: D, E, N, Q, R \
  2: F, H, W, Y],
  "PiPi F",      "Pi Pi ",                                                                            "",       [0: C, I, K, L, M, P, T, V \
  1: A, D, E, H, N, Q, S, W \
  2: F, G, R, Y ],
  table.hline()
), caption: "Mappings used by the block decomposition algorithm") <mappings>

== Models

The first model created was a simple one layer @cnn. It was meant as a test if
neural networks are able to learn how to predict @llps from the block
decomposition or the sequence of proteins. The one layer @cnn:pl were tested on all
datasets.

This model had the following structure:

#cetz.canvas({
  import cetz.draw: *
  grid(
    (0, 0),
    (0.8, 8),
    step: 0.8
  )
  content((0.4, 0.3), box(fill: white, [$X_1$]))
  content((0.4, 1.1), box(fill: white, [$X_1$]))
})

To start the development of a new @llps predictor a few more models were created to
see which might suit the task best.

#figure(diagram(
  node-fill: grey,
  spacing: (3.5em, 2.5em),
  node((0, 0), [Start], name: <a>),

  edge(<a>, <b>, "-"),
  node((-1.5, 1), [Block Decomposition], name: <b>),

  edge(<b>, <d>, "-"),
  node((-2.5, 2), [CNN 1L], name: <d>),

  edge(<b>, <e>, "-"),
  node((-1.5, 2), [CNN 2L], name: <e>),

  edge(<b>, <f>, "-"),
  node((-0.5, 2), [XGBoost], name: <f>),

  edge(<a>, <c>),
  node((1.5, 1), [Sequence], name: <c>),

  edge(<c>, <g>, "-"),
  node((0.5, 2), [CNN 1L], name: <g>),

  edge(<c>, <h>, "-"),
  node((1.5, 3), [CNN 3L], name: <h>),

  edge(<c>, <i>, "-"),
  node((2.5, 2), [BLSTM], name: <i>),

  edge(<c>, <j>, "=>"),
  node((0.5, 3), [CNN2L], name: <j>),

  edge(<c>, <k>, "-"),
  node((2.5, 3), [Transformer], name: <k>),
), caption: [Visualization of finding the best @ml model. ])

#pagebreak()
