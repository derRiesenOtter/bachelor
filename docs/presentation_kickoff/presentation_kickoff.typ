#import "@preview/touying:0.6.1": *
#import themes.simple: *

#show: simple-theme.with(aspect-ratio: "16-9")
#show "Rust": name => box[
  #box(image("figures/rustacean.png"), height: 0.6em)
  #name
]
#show "Python": name => box[
  #box(image("figures/python.png"), height: 0.6em)
  #name
]

= Kick Off Meeting: Bachelor thesis of Robin Ender
#place(top + left, image("figures/cbdm_logo.png", width: 40%))
#place(top + right, image("figures/th_bingen_logo.jpg", width: 20%))
Date: _2025-05-08_ \
Attendees: _Robin Ender, Asis Hallab, Eric Schumbera_

== The Task / Scientific Question
#show: magic.bibliography-as-footnote.with(bibliography("bachelor.bib", title: none))
#v(30%)
#align(center)[
  Are *multiple block decompositions* of protein sequences able to predict *Phase
  Separation Propensity*?
]

== How to answer this question?
+ rewriting the current block decomposition algorithm to get a more meaningful
  output for Deep Learning Models
+ acquiring and processing curated training / test data
+ find meaningful mappings for running the block decomposition
+ run the block decomposition on the data
+ train Deep Learning Models with the output
+ compare the capabilities to other Phase Separation Predictors

== Background
- phase separation is mainly driven by two forces: @hou_machine_2024
  - protein-protein or protein-RNA interaction domains
  - interactions between intrinsically disordered regions
- current predictors use machine learning models that are trained on properties
  like fraction of each amino acid, or fraction of intrinsically disordered
  regions
  - one also integrates structural information obtained from AlphaFold

== The Block Decomposition Algorithm
- the block decomposition algorithm is able to find all factors of a sequence
  that have a balance lower or equal to the balance threshold
  - the balance threshold ensures that these blocks have a certain homogeneity
- to be less sensitive to substitutions the protein sequences are mapped before
  the decomposition
  - this leads to homogeneous blocks in the context of the current mapping
    that can be labeled to provide additional info

== The Block Decomposition Algorithm
- The labeling function will give information about the main components of the
  block
  - if a block consists of mainly amino acids with a mapping of 1, the label would
    be 1
  - if a block consists mainly of two amino acids mapped to 1 and 2 it gets the
    label 12 ...
- leading to a decomposition that could look like:
```
[rep(0, 3), rep(1, 14), rep(0, 2), rep(12, 30), ... ]
```

== The Idea
- if many different mappings are used where each represents a different property
  of the amino acids in it, a multidimensional block decomposition is created
- using deep learning models like convolutional neural networks (CNNs) the different
  decompositions can be treated similar to the color channels of a picture, like
  so:
#text(red)[0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0
2 2 2 2 2 2] \
#text(green)[3 3 3 3 2 2 2 2 2 2 0 0 0 0 0 0 1 1 1 4 4 4 4 4 4 4 4 3 3 3 3
3 3 3 0 0 0] \
#text(blue)[0 0 0 2 2 2 2 1 1 1 1 0 0 0 3 3 3 3 4 4 4 4 4 2 2 2 2 2 2 2 4
4 4 4 6 6 6] \

== The Idea
- as CNNs alone are not able to work with long range interactions a hybrid with
  Long short-term Memory or transformers may be able to observe relations
  between the different channels that is able to predict attributes like phase
  separation

= The Steps in Detail

== Reimplementing the block decomposition algorithm
- currently the block decomposition algorithm outputs data in the form of a
  block list, where each entry is a list containing a start and end position:
  - ```
  [[1, 20], [24, 45], ...]
  ```
  - This output lacks a label for the block as well as compatibility with deep
    learning models
- The desired output would look like this, where each position is the position
  of a amino acid, and the number is the label:
  - ``` [0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2...] ```

== Reimplementing the block decomposition algorithm
- This will be implemented as a Python module
- if python is too slow, the algorithm will be rewritten in a low level language
  like Rust

== Data Acquisition
- the data will be obtained from previous studies that developed phase
  separation predictors @hou_machine_2024 @chen_screening_2022
  - these studies took their data from curated databases for phase separation
- this makes the results comparable and saves time

== Finding meaningful Mappings
- To find meaningful mappings the results of the previous studies will be
  investigated as they provide lists of features that contributed most to phase
  separation propensity
  - for example mappings for idr related amino acids or amino acids that are
    involved in pi-pi interactions should be created

== Training Models
- As already described, the multidimensional block decomposition can be
  interpreted as a one dimensional image with many color channels, therefore
  1dCNNs will be used
- They will be integrated with Long short-term memory or transformer models to
  account for long distance relations
- pyTorch will probably be used for this

== Benchmarking
- if the trained models are capable of predicting phase separation a comparison
  to the other models will be made

// Create Mappings:
// #[
//   #show table.cell.where(x: 0): set text(blue)
//   #show table.cell.where(x: 1): set text(green)
//   #show table.cell.where(x: 2): set text(orange)
//   #show table.cell.where(x: 3): set text(gray)
//   #table(
//     columns: (15fr, 6fr, 6fr, 6fr),
//     [A C G H I K L M N P Q S T V], [D E],      [K E],      [F W Y],
//     [Aliphatic],                   [Negative], [Positive], [Aromatic]
//   )
// ]Get training data:
// #[ #show table.cell.where(x: 0): set text(green)
// #show table.cell.where(x: 1): set text(red)
// #table(
//   columns: 2,
//   [PS Positive Proteins], [PS Negative Proteins],
//   [abc123],               [def456],
//   [cba321],               [fed654],
//   [...],                  [..]
// )]
//
// Create Block Decomposition with labels (with different parameters):
// #block[
//   #v(1em)
//   #image("figures/block_decomposition.png")
//   #place(top + center, dx: -4em, [AN])
//   #place(top + center, dx: 9em, [NA])
// ]
//
// Train machine learning models:
//
// #align(center, image("figures/tf.jpg", width: 30%))
