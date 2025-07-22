#import "@preview/glossy:0.8.0": *
#import "@preview/subpar:0.2.2"
#import "state.typ": bib_state
#context bib_state.get()
#show: init-glossary.with(yaml("glossary.yaml"))

= Results

== Data Preparation
During data preparation, proteins that were longer than 2700 residues as well as
proteins that contained letters in their amino acid sequence that are not one of
the 20 common amino acids were filtered out. Including the @rsa values also
resulted in filtering the datasets, as some proteins had no structure file
available on AlphaFold. The amount of proteins that were filtered out as well
as the number of proteins that at least have one annotation for a @ptm are
shown in @dataprep. Only a small fraction of samples were filtered out. Many
proteins in the datasets are missing annotations for @ptm::pl.

#figure(table(
  columns: 7,
  align: (left, left, center, center, center, center, center),
  table.cell(rowspan: 2)[Dataset],  table.vline(stroke: 0.5pt),       table.cell(rowspan: 2)[Filter],   table.vline(stroke: 0.5pt),       table.cell(rowspan: 2)[All],    table.vline(stroke: 0.5pt),             table.cell(colspan: 2)[Train],
                                    table.vline(stroke: 0.5pt),                                         table.cell(colspan: 2)[Test],                                     table.cell(fill: none)[Positive],       table.hline(stroke: 0.5pt),
  table.cell(fill: none)[Negative], table.cell(fill: none)[Positive], table.cell(fill: none)[Negative], table.cell(rowspan: 4)[PPMC-lab], [None],                         [2876],                                 [604],
  [1696],                           [151],                            [425],                                                              [Length, Unknown Letters],      [2823],                                 [597],
  [1661],                           [149],                            [416],                                                              [@rsa],                         [2600],                                 [572],
  [1508],                           [143],                            [377],                                                              [@ptm],                         [1072],                                 [425],
  [454],                            [99],                             [94],                             table.hline(),                    table.cell(rowspan: 4)[PSPire], [None],                                 [10801],
  [259],                            [8323],                           [258],                            [1961],                                                           [Length, Unknown Letters],              [10748],
  [259],                            [8275],                           [258],                            [1956],                                                           [@rsa],                                 [10748],
  [259],                            [8275],                           [258],                            [1956],                                                           [@ptm],                                 [5804],
  [70],                             [4591],                           [64],                             [1079],                           table.hline(),                  table.cell(rowspan: 4)[catGranule 2.0], [None],
  [9383],                           [3333],                           [3252],                           [1422],                           [1376],                                                                 [Length, Unknown Letters],
  [9255],                           [3273],                           [3215],                           [1404],                           [1363],                                                                 [@rsa],
  [9255],                           [3273],                           [3215],                           [1404],                           [1363],                                                                 [@ptm],
  [6124],                           [2817],                           [1666],                           [980],                            [679],                          table.hline()
), caption: [Summary of the number of samples after each filter step. The PTM rows were not filtered, they are only included to show the fraction of proteins that do contain annotations
for PTMs.]) <dataprep>

The distribution of the sequence length for each dataset after the filter first
filter step can be seen in @datadist. It shows that most tested proteins are
smaller than 1000 residues.

#figure(image("figures/data_preparation_length_distribution.png", width: 60%), caption: [Histogram of the sequence length in the tested datasets.]) <datadist>

== The First Set of Models <first_set>

The first set of models, consisting of the one and two layer @cnn::pl, were compared
on both the PPMC-lab dataset and the PSPire dataset. The results are shown in
@first_models. The models using the raw sequence mostly outperformed the models
using the block decomposition as input.

#figure(table(
  columns: 6,
  align: center,
  table.cell(rowspan: 2)[Data Set / \
  Model],                                    table.cell(rowspan: 2)[@auc],               table.cell(colspan: 2, [@idr], align: center),                                           table.cell(colspan: 2, [non-@idr], align: center),
                                                                                          table.cell(fill: none)[Sequence],           table.cell(fill: none)[Block Decomposition], table.cell(fill: none)[Sequence],           table.cell(fill: none)[Block Decomposition],
  table.hline(stroke: 0.5pt),                 table.cell(rowspan: 2)[PPMC-lab / \
  1 Layer],                                  table.vline(stroke: 0.5pt),                 [@roc],                                      table.hline(stroke: 0.5pt),                 table.cell(fill: rgb(255, 255, 255))[0.87],
  table.cell(fill: rgb(245, 255, 245))[0.88],                                             table.cell(fill: rgb(255, 255, 255))[0.65], table.cell(fill: rgb(255, 215, 215))[0.61],  table.cell(fill: none)[@prc],               table.vline(stroke: 0.5pt),
  table.cell(fill: rgb(255, 255, 255))[0.66], table.cell(fill: rgb(255, 225, 225))[0.63], table.vline(stroke: 0.5pt),                 table.cell(fill: rgb(255, 255, 255))[0.72],  table.hline(stroke: 0.5pt),                 table.cell(fill: rgb(255, 245, 245))[0.71],
  table.cell(rowspan: 2)[PPMC-lab / \
  2 Layer],                                  [@roc],                                     table.hline(stroke: 0.5pt),                 table.cell(fill: rgb(245, 255, 245))[0.88],  table.cell(fill: rgb(255, 245, 245))[0.86], table.cell(fill: rgb(215, 255, 215))[0.69],
                                              table.cell(fill: rgb(215, 255, 215))[0.69], table.cell(fill: none)[@prc],               table.vline(stroke: 0.5pt),                  table.cell(fill: rgb(235, 255, 235))[0.68], table.cell(fill: rgb(255, 225, 225))[0.63],
  table.vline(stroke: 0.5pt),                 table.cell(fill: rgb(175, 255, 175))[0.73], table.cell(fill: rgb(205, 255, 205))[0.78], table.hline(),                               table.cell(rowspan: 2)[PSPire / \
  1 Layer],                                  [@roc],
  table.hline(stroke: 0.5pt),                 table.cell(fill: rgb(255, 255, 255))[0.70], table.cell(fill: rgb(255, 235, 235))[0.68], table.cell(fill: rgb(255, 255, 255))[0.63],                                              table.cell(fill: rgb(255, 215, 215))[0.59],
  table.cell(fill: none)[@prc],               table.vline(stroke: 0.5pt),                 table.cell(fill: rgb(255, 255, 255))[0.20], table.cell(fill: rgb(255, 255, 255))[0.20],  table.vline(stroke: 0.5pt),                 table.cell(fill: rgb(255, 255, 255))[0.06],
  table.cell(fill: rgb(255, 235, 235))[0.04], table.hline(stroke: 0.5pt),                 table.cell(rowspan: 2)[PSPire / \
  2 Layer],                                  [@roc],                                      table.hline(stroke: 0.5pt),                 table.cell(fill: rgb(165, 255, 165))[0.79],
  table.cell(fill: rgb(235, 255, 235))[0.72], table.cell(fill: rgb(255, 245, 245))[0.62],                                             table.cell(fill: rgb(255, 185, 185))[0.56],  table.cell(fill: none)[@prc],               table.vline(stroke: 0.5pt),
  table.cell(fill: rgb(85, 255, 85))[0.37],   table.cell(fill: rgb(195, 255, 195))[0.26], table.vline(stroke: 0.5pt),                 table.cell(fill: rgb(245, 255, 245))[0.05],  table.cell(fill: rgb(235, 255, 235))[0.04], table.hline()
), caption: [Comparison of the AUC values for the one and two layer CNN
with the block decomposition and the raw sequence as input. The one layer models
that used the sequence as input where used as the baseline. Better performance
is visualized with green, worse with red filling.]) <first_models>

== The Second Set of Models <secon_set>

The second set of models was only run on the PSPire dataset. The results of
these runs are shown in @second_phase. While the three layer @cnn performed
slightly better on the non-@idr proteins it performed worse on the proteins
containing @idr::pl. None of the new models outperform the two layer model.

#figure(table(
  columns: 4,
  align: (left, center, center, center),
  [model],                                    [@auc],                                               table.vline(stroke: 0.5pt),                         [@idr],
  table.vline(stroke: 0.5pt),                 [non-@idr],                                           table.cell(fill: none, rowspan: 2)[Two Layer @cnn], table.vline(stroke: 0.5pt),
  table.cell(fill: none)[@roc],               table.hline(stroke: 0.5pt),                                                                               table.cell(fill: rgb(255, 255, 255))[0.79],
  table.cell(fill: rgb(255, 255, 255))[0.62], [@prc],                                               table.hline(stroke: 0.5pt),                         table.cell(fill: rgb(255, 255, 255))[0.37],
  table.cell(fill: rgb(255, 255, 255))[0.05], table.cell(fill: none, rowspan: 2)[Three Layer @cnn], table.cell(fill: none)[@roc],                       table.hline(stroke: 0.5pt),
  table.cell(fill: rgb(255, 195, 195))[0.73],                                                       table.cell(fill: rgb(215, 255, 215))[0.68],         [@prc],
  table.hline(stroke: 0.5pt),                 table.cell(fill: rgb(255, 175, 175))[0.29],           table.cell(fill: rgb(235, 255, 235))[0.07],         table.cell(fill: none, rowspan: 2)[XGBoost (Block Decomposition)],
  table.cell(fill: none)[@roc],               table.hline(stroke: 0.5pt),                           table.cell(fill: rgb(255, 205, 205))[0.74],                                                                            table.cell(fill: rgb(255, 175, 175))[0.54], [@prc],                                               table.hline(stroke: 0.5pt),                         table.cell(fill: rgb(255, 135, 135))[0.25],
  table.cell(fill: rgb(255, 225, 225))[0.03], table.cell(fill: none, rowspan: 2)[@bilstm],          table.cell(fill: none)[@roc],                       table.hline(stroke: 0.5pt),
  table.cell(fill: rgb(255, 245, 245))[0.78],                                                       table.cell(fill: rgb(255, 125, 125))[0.49],         [@prc],
  table.hline(stroke: 0.5pt),                 table.cell(fill: rgb(255, 135, 135))[0.25],           table.cell(fill: rgb(255, 235, 235))[0.03],         table.cell(fill: none, rowspan: 2)[Transformer],
  table.cell(fill: none)[@roc],               table.hline(stroke: 0.5pt),                           table.cell(fill: rgb(245, 255, 245))[0.80],                                                                            table.cell(fill: rgb(255, 185, 185))[0.55], [@prc],                                               table.cell(fill: rgb(255, 185, 185))[0.29],         table.cell(fill: rgb(255, 225, 225))[0.03],
  table.hline()
), caption: [Comparison of the AUC values for the models created during
the second phase and the two layer CNN. The two layer model was used
as baseline. Better performance
is visualized with green, worse with red filling.]) <second_phase>

== Optimizing the Two Layer Convolutional Neural Network <opt_two_layer>

The first optimizations to the two layer @cnn, that consisted of doubling the
dropout value as well as splitting the model into @idr and non-@idr proteins,
were carried out on the PSPire dataset and the PPMC-lab dataset. The results
are shown in @split_dropout. While the results for the proteins with @idr::pl
almost remained the same, the results for the proteins without @idr::pl
improved on the PSPire dataset.

#figure(table(
  columns: 6,
  align: (left, center, center, center, center),
  table.cell(rowspan: 2)[Dataset],            table.vline(stroke: 0.5pt),                 table.cell(rowspan: 2)[@auc],               table.cell(colspan: 2)[@idr],                                                           table.cell(colspan: 2)[non-@idr],
                                              table.cell(fill: none)[Base],                                                           table.hline(stroke: 0.5pt),                 table.cell(fill: none)[Dropout + Split],    table.cell(fill: none)[Base],
  table.cell(fill: none)[Dropout + Split],    table.cell(rowspan: 2)[PPMC-lab],           [@roc],                                     table.hline(stroke: 0.5pt),                 table.cell(fill: rgb(255, 255, 255))[0.88], table.cell(fill: rgb(255, 225, 225))[0.85],
  table.cell(fill: rgb(255, 255, 255))[0.69],                                             table.cell(fill: rgb(255, 225, 225))[0.66], table.cell(fill: none)[@prc],               table.hline(),                              table.vline(stroke: 0.5pt),
  table.cell(fill: rgb(255, 255, 255))[0.68], table.cell(fill: rgb(255, 235, 235))[0.66], table.vline(stroke: 0.5pt),                 table.cell(fill: rgb(255, 255, 255))[0.73], table.cell(fill: rgb(245, 255, 245))[0.74], table.cell(rowspan: 2)[PSPire],
  [@roc],                                     table.hline(stroke: 0.5pt),                 table.cell(fill: rgb(255, 255, 255))[0.79], table.cell(fill: rgb(255, 245, 245))[0.78], table.cell(fill: rgb(255, 255, 255))[0.62],                                             table.cell(fill: rgb(165, 255, 165))[0.71], table.cell(fill: none)[@prc],               table.hline(stroke: 0.5pt),                 table.vline(stroke: 0.5pt),                 table.cell(fill: rgb(255, 255, 255))[0.37], table.cell(fill: rgb(245, 255, 245))[0.38],
  table.vline(stroke: 0.5pt),                 table.cell(fill: rgb(255, 255, 255))[0.05], table.cell(fill: rgb(175, 255, 175))[0.13], table.hline(stroke: 1pt)
), caption: [Results of adjusting the dropout to 0.6 and splitting the dataset into IDR and non-IDR. Better performance
is visualized with green, worse with red filling. ]) <split_dropout>

The results of the further optimization on the PSPire dataset are displayed in @opti.
Here the different integrations of the @rsa values as well as @bn and the integration
of @ptm::pl was tested.

#figure(table(
  columns: 12,
  align: (left, center, center, center, center, center, center, center, center, center),
  table.cell(rowspan: 2)[Dataset],            table.vline(stroke: 0.5pt),                 table.cell(rowspan: 2)[@auc],               table.cell(colspan: 5)[@idr],                                                                                                                                                                                               table.cell(colspan: 5)[non-@idr],
                                              table.cell(fill: none)[Base],                                                           table.cell(fill: none)[@rsa],               table.cell(fill: none)[@rsa weight],        table.cell(fill: none)[@bn],                table.cell(fill: none)[@ptm],               table.cell(fill: none)[Base],               table.cell(fill: none)[@rsa],               table.cell(fill: none)[@rsa weight],        table.hline(stroke: 0.5pt),                 table.cell(fill: none)[@bn],
  table.cell(fill: none)[@ptm],               table.cell(rowspan: 2)[PPMC-lab],           [@roc],                                     table.hline(stroke: 0.5pt),                 table.cell(fill: rgb(255, 255, 255))[0.85], table.cell(fill: rgb(225, 255, 225))[0.88], table.cell(fill: rgb(225, 255, 225))[0.88], table.cell(fill: rgb(255, 195, 195))[0.79], table.cell(fill: rgb(225, 255, 225))[0.88], table.vline(stroke: 0.5pt),                 table.cell(fill: rgb(255, 255, 255))[0.66], table.cell(fill: rgb(225, 255, 225))[0.69],
  table.cell(fill: rgb(205, 255, 205))[0.71],                                             table.cell(fill: rgb(255, 245, 245))[0.65], table.cell(fill: rgb(245, 255, 245))[0.67], table.cell(fill: none)[@prc],               table.hline(),                              table.vline(stroke: 0.5pt),                 table.cell(fill: rgb(255, 255, 255))[0.66], table.cell(fill: rgb(255, 255, 255))[0.66], table.cell(fill: rgb(225, 255, 225))[0.69], table.cell(fill: rgb(255, 125, 125))[0.53], table.cell(fill: rgb(215, 255, 215))[0.70],
  table.cell(fill: rgb(255, 255, 255))[0.74], table.cell(fill: rgb(175, 255, 175))[0.82], table.cell(fill: rgb(155, 255, 155))[0.84], table.cell(fill: rgb(255, 245, 245))[0.73], table.cell(fill: rgb(235, 255, 235))[0.76], table.cell(rowspan: 2)[PSPire],             [@roc],                                     table.hline(stroke: 0.5pt),                 table.cell(fill: rgb(255, 255, 255))[0.78], table.cell(fill: rgb(255, 255, 255))[0.78], table.cell(fill: rgb(235, 255, 235))[0.80], table.cell(fill: rgb(255, 205, 205))[0.73],
  table.cell(fill: rgb(255, 145, 145))[0.67], table.vline(stroke: 0.5pt),                 table.cell(fill: rgb(255, 255, 255))[0.71], table.cell(fill: rgb(125, 255, 125))[0.84], table.cell(fill: rgb(115, 255, 115))[0.85],                                             table.cell(fill: rgb(195, 255, 195))[0.77], table.cell(fill: rgb(175, 255, 175))[0.79], table.cell(fill: none)[@prc],               table.hline(stroke: 0.5pt),                 table.vline(stroke: 0.5pt),                 table.cell(fill: rgb(255, 255, 255))[0.37],
  table.cell(fill: rgb(235, 255, 235))[0.39], table.cell(fill: rgb(205, 255, 205))[0.42], table.cell(fill: rgb(255, 135, 135))[0.25], table.cell(fill: rgb(255, 85, 85))[0.20],   table.cell(fill: rgb(255, 255, 255))[0.13], table.cell(fill: rgb(185, 255, 185))[0.20], table.cell(fill: rgb(205, 255, 205))[0.18], table.cell(fill: rgb(195, 255, 195))[0.19], table.cell(fill: rgb(255, 215, 215))[0.09], table.hline(stroke: 1pt)
), caption: [Results of optimizing the model using different approaches. The two layer CNN is used as base model. Better performance
is visualized with green, worse with red filling. ]) <opti>

The last step of this optimization was to test the combination of multiple
approaches. This was only conducted on the PSPire dataset.
The results for this are shown in @combined.

#figure(table(
  columns: 9,
  align: (left, center, center, center, center, center, center, center, center),
  table.cell(rowspan: 2)[@auc],               table.cell(colspan: 4)[@idr],                                                                                                                                                   table.cell(colspan: 4)[non-@idr],
                                              table.cell(fill: none)[Base],               table.cell(fill: none)[@rsa w + @bn],       table.cell(fill: none)[@rsa w + @ptm],      table.cell(fill: none)[all],                table.vline(stroke: 0.5pt),                 table.cell(fill: none)[Base],               table.cell(fill: none)[@rsa w + @bn],       table.cell(fill: none)[@rsa w + @ptm],
  table.cell(fill: none)[all],                table.hline(stroke: 0.5pt),                 table.cell(fill: none)[@roc],               table.vline(stroke: 0.5pt),                 table.cell(fill: rgb(255, 255, 255))[0.78], table.cell(fill: rgb(215, 255, 215))[0.82], table.cell(fill: rgb(255, 145, 145))[0.67], table.cell(fill: rgb(255, 245, 245))[0.79], table.cell(fill: rgb(255, 255, 255))[0.71],
  table.cell(fill: rgb(105, 255, 105))[0.86], table.cell(fill: rgb(95, 255, 95))[0.87],   table.cell(fill: rgb(85, 255, 85))[0.88],   table.hline(stroke: 0.5pt),                 table.cell(fill: none)[@prc],               table.cell(fill: rgb(255, 255, 255))[0.37], table.cell(fill: rgb(205, 255, 205))[0.42], table.cell(fill: rgb(255, 95, 95))[0,21],   table.cell(fill: rgb(255, 255, 255))[0.37],
  table.cell(fill: rgb(255, 255, 255))[0.13], table.cell(fill: rgb(215, 255, 215))[0.17], table.cell(fill: rgb(195, 255, 195))[0.19], table.cell(fill: rgb(135, 255, 135))[0.25], table.hline(),
), caption: [Results of the combined approaches. Better performance
is visualized with green, worse with red filling.]) <combined>

#pagebreak()

== Evaluation of the model

The final evaluation was carried out on the PSPire dataset, the PPMC-lab dataset,
the @mlo datasets and the catGranule 2.0 dataset.

=== Evaluation on the PSPire Dataset

@final_model_pspire shows the @roc::pl and @prc::pl of the final models trained on the
PSPire dataset. @final_model_pspire_table shows the @auc values for this model
and compares them to the values that PSPire and PdPS achieved on the same
test sets. Their results are taken from the PSPire article @hou_machine_2024.

#subpar.grid(
  figure(image("figures/run_cnn2l_pspire_rsa_weight_idr_rocauc_idr.png"), caption: ""),
  <final_model_pspirer_a>,
  figure(image("figures/run_cnn2l_pspire_rsa_weight_idr_prauc_idr.png"), caption: ""),
  <final_model_pspire_b>,
  figure(image("figures/run_cnn2l_pspire_rsa_weight_bn_nidr_ptm_rocauc_nidr.png"), caption: ""),
  <final_model_pspire_c>,
  figure(image("figures/run_cnn2l_pspire_rsa_weight_bn_nidr_ptm_prauc_nidr.png"), caption: ""),
  <final_model_pspire_d>,
  columns: (1fr, 1fr),
  caption: [Results of the final model on the PSPire data. (a, b) ROCAUC and
  PRCAUC for Proteins containing IDRs. (c, d) ROCAUC and PRCAUC for
  Proteins containing no IDRs.],
  label: <final_model_pspire>,
)

#figure(table(
  columns: 7,
  table.cell(rowspan: 2)[@auc],             table.cell(colspan: 3, [@idr], align: center),                                                                                      table.cell(colspan: 3, [non-@idr], align: center),
                                            table.cell(fill: none)[Final Model],        table.cell(fill: none)[PSPire],             table.cell(fill: none)[PdPS],               table.cell(fill: none)[Final Model],        table.cell(fill: none)[PSPire],             table.cell(fill: none)[PdPS],
  table.hline(stroke: 0.5pt),               [@roc],                                     table.cell(fill: rgb(255, 255, 255))[0.80], table.cell(fill: rgb(195, 255, 195))[0.86], table.cell(fill: rgb(215, 255, 215))[0.84], table.cell(fill: rgb(255, 255, 255))[0.88], table.cell(fill: rgb(255, 215, 215))[0.84],
  table.cell(fill: rgb(255, 55, 55))[0.68], table.hline(stroke: 0.5pt),                 table.cell(fill: none)[@prc],               table.vline(stroke: 0.5pt),                 table.cell(fill: rgb(255, 255, 255))[0.42], table.cell(fill: rgb(165, 255, 165))[0.51], table.cell(fill: rgb(255, 255, 255))[0.42],
  table.vline(stroke: 0.5pt),               table.cell(fill: rgb(255, 255, 255))[0.25], table.cell(fill: rgb(255, 245, 245))[0.24], table.cell(fill: rgb(255, 85, 85))[0.08],   table.hline()
), caption: [Comparison of the AUC values for the final models, PSPire and PdPS. The values for PSPire and PdPS are taken from the PSPire article. The final models are used as base line. Better performance
is visualized with green, worse with red filling.]) <final_model_pspire_table>

#pagebreak()

=== Evaluation on the PPMC-lab Dataset

@final_ppmclab shows the final values of the PPMC-lab dataset.

#figure(table(
  columns: 3,
  align: (left, center, center),
  [@auc],        [@idr],                     [non-@idr],
  [@roc],        table.hline(stroke: 0.5pt), [0.88],
  [0.74],        [@prc],                     table.vline(stroke: 0.5pt),
  [0.69],        table.vline(stroke: 0.5pt), [0.86],
  table.hline()
), caption: [Results from the final model on the PPMC-lab dataset.]) <final_ppmclab>

=== Evaluation on the Membraneless Organells data sets

@evaluation_mlo compares the @auc values for the five @mlo datasets, that
were received using the final models trained on the PSPire training dataset. They
are compared to the values of the PSPire and PdPS predictor. Their values are
taken from the PSPire article @hou_machine_2024. For
proteins that do contain @idr::pl the model created in this work does perform
worse than both PSPire and PdPS. For proteins containing no @idr::pl however
this model outperforms PdPS consistently and has comparable performance to PSPire.

#figure(table(
  columns: 8,
  align: (left, left, center, center, center, center, center, center),
  table.cell(rowspan: 2)[Dataset],            table.cell(rowspan: 2)[AUC],                        table.cell(colspan: 3, [@idr]),                                                                                                               table.cell(colspan: 3, [non-@idr]),
                                                                                                  table.cell(fill: none)[Final Model],        table.cell(fill: none)[PSPire],                       table.cell(fill: none)[PdPS],               table.cell(fill: none)[Final Model],          table.cell(fill: none)[PSPire],             table.cell(fill: none)[PdPS],
  table.hline(stroke: 0.5pt),                 table.cell(rowspan: 2, "G3BP1 proximity labeling"), [@roc],                                     table.cell(fill: rgb(255, 255, 255))[0.78],           table.cell(fill: rgb(125, 255, 125))[0.91], table.cell(fill: rgb(175, 255, 175))[0.86],   table.cell(fill: rgb(255, 255, 255))[0.96], table.cell(fill: rgb(255, 235, 225))[0.93],
  table.cell(fill: rgb(255, 105, 105))[0.81],                                                     table.hline(stroke: 0.5pt),                 table.cell(fill: none)[@prc],                         table.cell(fill: rgb(255, 255, 255))[0.34], table.cell(fill: rgb(15, 255, 15))[0.58],     table.cell(fill: rgb(185, 255, 185))[0.41], table.cell(fill: rgb(255, 255, 255))[0.51],
  table.hline(),                              table.cell(fill: rgb(105, 255, 105))[0.66],         table.cell(fill: rgb(255, 0, 0))[0.18],     table.cell(rowspan: 2, "DACT1-particulate proteome"), [@roc],                                     table.hline(stroke: 0.5pt),                   table.cell(fill: rgb(255, 255, 255))[0.72], table.cell(fill: rgb(95, 255, 95))[0.88],
  table.cell(fill: rgb(125, 255, 125))[0.85], table.cell(fill: rgb(255, 255, 255))[0.90],         table.cell(fill: rgb(225, 255, 225))[0.93],                                                       table.cell(fill: rgb(255, 165, 165))[0.81], table.cell(fill: none)[@prc],                 table.cell(fill: rgb(255, 255, 255))[0.22], table.cell(fill: rgb(125, 255, 125))[0.35],
  table.cell(fill: rgb(145, 255, 145))[0.33], table.cell(fill: rgb(255, 255, 255))[0.49],         table.cell(fill: rgb(145, 255, 145))[0.60], table.cell(fill: rgb(255, 0, 0))[0.18],               table.hline(),                              table.cell(rowspan: 2, "RNAgranuleDB Tier1"), [@roc],                                     table.hline(stroke: 0.5pt),
  table.cell(fill: rgb(255, 255, 255))[0.77], table.cell(fill: rgb(185, 255, 185))[0.84],         table.cell(fill: rgb(205, 255, 205))[0.82], table.cell(fill: rgb(255, 255, 255))[0.88],           table.cell(fill: rgb(235, 255, 235))[0.90],                                               table.cell(fill: rgb(255, 55, 55))[0.68],   table.cell(fill: none)[@prc],
  table.cell(fill: rgb(255, 255, 255))[0.42], table.cell(fill: rgb(195, 255, 195))[0.48],         table.hline(),                              table.cell(fill: rgb(255, 255, 255))[0.42],           table.cell(fill: rgb(255, 255, 255))[0.18], table.cell(fill: rgb(155, 255, 155))[0.28],   table.cell(fill: rgb(255, 155, 155))[0.08], table.cell(rowspan: 2, "PhaSepDB low and high throughput MLO"),
  table.hline(stroke: 0.5pt),                 [@roc],                                             table.cell(fill: rgb(255, 255, 255))[0.70], table.cell(fill: rgb(235, 255, 235))[0.72],           table.cell(fill: rgb(215, 255, 215))[0.74], table.cell(fill: rgb(255, 255, 255))[0.85],   table.cell(fill: rgb(255, 205, 205))[0.80],                                                                 table.cell(fill: rgb(255, 55, 55))[0.65],   table.cell(fill: none)[@prc],                       table.hline(),                              table.cell(fill: rgb(255, 255, 255))[0.70],           table.cell(fill: rgb(165, 255, 165))[0.79], table.cell(fill: rgb(155, 255, 155))[0.80],   table.cell(fill: rgb(255, 255, 255))[0.73], table.cell(fill: rgb(255, 235, 235))[0.71],
  table.cell(fill: rgb(255, 0, 0))[0.47],     table.cell(rowspan: 2, "DrLLPS MLO"),               table.vline(stroke: 0.5pt),                 table.hline(stroke: 0.5pt),                           [@roc],                                     table.cell(fill: rgb(255, 255, 255))[0.68],   table.cell(fill: rgb(185, 255, 185))[0.75], table.cell(fill: rgb(175, 255, 175))[0.76],
  table.cell(fill: rgb(255, 255, 255))[0.80],                                                     table.cell(fill: rgb(205, 255, 205))[0.85], table.cell(fill: rgb(255, 135, 135))[0.68],           table.cell(fill: none)[@prc],               table.vline(stroke: 0.5pt),                   table.cell(fill: rgb(255, 255, 255))[0.72], table.cell(fill: rgb(195, 255, 195))[0.78],
  table.cell(fill: rgb(205, 255, 205))[0.77], table.vline(stroke: 0.5pt),                         table.cell(fill: rgb(255, 255, 255))[0.72], table.cell(fill: rgb(235, 255, 235))[0.74],           table.cell(fill: rgb(255, 0, 0))[0.45],     table.hline()
), caption: [Evaluation Summary of the final model on the MLO datasets. The values for PSPire and PdPS are taken from the PSPire article. The final models of this work are
used as base line. Better performance
is visualized with green, worse with red filling.]) <evaluation_mlo>

@evaluation_mlo_cmp compares the results obtained when using the models trained
on the PSPire dataset to the results obtained using the PPMC-lab dataset. The final
models that was trained on the PSPire dataset outperform the models trained on the
PPMC-lab dataset significantly. Especially for the non-@idr proteins, where the model
trained on the PPMC-lab performs similar to random guessing.

#figure(table(
  columns: 6,
  table.cell(rowspan: 2)[Dataset],            table.cell(rowspan: 2)[AUC],                                    table.cell(colspan: 2, align: center, [@idr]),                                                    table.cell(colspan: 2, [non-@idr], align: center),
                                                                                                              table.cell(fill: none)[PSPire],             table.cell(fill: none)[PPMC-lab],                     table.cell(fill: none)[PSPire],             table.cell(fill: none)[PPMC-lab],
  table.hline(stroke: 0.5pt),                 table.cell(rowspan: 2, "G3BP1 proximity labeling"),             [@roc],                                     table.cell(fill: rgb(255, 255, 255))[0.78],           table.cell(fill: rgb(255, 125, 125))[0.65], table.cell(fill: rgb(255, 255, 255))[0.96],
  table.cell(fill: rgb(255, 0, 0))[0.49],                                                                     table.hline(stroke: 0.5pt),                 table.cell(fill: none)[@prc],                         table.cell(fill: rgb(255, 255, 255))[0.34], table.cell(fill: rgb(255, 95, 95))[0.18],
  table.cell(fill: rgb(255, 255, 255))[0.51], table.hline(),                                                  table.cell(fill: rgb(255, 0, 0))[0.05],     table.cell(rowspan: 2, "DACT1-particulate proteome"), [@roc],                                     table.hline(stroke: 0.5pt),
  table.cell(fill: rgb(255, 255, 255))[0.72], table.cell(fill: rgb(255, 135, 135))[0.60],                     table.cell(fill: rgb(255, 255, 255))[0.90],                                                       table.cell(fill: rgb(255, 0, 0))[0.51],     table.cell(fill: none)[@prc],
  table.cell(fill: rgb(255, 255, 255))[0.22], table.cell(fill: rgb(255, 145, 145))[0.11],                     table.cell(fill: rgb(255, 255, 255))[0.49], table.cell(fill: rgb(255, 0, 0))[0.07],               table.hline(),                              table.cell(rowspan: 2, "RNAgranuleDB Tier1"),
  [@roc],                                     table.hline(stroke: 0.5pt),                                     table.cell(fill: rgb(255, 255, 255))[0.77], table.cell(fill: rgb(255, 255, 255))[0.77],           table.cell(fill: rgb(255, 255, 255))[0.88],                                               table.cell(fill: rgb(255, 0, 0))[0.53],     table.cell(fill: none)[@prc],                                   table.cell(fill: rgb(255, 255, 255))[0.42], table.hline(),                                        table.cell(fill: rgb(255, 255, 255))[0.42], table.cell(fill: rgb(255, 255, 255))[0.18],
  table.cell(fill: rgb(255, 105, 105))[0.03], table.cell(rowspan: 2, "PhaSepDB low and high throughput MLO"), table.hline(stroke: 0.5pt),                 [@roc],                                               table.cell(fill: rgb(255, 255, 255))[0.70], table.cell(fill: rgb(255, 215, 215))[0.66],
  table.cell(fill: rgb(255, 255, 255))[0.85],                                                                 table.cell(fill: rgb(255, 0, 0))[0.46],     table.cell(fill: none)[@prc],                         table.hline(),                              table.cell(fill: rgb(255, 255, 255))[0.70],
  table.cell(fill: rgb(255, 255, 255))[0.70], table.cell(fill: rgb(255, 255, 255))[0.73],                     table.cell(fill: rgb(255, 0, 0))[0.31],     table.cell(rowspan: 2, "DrLLPS MLO"),                 table.vline(stroke: 0.5pt),                 table.hline(stroke: 0.5pt),
  [@roc],                                     table.cell(fill: rgb(255, 255, 255))[0.68],                     table.cell(fill: rgb(255, 235, 235))[0.66],                                                       table.cell(fill: rgb(255, 255, 255))[0.80], table.cell(fill: rgb(255, 0, 0))[0.49],
  table.cell(fill: none)[@prc],               table.vline(stroke: 0.5pt),                                     table.cell(fill: rgb(255, 255, 255))[0.72], table.cell(fill: rgb(255, 205, 205))[0.67],           table.vline(stroke: 0.5pt),                 table.cell(fill: rgb(255, 255, 255))[0.72],
  table.cell(fill: rgb(255, 0, 0))[0.33],     table.hline()
), caption: [Comparison of the models performance trained on the PSPire dataset and the models trained on the PPMC-lab dataset.]) <evaluation_mlo_cmp>

As both models in the previous comparison used the negatives of the PSPire test
datasets, it was tested if the performance of the model trained on the PPMC-lab
dataset would improve if the negatives of its own dataset were used instead.
The DACT1-particulate proteome was used for this. The results are shown
in @self. There were no improvements.

#figure(table(
  columns: 3,
  align: (),
  [@auc],        [@idr], [non-@idr],
  [@roc],        [0.54], [0.54],
  [@prc],        [0.29], [0.30],
  table.hline()
), caption: [Results of evaluating the models trained on the PPMC-lab dataset
on the DACT1-particulate MLO dataset while using the PPMC-lab negative test set. ]) <self>

=== Evaluation on the catGranule 2.0 Dataset

@final_cat compares the results of the final models trained on the catGranule
2.0 training dataset to the results of the other predictors that were evaluated
in the catGranule
2.0 paper @monti_catgranule_2025. The non-@idr model of this work is able to
slightly outperform the other models. This model achieved a @prc@auc value of
0.81 for the non-@idr and 0.71 for the @idr model.

#figure(table(
  columns: 9,
  [@auc],                                     [catGranule 1.0],           [MaGS],                                     [PSPHunter],                                [PICNIC],                                   [PICNIC-GO],                                [catGranule 2.0],                           [non-@idr model],                           [@idr model],
  table.cell(fill: none)[@roc],               table.vline(stroke: 0.5pt), table.cell(fill: rgb(255, 255, 255))[0.66], table.cell(fill: rgb(255, 255, 255))[0.74], table.cell(fill: rgb(255, 255, 255))[0.74], table.cell(fill: rgb(255, 255, 255))[0.73], table.cell(fill: rgb(255, 255, 255))[0.75], table.cell(fill: rgb(255, 255, 255))[0.76], table.cell(fill: rgb(255, 255, 255))[0.80],
  table.cell(fill: rgb(255, 255, 255))[0.74], table.hline()
), caption: [Comparison of the ROCAUC of several LLPS predictors on the catGranule 2.0 test data set.
The values for all predictors but my own are taken from the catGranule 2.0 article.]) <final_cat>

=== Visualization of Input Features

@salinity_idr shows the saliency scores visualized along the amino acid
sequence of P04264, which is labeled as a @idr @llps protein @hou_machine_2024.
Therefore, the results of the @idr model are shown here. The predicted probability
for it to be a @llps protein was 98 % with this works model. There are two
bright colored bands visible at around the residues 190 to 200 and 555 to 570.
Many more less bright bands are visible as well. The left bright band lies
within a intermediate filament rod domain @noauthor_prorule_nodate. The right
bright band is near an @idr @noauthor_mobidb_nodate.

#figure(image("figures/captum_idr_P04264_1.0_0.9834264516830444_ID-PSP.png"), caption: [Visualization of the Input Features using the IDR model on the protein P04264.]) <salinity_idr>

For comparison, using the catGranule 2.0 web app yielded a score of 0.845 (1 is
the highest possible value) and
the LLPS propensity profile seen in @llpsprofile @monti_catgranule_2025. While
they both show some similar regions to be important for @llps, the left bright
band in the saliency map is missing in the plot of catGranule 2.0.

#figure(image("figures/sp|P04264|K2C1_HUMAN_Profile.png", width: 70%), caption: [LLPS Propensity profile of the protein P04264 predicted by catGranule 2.0.]) <llpsprofile>

@cmpms shows the same graph for a different protein. In this case the protein
P42766, which is a non-@idr labeled @llps protein @hou_machine_2024. The
figure shows the saliency score visualizations for both models. It is important
to mention that saliency scores are neither comparable between samples nor
between models. The non-@idr model scored a probability of 44 % while the @idr
model scored a probability of
21 %. Viewing this protein with the UniProt feature viewer revealed only one
entry under domains. An @idr from residue 86 to the end @noauthor_mobidb_nodate-1.

#subpar.grid(columns: (1fr), figure(image("figures/captum_nidr_P42766_1.0_0.44323086738586426_noID-PSP.png"), caption: []), figure(image("figures/captum_idr_P42766_1.0_0.20686744153499603_noID-PSP.png"), caption: []), caption: [Comparison of the feature relevance for both the non-IDR model (a)
and the IDR model (b) on the protein P42766.], label: <cmpms>)

The protein was also tested with catGranule 2.0s web app and yielded a score of
0.835 as well as the @llps propensity profile seen in @llpsproile
@monti_catgranule_2025. There are again similarities as well as differences
between these representations.

#figure(image("figures/sp|P42766|RL35_HUMAN_Profile.png", width: 70%), caption: [
  LLPS Propensity profile of the protein P42766 predicted by catGranule 2.0.
]) <llpsproile>

#pagebreak()
