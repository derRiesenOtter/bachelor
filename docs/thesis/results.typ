#import "@preview/glossy:0.8.0": *
#import "@preview/subpar:0.2.2"
#import "state.typ": bib_state
#context bib_state.get()
#show: init-glossary.with(yaml("glossary.yaml"))

= Results

== Data Preparation

== The First Set of Models

The first set of models, consisting of the one and two layer @cnn::pl, were compared
on both the PPMC-lab dataset and the PSPire dataset. The results are shown in
@first_models. The models using the raw sequence mostly outperformed the models
using the block decomposition as input.

#figure(table(
  columns: 6,
  align: center,
  table.cell(rowspan: 2)[Data Set / \
  Model],                                    table.cell(rowspan: 2)[@auc],              table.cell(colspan: 2, [@idr], align: center),                                          table.cell(colspan: 2, [non-@idr], align: center),
                                                                                         table.cell(fill: none)[Sequence],          table.cell(fill: none)[Block Decomposition], table.cell(fill: none)[Sequence],          table.cell(fill: none)[Block Decomposition],
  table.hline(stroke: 0.5pt),                 table.cell(rowspan: 2)[PPMC-lab / \
  1 Layer],                                 table.vline(stroke: 0.5pt),                [@roc],                                      table.hline(stroke: 0.5pt),                table.cell(fill: rgb(0, 0, 0, 60))[0.87],
  table.cell(fill: rgb(0, 10, 0, 60))[0.88],                                             table.cell(fill: rgb(0, 0, 0, 60))[0.65],  table.cell(fill: rgb(40, 0, 0, 60))[0.61],   [@prc],                                    table.vline(stroke: 0.5pt),
  table.cell(fill: rgb(0, 0, 0, 60))[0.66],   table.cell(fill: rgb(30, 0, 0, 60))[0.63], table.vline(stroke: 0.5pt),                table.cell(fill: rgb(0, 0, 0, 60))[0.72],    table.hline(stroke: 0.5pt),                table.cell(fill: rgb(10, 0, 0, 60))[0.71],
  table.cell(rowspan: 2)[PPMC-lab / \
  2 Layer],                                  [@roc],                                    table.hline(stroke: 0.5pt),                table.cell(fill: rgb(0, 10, 0, 60))[0.88],   table.cell(fill: rgb(0, 10, 0, 60))[0.86], table.cell(fill: rgb(0, 40, 0, 60))[0.69],
                                              table.cell(fill: rgb(0, 40, 0, 60))[0.69], [@prc],                                    table.vline(stroke: 0.5pt),                  table.cell(fill: rgb(0, 20, 0, 60))[0.68], table.cell(fill: rgb(30, 0, 0, 60))[0.63],
  table.vline(stroke: 0.5pt),                 table.cell(fill: rgb(0, 10, 0, 60))[0.73], table.cell(fill: rgb(0, 60, 0, 60))[0.78], table.hline(),                               table.cell(rowspan: 2)[PSPire / \
  1 Layer],                                 [@roc],
  table.hline(stroke: 0.5pt),                 table.cell(fill: rgb(0, 0, 0, 60))[0.70],  table.cell(fill: rgb(20, 0, 0, 60))[0.68], table.cell(fill: rgb(0, 0, 0, 60))[0.63],                                               table.cell(fill: rgb(40, 0, 0, 60))[0.59],
  [@prc],                                     table.vline(stroke: 0.5pt),                table.cell(fill: rgb(0, 0, 0, 60))[0.20],  table.cell(fill: rgb(0, 0, 0, 60))[0.20],    table.vline(stroke: 0.5pt),                table.cell(fill: rgb(0, 0, 0, 60))[0.06],
  table.cell(fill: rgb(20, 0, 0, 60))[0.04],  table.hline(stroke: 0.5pt),                table.cell(rowspan: 2)[PSPire / \
  2 Layer],                                 [@roc],                                      table.hline(stroke: 0.5pt),                table.cell(fill: rgb(0, 90, 0, 60))[0.79],
  table.cell(fill: rgb(0, 20, 0, 60))[0.72],  table.cell(fill: rgb(10, 0, 0, 60))[0.62],                                            table.cell(fill: rgb(70, 0, 0, 60))[0.56],   [@prc],                                    table.vline(stroke: 0.5pt),
  table.cell(fill: rgb(0, 170, 0, 60))[0.37], table.cell(fill: rgb(0, 60, 0, 60))[0.26], table.vline(stroke: 0.5pt),                table.cell(fill: rgb(10, 0, 0, 60))[0.05],   table.cell(fill: rgb(20, 0, 0, 60))[0.04], table.hline()
), caption: [Comparison of the @auc values for the one and two layer @cnn
with the block decomposition and the raw sequence as input. PPMC-lab dataset.]) <first_models>

== The Second Set of Models

The second set of models was only run on the PSPire dataset. The results of these runs are
shown in @second_phase.

#figure(table(
  columns: 4,
  align: (left, center, center, center),
  [model],                                   [@auc],                                               table.vline(stroke: 0.5pt),                         [@idr],
  table.vline(stroke: 0.5pt),                [non-@idr],                                           table.cell(fill: none, rowspan: 2)[Two Layer @cnn], table.vline(stroke: 0.5pt),
  [@roc],                                    table.hline(stroke: 0.5pt),                                                                               table.cell(fill: rgb(0, 0, 0, 60))[0.79],
  table.cell(fill: rgb(0, 0, 0, 60))[0.62],  [@prc],                                               table.hline(stroke: 0.5pt),                         table.cell(fill: rgb(0, 0, 0, 60))[0.37],
  table.cell(fill: rgb(0, 0, 0, 60))[0.05],  table.cell(fill: none, rowspan: 2)[Three Layer @cnn], [@roc],                                             table.hline(stroke: 0.5pt),
  table.cell(fill: rgb(60, 0, 0, 60))[0.73],                                                       table.cell(fill: rgb(0, 60, 0, 60))[0.68],          [@prc],
  table.hline(stroke: 0.5pt),                table.cell(fill: rgb(80, 0, 0, 60))[0.29],            table.cell(fill: rgb(0, 20, 0, 60))[0.07],          table.cell(fill: none, rowspan: 2)[XGBoost (Block Decomposition)],
  [@roc],                                    table.hline(stroke: 0.5pt),                           table.cell(fill: rgb(50, 0, 0, 60))[0.74],                                                                             table.cell(fill: rgb(80, 0, 0, 60))[0.54], [@prc],                                               table.hline(stroke: 0.5pt),                         table.cell(fill: rgb(120, 0, 0, 60))[0.25],
  table.cell(fill: rgb(20, 0, 0, 60))[0.03], table.cell(fill: none, rowspan: 2)[@bilstm],          [@roc],                                             table.hline(stroke: 0.5pt),
  table.cell(fill: rgb(10, 0, 0, 60))[0.78],                                                       table.cell(fill: rgb(130, 0, 0, 60))[0.49],         [@prc],
  table.hline(stroke: 0.5pt),                table.cell(fill: rgb(120, 0, 0, 60))[0.25],           table.cell(fill: rgb(20, 0, 0, 60))[0.03],          table.cell(fill: none, rowspan: 2)[Transformer],
  [@roc],                                    table.hline(stroke: 0.5pt),                           table.cell(fill: rgb(0, 10, 0, 60))[0.80],                                                                             table.cell(fill: rgb(70, 0, 0, 60))[0.55], [@prc],                                               table.cell(fill: rgb(80, 0, 0, 60))[0.29],          table.cell(fill: rgb(20, 0, 0, 60))[0.03],
  table.hline()
), caption: [Comparison of the @auc values for the models created during
the second phase and the two layer @cnn. The best values are written in bold.]) <second_phase>

== Optimizing the Two Layer @cnn
The first optimizations of the two layer @cnn was carried out on the PSPire
dataset. The results are shown in @split_dropout. While the results for the
proteins with @idr::pl almost remained the same, the results for the proteins
without @idr::pl improved.

#figure(table(
  columns: 5,
  align: (left, center, center, center, center),
  table.cell(rowspan: 2)[@auc],             table.cell(colspan: 2)[@idr],                                          table.cell(colspan: 2)[non-@idr],
                                            table.cell(fill: none)[Base],              table.hline(stroke: 0.5pt), table.cell(fill: none)[Dropout + Split],  table.cell(fill: none)[Base],
  table.cell(fill: none)[Dropout + Split],  [@roc],                                    table.hline(stroke: 0.5pt), table.cell(fill: rgb(0, 0, 0, 60))[0.79], table.cell(fill: rgb(10, 0, 0, 60))[0.78],
  table.cell(fill: rgb(0, 0, 0, 60))[0.62], table.cell(fill: rgb(0, 90, 0, 60))[0.71], [@prc],                     table.hline(stroke: 0.5pt),               table.vline(stroke: 0.5pt),
  table.cell(fill: rgb(0, 0, 0, 60))[0.37], table.cell(fill: rgb(0, 10, 0, 60))[0.38], table.vline(stroke: 0.5pt), table.cell(fill: rgb(0, 0, 0, 60))[0.05], table.cell(fill: rgb(0, 80, 0, 60))[0.13],
  table.hline(stroke: 1pt)
), caption: [Results of adjusting the dropout to 0.6 and splitting the dataset into @idr and non@idr. ]) <split_dropout>

The results of the further optimization are displayed in @opti.

#figure(table(
  columns: 11,
  align: (left, center, center, center, center, center, center, center, center, center),
  table.cell(rowspan: 2)[@auc],               table.cell(colspan: 5)[@idr],                                                                                                                                                                                          table.cell(colspan: 5)[non-@idr],
                                              table.cell(fill: none)[Base],              table.cell(fill: none)[@rsa],              table.cell(fill: none)[@rsa as weight],    table.cell(fill: none)[@bn],               table.cell(fill: none)[@ptm],              table.cell(fill: none)[Base],              table.cell(fill: none)[@rsa],               table.cell(fill: none)[@rsa as weight],    table.hline(stroke: 0.5pt),                 table.cell(fill: none)[@bn],
  table.cell(fill: none)[@ptm],               [@roc],                                    table.hline(stroke: 0.5pt),                table.cell(fill: rgb(0, 0, 0, 60))[0.78],  table.cell(fill: rgb(0, 0, 0, 60))[0.78],  table.cell(fill: rgb(0, 20, 0, 60))[0.80], table.cell(fill: rgb(50, 0, 0, 60))[0.73], table.cell(fill: rgb(110, 0, 0, 60))[0.67], table.vline(stroke: 0.5pt),                table.cell(fill: rgb(0, 0, 0, 60))[0.71],   table.cell(fill: rgb(0, 130, 0, 60))[0.84],
  table.cell(fill: rgb(0, 140, 0, 60))[0.85], table.cell(fill: rgb(0, 60, 0, 60))[0.77], table.cell(fill: rgb(0, 80, 0, 60))[0.79], [@prc],                                    table.hline(stroke: 0.5pt),                table.vline(stroke: 0.5pt),                table.cell(fill: rgb(0, 0, 0, 60))[0.37],  table.cell(fill: rgb(0, 20, 0, 60))[0.39],  table.cell(fill: rgb(0, 50, 0, 60))[0.42], table.cell(fill: rgb(120, 0, 0, 60))[0.25], table.cell(fill: rgb(170, 0, 0, 60))[0.20],
  table.cell(fill: rgb(0, 0, 0, 60))[0.13],   table.cell(fill: rgb(0, 70, 0, 60))[0.20], table.cell(fill: rgb(0, 50, 0, 60))[0.18], table.cell(fill: rgb(0, 60, 0, 60))[0.19], table.cell(fill: rgb(40, 0, 0, 60))[0.09], table.hline(stroke: 1pt)
), caption: [Results of optimizing the model using different approaches. The two layer @cnn is used as base model. Green represents improvement while red represents worsening. ]) <opti>

== Evaluation of the model

The final evaluation was carried out on the PSPire dataset, the @mlo datasets
and the catGranule 2.0 dataset.

=== PSPire Dataset

@final_model_pspire shows the @roc and @prc of the final model trained on the
PSPire dataset. @final_model_pspire_table shows the @auc values for this model
and compares them to the values obtained from PSPire and PdPS.

#subpar.grid(
  figure(image("figures/run_cnn2l_pspire_rsa_weight_bn_idr_ptm_rocauc_idr.png"), caption: ""),
  <final_model_pspirer_a>,
  figure(image("figures/run_cnn2l_pspire_rsa_weight_bn_idr_ptm_prauc_idr.png"), caption: ""),
  <final_model_pspire_b>,
  figure(image("figures/run_cnn2l_pspire_rsa_weight_bn_nidr_ptm_rocauc_nidr.png"), caption: ""),
  <final_model_pspire_c>,
  figure(image("figures/run_cnn2l_pspire_rsa_weight_bn_nidr_ptm_prauc_nidr.png"), caption: ""),
  <final_model_pspire_d>,
  columns: (1fr, 1fr),
  caption: [Results of the final model on the PSPire data. (a, b) @roc@auc and
  @prc@auc for Proteins containing @idr::pl. (c, d) @roc@auc and @prc@auc for
  Proteins containing no @idr::pl. TODO],
  label: <final_model_pspire>,
)

#figure(table(
  columns: 7,
  table.cell(rowspan: 2)[@auc],               table.cell(colspan: 3, [@idr], align: center),                                                                                    table.cell(colspan: 3, [non-@idr], align: center),
                                              table.cell(fill: none)[My Model],          table.cell(fill: none)[PSPire],             table.cell(fill: none)[PdPS],              table.cell(fill: none)[My Model],          table.cell(fill: none)[PSPire],           table.cell(fill: none)[PdPS],
  table.hline(stroke: 0.5pt),                 [@roc],                                    table.cell(fill: rgb(0, 0, 0, 60))[0.80],   table.cell(fill: rgb(0, 60, 0, 60))[0.86], table.cell(fill: rgb(0, 40, 0, 60))[0.84], table.cell(fill: rgb(0, 0, 0, 60))[0.88], table.cell(fill: rgb(40, 0, 0, 60))[0.84],
  table.cell(fill: rgb(200, 0, 0, 60))[0.68], [@prc],                                    table.vline(stroke: 0.5pt),                 table.cell(fill: rgb(0, 0, 0, 60))[0.42],  table.cell(fill: rgb(0, 90, 0, 60))[0.51], table.cell(fill: rgb(0, 0, 0, 60))[0.42], table.vline(stroke: 0.5pt),
  table.cell(fill: rgb(0, 0, 0, 60))[0.25],   table.cell(fill: rgb(10, 0, 0, 60))[0.24], table.cell(fill: rgb(170, 0, 0, 60))[0.08], table.hline()
), caption: [Comparison of the @auc values for my model, PSPire and PdPS. The values for PSPire and PdPS are taken from the PSPire article. @hou_machine_2024]) <final_model_pspire_table>

=== Evaluation on the @mlo data sets

@evaluation_mlo compares the @auc values for the five @mlo datasets. For
proteins that do contain @idr::pl the model created in this work does perform
worse than both PSPire and PdPS. For proteins containing no @idr::pl however
this model outperforms PdPS consistently and directly competes with PSPire.

#figure(table(
  columns: 8,
  table.cell(rowspan: 2)[Dataset],             table.cell(rowspan: 2)[AUC],                                    table.cell(colspan: 3, align: center, [@idr]),                                                                                      table.cell(colspan: 3, [non-@idr], align: center),
                                                                                                               table.cell(fill: none)[My Model],           table.cell(fill: none)[PSPire],             table.cell(fill: none)[PdPS],               table.cell(fill: none)[My Model],           table.cell(fill: none)[PSPire],            table.cell(fill: none)[PdPS],
  table.hline(stroke: 0.5pt),                  table.cell(rowspan: 2, "G3BP1 proximity labeling"),             [@roc],                                     table.cell(fill: rgb(0, 0, 0, 60))[0.78],   table.cell(fill: rgb(0, 130, 0, 60))[0.91], table.cell(fill: rgb(0, 80, 0, 60))[0.86],  table.cell(fill: rgb(0, 0, 0, 60))[0.96],  table.cell(fill: rgb(30, 0, 0, 60))[0.93],
  table.cell(fill: rgb(150, 0, 0, 60))[0.81],                                                                  [@prc],                                     table.cell(fill: rgb(0, 0, 0, 60))[0.34],   table.cell(fill: rgb(0, 240, 0, 60))[0.58], table.cell(fill: rgb(0, 70, 0, 60))[0.41],  table.cell(fill: rgb(0, 0, 0, 60))[0.51],  table.cell(fill: rgb(0, 150, 0, 60))[0.66],
  table.cell(fill: rgb(250, 0, 0, 210))[0.18], table.cell(rowspan: 2, "DACT1-particulate proteome"),           [@roc],                                     table.cell(fill: rgb(0, 0, 0, 60))[0.72],   table.cell(fill: rgb(0, 160, 0, 60))[0.88], table.cell(fill: rgb(0, 130, 0, 60))[0.85], table.cell(fill: rgb(0, 0, 0, 60))[0.90],  table.cell(fill: rgb(0, 30, 0, 60))[0.93],
  table.cell(fill: rgb(90, 0, 0, 60))[0.81],                                                                   [@prc],                                     table.cell(fill: rgb(0, 0, 0, 60))[0.22],   table.cell(fill: rgb(0, 130, 0, 60))[0.35], table.cell(fill: rgb(0, 110, 0, 60))[0.33], table.cell(fill: rgb(0, 0, 0, 60))[0.49],  table.cell(fill: rgb(0, 110, 0, 60))[0.60],
  table.cell(fill: rgb(250, 0, 0, 120))[0.18], table.cell(rowspan: 2, "RNAgranuleDB Tier1"),                   [@roc],                                     table.cell(fill: rgb(0, 0, 0, 60))[0.77],   table.cell(fill: rgb(0, 70, 0, 60))[0.84],  table.cell(fill: rgb(0, 50, 0, 60))[0.82],  table.cell(fill: rgb(0, 0, 0, 60))[0.88],  table.cell(fill: rgb(0, 20, 0, 60))[0.90],
  table.cell(fill: rgb(200, 0, 0, 60))[0.68],                                                                  [@prc],                                     table.cell(fill: rgb(0, 0, 0, 60))[0.42],   table.cell(fill: rgb(0, 60, 0, 60))[0.48],  table.cell(fill: rgb(0, 0, 0, 60))[0.42],   table.cell(fill: rgb(0, 0, 0, 60))[0.18],  table.cell(fill: rgb(0, 100, 0, 60))[0.28],
  table.cell(fill: rgb(100, 0, 0, 60))[0.08],  table.cell(rowspan: 2, "PhaSepDB low and high throughput MLO"), [@roc],                                     table.cell(fill: rgb(0, 0, 0, 60))[0.70],   table.cell(fill: rgb(0, 20, 0, 60))[0.72],  table.cell(fill: rgb(0, 40, 0, 60))[0.74],  table.cell(fill: rgb(0, 0, 0, 60))[0.85],  table.cell(fill: rgb(50, 0, 0, 60))[0.80],
  table.cell(fill: rgb(200, 0, 0, 60))[0.65],                                                                  [@prc],                                     table.cell(fill: rgb(0, 0, 0, 60))[0.70],   table.cell(fill: rgb(0, 90, 0, 60))[0.79],  table.cell(fill: rgb(0, 100, 0, 60))[0.80], table.cell(fill: rgb(0, 0, 0, 60))[0.73],  table.cell(fill: rgb(20, 0, 0, 60))[0.71],
  table.cell(fill: rgb(250, 0, 0, 70))[0.47],  table.cell(rowspan: 2, "DrLLPS MLO"),                           table.vline(stroke: 0.5pt),                 [@roc],                                     table.cell(fill: rgb(0, 0, 0, 60))[0.68],   table.cell(fill: rgb(0, 70, 0, 60))[0.75],  table.cell(fill: rgb(0, 80, 0, 60))[0.76], table.cell(fill: rgb(0, 0, 0, 60))[0.80],
  table.cell(fill: rgb(0, 50, 0, 60))[0.85],                                                                   table.cell(fill: rgb(120, 0, 0, 60))[0.68], [@prc],                                     table.vline(stroke: 0.5pt),                 table.cell(fill: rgb(0, 0, 0, 60))[0.72],   table.cell(fill: rgb(0, 60, 0, 60))[0.78], table.cell(fill: rgb(0, 50, 0, 60))[0.77],
  table.vline(stroke: 0.5pt),                  table.cell(fill: rgb(0, 0, 0, 60))[0.72],                       table.cell(fill: rgb(0, 20, 0, 60))[0.74],  table.cell(fill: rgb(250, 0, 0, 80))[0.45], table.hline()
), caption: [Evaluation Summary of the final model on the @mlo datasets. The best for each row are marked in bold. The values for PSPire and PdPS are taken from the PSPire article. @hou_machine_2024]) <evaluation_mlo>

=== PPMC-lab Dataset

@final_ppmclab.

#figure(table(
  columns: 3,
  table.hline()
), caption: [Results from the final model on the PPMC-lab dataset.]) <final_ppmclab>

=== catGranule 2.0 Data Set
@final_cat compares the results of the final model trained on the catGranule
2.0 training dataset to the results of the other predictors on the catGranule
2.0 testing dataset. The model of this work is able to slightly outperform the
other models. As the catGranule 2.0 paper did not provide data on the @prc@auc
the comparison is not as strong as it could be. This model achieved a @prc@auc
value of 0.81 for the non-@idr and 0.71 for the @idr model.

#figure(table(
  columns: 9,
  [@auc],                                    [catGranule 1.0],           [MaGS],                                     [PSPHunter],                               [PICNIC],                                  [PICNIC-GO],                               [catGranule 2.0],                          [non-@idr model],                          [@idr model],
  [@roc],                                    table.vline(stroke: 0.5pt), table.cell(fill: rgb(140, 0, 0, 60))[0.66], table.cell(fill: rgb(60, 0, 0, 60))[0.74], table.cell(fill: rgb(60, 0, 0, 60))[0.74], table.cell(fill: rgb(70, 0, 0, 60))[0.73], table.cell(fill: rgb(50, 0, 0, 60))[0.75], table.cell(fill: rgb(40, 0, 0, 60))[0.76], table.cell(fill: rgb(0, 0, 0, 60))[0.80],
  table.cell(fill: rgb(60, 0, 0, 60))[0.74], table.hline()
), caption: [Comparison of the @roc@auc of several @llps predictors on the catGranule 2.0 test data set.
The values for all predictors but my own are taken from the catGranule 2.0 article. @monti_catgranule_2025]) <final_cat>

#pagebreak()
