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
  Model],                    table.cell(rowspan: 2)[@auc], table.cell(colspan: 2, [@idr], align: center),                                 table.cell(colspan: 2, [non-@idr], align: center),
                                                            table.cell(fill: none)[Sequence], table.cell(fill: none)[Block Decomposition], table.cell(fill: none)[Sequence], table.cell(fill: none)[Block Decomposition],
  table.hline(stroke: 0.5pt), table.cell(rowspan: 2)[PPMC-lab / \
  1 Layer],                    table.vline(stroke: 0.5pt),       [@roc],                                      [0.87],                           [*0.88*],
  [*0.65*],                                                 [0.61],                           [@prc],                                      table.vline(stroke: 0.5pt),       [*0.66*],
  [0.63],                     table.vline(stroke: 0.5pt),   [*0.72*],                         [0.71],                                      table.cell(rowspan: 2)[PPMC-lab / \
  2 Layer],                        [@roc],
  [*0.88*],                   [0.86],                       [*0.69*],                         [*0.69*],                                                                      [@prc],
  table.vline(stroke: 0.5pt), [*0.68*],                     [0.63],                           table.vline(stroke: 0.5pt),                  [0.73],                           [*0.78*],
  table.cell(rowspan: 2)[PSPire / \
  1 Layer],                  [@roc],                       [*0.70*],                         [0.68],                                      [*0.63*],                         [0.59],
                              [@prc],                       table.vline(stroke: 0.5pt),       [*0.20*],                                    [0.20],                           table.vline(stroke: 0.5pt),
  [*0.06*],                   [0.04],                       table.cell(rowspan: 2)[PSPire / \
  1 Layer],                        [@roc],                                      [*0.79*],                         [0.72],
  [*0.62*],                   [0.56],                                                         [@prc],                                      table.vline(stroke: 0.5pt),       [*0.37*],
  [0.26],                     table.vline(stroke: 0.5pt),   [*0.05*],                         [0.04],                                      table.hline()
), caption: [Comparison of the @auc values for the one and two layer @cnn
with the block decomposition and the raw sequence as input. PPMC-lab dataset.]) <first_models>

== The Second Set of Models

== Final Model

=== PPMC-lab Dataset

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
  caption: [Results of the final model on the PSPire data. (a, b) ROC-AUC and
  PR-AUC for Proteins containing @idr::pl. (c, d) ROC-AUC and PR-AUC for
  Proteins containing no @idr::pl. ],
  label: <final_model_pspire>,
)

#figure(table(
  columns: 7,
  table.cell(rowspan: 2)[@auc], table.cell(colspan: 3, [@idr], align: center),                                                  table.cell(colspan: 3, [non-@idr], align: center),
                                table.cell(fill: none)[My Model], table.cell(fill: none)[PSPire], table.cell(fill: none)[PdPS], table.cell(fill: none)[My Model], table.cell(fill: none)[PSPire], table.cell(fill: none)[PdPS],
  table.hline(stroke: 0.5pt),   [@roc],                           [0.79],                         [*0.86*],                     [0.84],                           [*0.88*],                       [0.84],
  [0.68],                       [@prc],                           table.vline(stroke: 0.5pt),     [0.37],                       [*0.51*],                         [0.42],                         table.vline(stroke: 0.5pt),
  [*0.25*],                     [0.24],                           [0.08],                         table.hline()
), caption: [Comparison of the @auc values for my model, PSPire and PdPS. The values for PSPire and PdPS are taken from the PSPire article. @hou_machine_2024]) <final_model_pspire_table>

=== Evaluation on the @mlo data sets

@evaluation_mlo compares the @auc values for the five @mlo datasets. For
proteins that do contain @idr::pl the model created in this work does perform
worse than both PSPire and PdPS. For proteins containing no @idr::pl however
this model outperforms PdPS consistently and directly competes with PSPire.

#figure(table(
  columns: 8,
  table.cell(rowspan: 2)[Dataset], table.cell(rowspan: 2)[AUC],                                    table.cell(colspan: 3, align: center, [@idr]),                                                  table.cell(colspan: 3, [non-@idr], align: center),
                                                                                                   table.cell(fill: none)[My Model], table.cell(fill: none)[PSPire], table.cell(fill: none)[PdPS], table.cell(fill: none)[My Model], table.cell(fill: none)[PSPire], table.cell(fill: none)[PdPS],
  table.hline(stroke: 0.5pt),      table.cell(rowspan: 2, "G3BP1 proximity labeling"),             [@roc],                           [0.74],                         [*0.91*],                     [0.86],                           [*0.96*],                       [0.93],
  [0.81],                                                                                          [@prc],                           [0.29],                         [*0.58*],                     [0.41],                           [0.51],                         [*0.66*],
  [0.18],                          table.cell(rowspan: 2, "DACT1-particulate proteome"),           [@roc],                           [0.72],                         [*0.88*],                     [0.85],                           [0.90],                         [*0.93*],
  [0.81],                                                                                          [@prc],                           [0.22],                         [*0.35*],                     [0.33],                           [0.49],                         [*0.60*],
  [0.18],                          table.cell(rowspan: 2, "RNAgranuleDB Tier1"),                   [@roc],                           [0.80],                         [*0.84*],                     [0.82],                           [0.88],                         [*0.90*],
  [0.68],                                                                                          [@prc],                           [0.39],                         [*0.48*],                     [0.42],                           [0.18],                         [*0.28*],
  [0.08],                          table.cell(rowspan: 2, "PhaSepDB low and high throughput MLO"), [@roc],                           [0.71],                         [0.72],                       [*0.74*],                         [*0.85*],                       [0.80],
  [0.65],                                                                                          [@prc],                           [0.70],                         [0.79],                       [*0.80*],                         [*0.73*],                       [0.71],
  [0.47],                          table.cell(rowspan: 2, "DrLLPS MLO"),                           [@roc],                           [0.69],                         [0.75],                       [*0.76*],                         [0.80],                         [*0.85*],
  [0.68],                                                                                          [@prc],                           table.vline(),                  [0.71],                       [*0.78*],                         [*0.77*],                       table.vline(),
  [0.72],                          [0.74],                                                         [0.45],                           table.hline()
), caption: [Evaluation Summary of the final model on the @mlo datasets. The best for each row are marked in bold. The values for PSPire and PdPS are taken from the PSPire article. @hou_machine_2024]) <evaluation_mlo>

=== catGranule 2.0 Data Set
@final_cat compares the results of the final model trained on the catGranule
2.0 training dataset to the results of the other predictors on the catGranule
2.0 testing dataset. The model of this work is able to slightly outperform the
other models.

#figure(table(
  columns: 8,
  [@auc],        [catGranule 1.0], [MaGS], [PSPHunter], [PICNIC], [PICNIC-GO], [catGranule 2.0], [My Model],
  [@roc],        [0.66],           [0.74], [0.74],      [0.73],   [0.75],      [0.76],           [*0.80*],
  table.hline()
), caption: [Comparison of the @roc@auc of several @llps predictors on the catGranule 2.0 test data set.
The values for all predictors but my own are taken from the catGranule 2.0 article. @monti_catgranule_2025]) <final_cat>

#pagebreak()
