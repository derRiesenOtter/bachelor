#import "@preview/glossy:0.8.0": *
#import "@preview/subpar:0.2.2"
#import "state.typ": bib_state
#context bib_state.get()
#show: init-glossary.with(yaml("glossary.yaml"))

= Results

== Data Preparation

== Final Model

=== PSPire Data Set

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

@roc, @prc
@final_model_pspire_table shows the @auc values for my model, PSPire and PdPS.
#figure(table(
  columns: 7,
  "AUC",  "My Model",    "PSPire",      "PdPS",            "My Model", "PSPire",      "PdPS",
  "",     table.cell(colspan: 3, [*@idr*], align: center), table.cell(colspan: 3, [*non-@idr*], align: center),
  [@roc], [0.79],        [*0.86*],      [0.84],            [*0.88*],   [0.84],        [0.68],
  [@prc], table.vline(), [0.37],        [*0.51*],          [0.42],     table.vline(), [*0.25*],
  [0.24], [0.08],        table.hline()
), caption: [Comparison of the @auc values for my model, PSPire and PdPS]) <final_model_pspire_table>

=== Evaluation on the @mlo data sets

@evaluation_mlo compares the @auc values for the five @mlo datasets.

#figure(table(
  columns: 8,
  "Dataset",                                                      "AUC",                      "My Model",    "PSPire", "PdPS",                 "My Model", "PSPire",      "PdPS",
  table.cell(fill: none, ""),                                     table.cell(fill: none, ""), table.cell(colspan: 3, align: center, [*@idr*]), table.cell(colspan: 3, [*non-@idr*], align: center),
  table.cell(rowspan: 2, "G3BP1 proximity labeling"),             [@roc],                     [0.74],        [*0.91*], [0.86],                 [*0.96*],   [0.93],        [0.81],
                                                                  [@prc],                     [0.29],        [*0.58*], [0.41],                 [0.51],     [*0.66*],      [0.18],
  table.cell(rowspan: 2, "DACT1-particulate proteome"),           [@roc],                     [0.72],        [*0.88*], [0.85],                 [0.90],     [*0.93*],      [0.81],
                                                                  [@prc],                     [0.22],        [*0.35*], [0.33],                 [0.49],     [*0.60*],      [0.18],
  table.cell(rowspan: 2, "RNAgranuleDB Tier1"),                   [@roc],                     [0.80],        [*0.84*], [0.82],                 [0.88],     [*0.90*],      [0.68],
                                                                  [@prc],                     [0.39],        [*0.48*], [0.42],                 [0.18],     [*0.28*],      [0.08],
  table.cell(rowspan: 2, "PhaSepDB low and high throughput MLO"), [@roc],                     [0.71],        [0.72],   [*0.74*],               [*0.85*],   [0.80],        [0.65],
                                                                  [@prc],                     [0.70],        [0.79],   [*0.80*],               [*0.73*],   [0.71],        [0.47],
  table.cell(rowspan: 2, "DrLLPS MLO"),                           [@roc],                     [0.69],        [0.75],   [*0.76*],               [0.80],     [*0.85*],      [0.68],
                                                                  [@prc],                     table.vline(), [0.71],   [*0.78*],               [*0.77*],   table.vline(), [0.72],
  [0.74],                                                         [0.45],                     table.hline()
), caption: [Evaluation Summary of the final model on the @mlo datasets. The best for each row are marked in bold.]) <evaluation_mlo>

#pagebreak()
