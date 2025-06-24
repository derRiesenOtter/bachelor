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
  caption: [Results of the final model on the PSPire data],
  label: <final_model_pspire>,
)

#pagebreak()
