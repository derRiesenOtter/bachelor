#import "@preview/glossy:0.8.0": *
#import "state.typ": bib_state
#context bib_state.get()
#show: init-glossary.with(yaml("glossary.yaml"))

= Material

Following datasets were used in this work (see @datasets).

#figure(table(
  columns: 3,
  "Article",                                               "Dataset",              "Description",
  table.cell(rowspan: 2, [PSPire @hou_machine_2024]),      "Supplementary Data 4", "Training and Test Data",
                                                           "Supplementary Data 5", "MLO Data",
  table.cell(rowspan: 3, [PhaSepDB @chen_screening_2022]), "Dataset S02",          "Training Data",
                                                           "Dataset S03",          "Test Data",
                                                           "Dataset S06",          "MLO Data",
  [ppmclab @pintado-grima_confident_2024],                 "datasets",             "Proteins classified into driver, client or negative",
  table.hline()
), caption: "Datasets used during this work.") <datasets>

The protein sequences for the PSPire dataset and the PhaSepDB were downloaded
from UniProt @the_uniprot_consortium_uniprot_2025. The structural data used to
calculate the relative surface availability was downloaded from AlphaFold
@jumper_highly_2021.

#pagebreak()
