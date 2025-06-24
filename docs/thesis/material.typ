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
@jumper_highly_2021. The posttranslational modification data was also
downloaded from UniProt @the_uniprot_consortium_uniprot_2025.

== PhaSePred Data Set

The data set to develop the PhaSePred model contains proteins of 49 organisms.
They were taken mainly from PhaSepDB. Some were also taken from LLPSDB
@wang_llpsdb_2022 and PhaSePro @meszaros_phasepro_2020. To reduce redundancy
the CD-HIT @fu_cd-hit_2012 algorithm was used. The positive proteins consist of 201
self-assembling proteins and 327 partner-dependent proteins. The positive
training set consisted of 128 self-assembling and 214 partner-dependent
proteins. The remaining positives were used for independent testing.

The negative datasets consists of 10 of the proteomes (_Homo sapiens_,
_Saccharomyces cerevisiae_, _Mus musculus_, _Drosophila melanogaster_,
_Caenorhabditis elegans_, _Rattus norvegicus_, _Xenopus laevis_, _Arabidosis
thaliana_, _Escherichia coli_, _Schizosaccharomyces pombe_) that are present in
the training data set. The CD-HIT algorithm was again used to reduce
redundancy. 60,251 proteins remained. 20 % were used in the testing set, while
the other 80 % were used in the training set.

== PSPire Data Set

The PSPire data set is based on the data set of the PhaSePred models. It also
incorporates proteins from LLPSDB @wang_llpsdb_2022, PhaSePro
@meszaros_phasepro_2020, PhaSepDB @hou_phasepdb_2023, and DRLLPS
@ning_drllps_2020. The PSPire training data set contains 259 positives with 195
containing at least one IDR and 64 without one and 8323 negative entries. The
testing data set contains 258 positive entries with 194 containing at least one
IDR and 64 without one and 1961 negative entries.

For evaluation, five human @mlo datasets were collected: G3BP1 proximity
labeling set, DACT1-particulate proteome set, RNAgranuleDB Tier1 set and the
PhaSepDB low and high throughput MLO set and the DRLLPS MLO set.

== PPMC-lab Data Set

The PPMC-lab data set was created to help developing better Phase Separation
Predictors. It tries to solve the challenge of the selection of an appropriate
negative training data set. Current models use negative data sets that do not
contain experimentally tested negatives. This data set also differentiates between
driver and client.
It contains 784 positives, where 367 are clients, 358 are drivers and 59 are both.
The negative set contains 2121 proteins where 1120 are structured proteins and 1001
are disordered proteins. The proteins are collected from DRLLPS
@ning_drllps_2020, LLPSDB @wang_llpsdb_2022, PhaSepDB @hou_phasepdb_2023,
PhaSePro @meszaros_phasepro_2020, CD-Code @rostam_cd-code_2023, DisProt
@aspromonte_disprot_2024 and PDB @berman_protein_2000.

#pagebreak()
