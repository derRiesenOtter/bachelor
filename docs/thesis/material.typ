#import "@preview/glossy:0.8.0": *
#import "state.typ": bib_state
#context bib_state.get()
#show: init-glossary.with(yaml("glossary.yaml"))

= Material

== Datasets

Four datasets were used during this this work. They are listed in @datasets.
The composition and reason for their inclusion is provided in the following
sections.

#figure(table(
  columns: 3,
  "Article",                                                            "Dataset Name",                    "Description",
  table.cell([PSPire @hou_machine_2024]),                               "Supplementary Data 4",            "Training and Testing Data",
  table.cell(rowspan: 3, [PhasePred @chen_screening_2022], fill: none), "Dataset S02",                     "Training Data",
                                                                        "Dataset S03",                     "Test Data",
                                                                        "Dataset S06",                     "MLO Data",
  [PPMC-lab @pintado-grima_confident_2024],                             "datasets",                        "Proteins classified into driver, client or negative",
  table.cell([CatGranule 2.0 @monti_catgranule_2025]),                  "13059_2025_3497_MOESM4_ESM.xlsx", "Training and Testing Data",
  table.hline()
), caption: [Datasets used during this work.]) <datasets>

== Protein Features

The protein features, like protein sequence, structural data and @ptm::pl were
downloaded from the sources listed in @features. The sequences for the PPMC-lab
dataset were already included and therefore not downloaded again. For the PhasePred
dataset the @ptm::pl were not downloaded.

#figure(table(
  columns: 3,
  [Protein Feature],  [Source],                                       [Version / Date],
  [Protein Sequence], [UniProt @the_uniprot_consortium_uniprot_2025], [PSPire: 19.05.2025 \
  PhasePred 23.05.2025 \
  PSPire @mlo: 28.05.2025 \
  PhasePred @mlo 28.05.2025 \
  catGranule 2.0: 24.06.2025],
  [Structural Data],  [AlphaFold @jumper_highly_2021],                [V2.0],
  [@ptm],             [UniProt @the_uniprot_consortium_uniprot_2025], [PSPire: 12.06.2025 \
  PSPire @mlo: 12.06.2025 \
  PPMC-lab: 16.06.2025 \
  catGranule 2.0: 25.06.2025],
  table.hline()
), caption: [Sources for protein features used during this work.]) <features>

== PhaSePred Data Set

The dataset to develop the PhaSePred model contains proteins of 49 organisms.
They were taken mainly from PhaSepDB @hou_phasepdb_2023. Some were also taken
from LLPSDB @wang_llpsdb_2022 and PhaSePro @meszaros_phasepro_2020. To reduce
redundancy the CD-HIT @fu_cd-hit_2012 algorithm was used. The positive @llps
proteins consist of 201 self-assembling and 327 partner-dependent proteins. The
positive training set consisted of 128 self-assembling and 214
partner-dependent proteins. The remaining positives were used for independent
testing. @chen_screening_2022

The negative dataset consisted of 10 proteomes (_Homo sapiens_, _Saccharomyces
cerevisiae_, _Mus musculus_, _Drosophila melanogaster_, _Caenorhabditis
elegans_, _Rattus norvegicus_, _Xenopus laevis_, _Arabidosis thaliana_,
_Escherichia coli_, _Schizosaccharomyces pombe_). The CD-HIT algorithm was
again used to reduce redundancy. 60,251 proteins remained. 20 % were used in
the testing set, while the other 80 % were used in the training set.
@chen_screening_2022

This data set was used to test the influence of grouping @llps
proteins into the two categories of self-assembling and partner-dependent
@llps proteins.

== PSPire Data Set

The PSPire dataset is based on the dataset of the PhaSePred models. It also
incorporates proteins from LLPSDB @wang_llpsdb_2022, PhaSePro
@meszaros_phasepro_2020, PhaSepDB @hou_phasepdb_2023, and DRLLPS
@ning_drllps_2020. The PSPire training data set contains 259 positives with 195
of these labeled as @idp::pl and 64 labeled as non-@idp::pl as well as 8323
negative entries. The testing data set contains 258 positive entries with 194
labeled as @idp::pl and 64 labeled as non-@idp::pl as well as 1961 negative entries.

For evaluation, five human @mlo datasets were collected: G3BP1 proximity
labeling set @yang_g3bp1_2020, DACT1-particulate proteome set
@esposito_tgf--induced_2021, RNAgranuleDB Tier1 set @youn_properties_2019, the
PhaSepDB low and high throughput MLO @hou_phasepdb_2023 set as well as the
DRLLPS MLO set @ning_drllps_2020.

The PSPire dataset was used due to its focus on @llps proteins that contain
no @idr and because it enables a broad comparison between many @llps predictors.
It also features the @mlo datasets to further evaluate and compare model performance.

== PPMC-lab Data Set

The PPMC-lab data set was specifically designed to help developing better Phase
Separation Predictors. It provides a curated negative dataset as well as
categorizations for @llps proteins into driver and clients. It contains 784
positives, where 367 are clients, 358 are drivers and 59 are both. The negative
set contains 2121 proteins where 1120 are structured proteins and 1001 are
disordered proteins. The proteins are collected from DRLLPS @ning_drllps_2020,
LLPSDB @wang_llpsdb_2022, PhaSepDB @hou_phasepdb_2023, PhaSePro
@meszaros_phasepro_2020, CD-Code @rostam_cd-code_2023, DisProt
@aspromonte_disprot_2024 and PDB @berman_protein_2000.

The PPMC-lab data set was chosen because it provides a curated negative data
set and to compare models trained on this dataset with models trained on the
PSPire dataset.

== CatGranule 2.0 Data set

The CatGranule 2.0 data set consists of 3333 positive and 3252 negative
proteins in the training data set and 1422 positives and 1376 negatives in the
testing data set. The sources of the proteins are similar to the other data
sets and include PhaSepDB @hou_phasepdb_2023, PhaSePro @meszaros_phasepro_2020,
LLPSDB @wang_llpsdb_2022, CD-Code @rostam_cd-code_2023, DRLLPS
@ning_drllps_2020 as well as data from the PRALINE database
@vandelli_praline_2023, proteins from previous @llps predictors
@kuechler_comparison_2023 and from an article about properties of stress
granule and P-body proteomes @youn_properties_2019. Like the previous studies
CD-Hit was used to filter out too similar proteins. The negative data set is
created by using the human proteome and excluding the proteins of the positive
data set and their first interactors. @monti_catgranule_2025

This data set was used to be able to compare the tool created in this work with
newer @llps predictors that claim to be the top performing state-of-the art
methods. This includes CatGranule 2.0, MaGS, PICNIC, and PICNIC-GO.
@monti_catgranule_2025

#pagebreak()
