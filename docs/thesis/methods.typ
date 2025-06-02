#import "@preview/glossy:0.8.0": *
#import "state.typ": bib_state
#context bib_state.get()
#show: init-glossary.with(yaml("glossary.yaml"))

= Methods

== Programs

The programs used during this work are listed in @programs.
#figure(table(
  columns: 3,
  "Program",                            "Package",                                   "Version",
  table.cell(rowspan: 10, "Python"),    "-",                                         "3.13.3",
                                        [numpy @harris_array_2020],                  "2.2.6",
                                        [bio @cock_biopython_2009],                  "1.8.0",
                                        [pandas @mckinney_data_2010],                "2.2.3",
                                        [matplotlib @hunter_matplotlib_2007],        "3.10.3",
                                        [requests ],                                 "2.32.3",
                                        [scikit-learn @pedregosa_scikit-learn_2011], "1.6.1",
                                        [seaborn @waskom_seaborn_2021],              "0.13.2",
                                        [torch @noauthor_pytorch_nodate],            "2.7.0",
                                        [xgboost @noauthor_xgboost_nodate],          "3.0.2",
  [DSSP @noauthor_pdb-redodssp_nodate], "-",                                         "4.5.3",
  table.hline()
), caption: "Programs used in this work") <programs>

== Block Decomposition

== Models

#pagebreak()
