#import "@preview/glossy:0.8.0": *
#import "state.typ": bib_state
#context bib_state.get()
#show: init-glossary.with(yaml("glossary.yaml"))

= Introduction

== Liquid-Liquid Phase Separation

@llps is a process where a homogeneous mixture spontaneous separates into two
liquid phases, one with a depleted and one with an increased concentration of
components. @llps can form colloids like milk or layers like a mixture of oil
and water. It is a widespread phenomena that is dependent on many factors like
temperature, pressure, differences in polarity, hydrophobicity and
hydrophilicity. Studying @llps in polymer systems started in the mid of the
20th century. In the late 20th century the phenomena was recognized as an
important process organisms and studies began to understand the implications of
@llps in an biological context. @xu_liquid-liquid_2023

Cells are capable of conducting various different tasks that involve many
biochemical reactions. As these reactions often need different educts, enzymes
and conditions a spatial organization helps the cell to conduct these reactions
effectively. Organelles are the compartments of the cell. They need a boundary
that separates them from the rest of the cell and the components inside the
compartment have to be able to move freely. Compartments that are confined by a
membrane like the nucleus or mitochondria are well known for a long time. A
more recent discovery was the existence of @mlo. They are created through
@llps. An example of one such @mlo is the nucleoli, but there are many other
@mlo that play important roles in cell and are the reason cells are able to do
these many diverse reactions. @xu_liquid-liquid_2023

The macro molecules responsible for @llps in cells are proteins and nucleic acids.
A combination of simultaneous weak and strong interactions between proteins and
proteins or proteins and nucleic acids seems to be the driving force. The presence of
many different binding sites - multivalency is a crucial parameter. The solubility
also effects the propensity to undergo @llps, as many proteins that undergo @llps
have a poor solubility in water. Other than proteins with multiple domains, @idp::pl
are often involved in @llps. They differ from globular molecules in two ways. They
do not posses a fixed conformation and they only use a small subset of the available
amino acids. Their function is generally less dependent on their exact sequence than
on more general characteristics like charge patterns. @alberti_phase_2017

As proteins that take part in @llps can serve different roles, groupings were
created. One way to organize them is to put them into the two groups driver (or
scaffolds) and clients. The driver proteins, that either induce the formation
of a condensate or are essential for the integrity of a condensate and the
clients that require a driver protein to form a condensate.
@rostam_cd-code_2023 There is also another slightly different categorization.
@llps proteins can also be divided into PS-Self and PS-Part. PS-Self
(self-assembling phase-separating) are proteins that are able to form
condensates on their own, while PS-Part (partner-dependent phase-separating)
needs a partner protein. @noauthor_about-phasepred_nodate

As @llps is extremely sensitive to changes in physico-chemical conditions, it
is possible that it also plays an important role in stress adaptation
@alberti_phase_2017. However, @llps is not always wanted. In some cases
proteins undergo @llps that normally would not. This can be due to several
reasons like mutations or posttranslational modifications. These unwanted
aggregates are suspected to lead to diseases like cancer or neurodegenerative
diseases. @xu_liquid-liquid_2023

== Phase Separation Predictors

With the rise of interest on the topic of @llps, tools were developed to
predict @llps propensity for proteins. These tools use sequential features of
proteins in many cases in combination with other data like @ptm::pl or
structural data. While these tools improved over time, there is still room for
improvement. Developing a model to predict @llps propensity comes with some
challenges. One big problem is the sparse experimental data on proteins that
undergo @llps. After filtering out proteins that are similar to each other,
less than 700 proteins remain. Defining a good negative data set is also
difficult as there is no data base that collects non-@llps proteins and
negative data is often not published. It is also a challenge to test if a
protein takes part in @llps, as it is context dependent, meaning the conditions
have to be right and for partner dependent proteins, said partner must be
available. Another issue is, that current tools tend to favor proteins that
contain @idr::pl or are labeled as partner-dependent. As partner-dependent
proteins usually lack @idr::pl this bias is probably caused by the same reason.

@pspredictors summarizes some of the @llps predictors. The predictors PhaSePred
and PSPire are explained more thoroughly.

#figure(table(
  columns: 4,
  align: (left, left, center, center),
  "Name",        "Information used for prediction",                                 "Release Article", "Source",
  "PLAAC",       [Amino acid frequencies in Prion Like Domains of _S. cerevisiae_], "2014",            [@lancaster_plaac_2014],
  "catGRANULE",  "Nucleic acid binding, disorder, length, R/G/F content",           "2016",            [@bolognesi_concentration-dependent_2016],
  "PScore",      "Pi-Pi contact frequencies",                                       "2018",            [@vernon_pi-pi_2018],
  "PSPer",       "Domain arrangements in FUS-like proteins",                        "2019",            [@orlando_computational_2019],
  "FuzDrop",     "",                                                                "2020",            [@hardenberg_widespread_2020],
  "PSAP",        "",                                                                "2021",            [@mierlo_predicting_2021],
  "PSPredictor", "",                                                                "2022",            [@chu_prediction_2022],
  "PdPS / SaPS", "",                                                                "2022",            [@chen_screening_2022],
  "PSPire",      "",                                                                "2024",            [@hou_machine_2024],
  table.hline()
), caption: [Summary of recent Liquid-liquid phase separation predictors.]) <pspredictors>

=== PhaSePred - PdPS / SaPS

PhaSePred is a meta predictor that uses some of the @llps predictors already
listet in @pspredictors (catGranule, PLAAC, PScore), an @idr predictor
(ESpritz), an low-complexity region predictor (LCR), hydropathy prediction from
CIDER, coiled-coil domain predictor DeepCoil, the immunofluorescence
image-based droplet-forming propenity predictor DeepPhase.

The tools PdPS and SaPS were specifically developed to tackle the challenge of
predicting PdPS. Both models have a version with 8 features, designed for
all-species data, as well as models with 10 features for human proteins, as
there is more data available. The eight features are hydropathy, fraction of
charged residues, @idr, low-complexity regions, PScore, PLAAC, catGranule and
coiled coil domains. The two additional features are the Phos frequency and the
IF image-based droplet forming propensities. Dividing their model into two
helped them to outperform other tools when it comes to the prediction of
partner-dependent @llps. They used XGBoost for their model. @noauthor_about-phasepred_nodate

=== PSPire

As current predictors still struggle to predict non-@idr @llps proteins PSPire
was developed. Their approach was to split protein data into @idr related
features and @ssup related features. The @ssup features only contained the
amino acids with an @rsa greater than 25%. The @ssup were calculated with
AlphaFold while the @rsa values were calculated with PSAIA from the AlphaFold
data. For both groups 44 features were calculated. Those features were fraction
of the amino acid per amino acid, fraction of several other groups of amino
acids (e.g. positively charged group), averaged isoelectric point value,
averaged molecular weight, averaged hydropathy score, averaged polarity score,
sequence length of @idr::pl, sequence length percentage of @idr::pl, total
charged sticker number divided by the residue number in @ssup, charged sticker
pair number divided by the residue number in @ssup and number of
phosphorylation sites divided by the length of the protein sequence. They also
used XGBoost as a model and were able to perform as good as the best other
predictors for @llps proteins containing @idr::pl and significantly better for
@llps proteins that do not contain @idr::pl. @hou_machine_2024

== Artificial Intelligence in Bioinformatics
Artificial Intelligence has brought new revolutionizing approaches to biological
questions. Applications like @ml, deep learning, or natural language processing
are already widely used and have helped to develop powerful bioinformatic tools.
These tools are used in a variety of bioinformatic domains such as genome
sequencing, protein structure prediction, drug discovery, systems biology,
personalized medicine, imaging, signal processing and text mining.
@jamialahmadi_artificial_2024

For example, the grounbraking tool AlphaFold, a @ml model developed by google is
able to predict the structure of proteins with almost experimental precision.
@jumper_highly_2021

=== Convolutional Neural Networks
@cnn belong to the @dl models. Along with Recurrent Neural Networks, Generative
Networks and Transformers they excel in tasks like image recognition or natural
language processing. They are mainly known for their success in computer vision,
yet they can also be used in other domains like natural language processing.
@ersavas_novel_2024

=== Long short-term memory
@lstm are able to identify patterns over longer distances.

=== Transformers
Transformers have emerged as a novel @dl model that is great at working with
time series data or sequences. It is able to capture long-term relations.
@ersavas_novel_2024

== Block Decomposition of Protein Sequences

The block decomposition algorithm by Martin Girard was created as part of a
surrogate model for low complexity protein sequences. The model itself is based
on combinatorics on words, particular sturmian words and their generalizations.
It was able to show that low complexity protein sequences have similar
properties to homopolymers with equivalent parameters, shown by their radius of
gyration. Changes to the radius of gyration are strongly correlated to changes in
the @llps propensity of a protein. @noauthor_files_2024

The block decomposition algorithm itself uses word balance as a measure of
homogeneity. It finds the longest segments of the sequence, that have a word
balance below the threshold. As the algorithm originates from the field of
polymer physics these segments are called blocks. A word $w$ is $q$-balanced
if, for any two equal-length substrings $u$ and $v$ within $w$, the count of
each character differs by no more than $q$. Once the largest homogeneous block
is identified, the algorithm recursively searches for the largest blocks to the
left and right of it. Segments shorter than a predefined length threshold are
discarded. Before applying the block decomposition algorithm a mapping is
applied to be less sensitive to mutations. @noauthor_files_2024

== Phase Separation Predictor using Block Decomposition and Neural Networks

The goal of this work is to develop @llps predictors using machine learning
methods. Two approaches will be tested. The first is using neural networks as
models. Current predictors use more conventional methods like random forest or
XGBoost models that take scalar values as inputs. A lot of information is lost
using only scalar values like fractions. Using models that take the whole
sequence may lead to better performance or models that add to the current
landscape of @llps predictors. The second approach involves using the block
decomposition algorithm with different mappings to divide protein sequences
into blocks of a certain homogeneity, where each mapping captures different
types of information. This output can then be used as input for neural networks
or applied in more traditional analysis methods. In contrast to other models
that simply consider the fraction of individual or grouped amino acids, this
approach only counts amino acids that occur within blocks of compositionally
similar residues.

#pagebreak()
