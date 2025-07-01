#import "@preview/glossy:0.8.0": *
#set math.equation(numbering: "1.")
#import "@preview/cetz:0.4.0"
#import "state.typ": bib_state
#context bib_state.get()
#show: init-glossary.with(yaml("glossary.yaml"))
#set heading(numbering: "1.")

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

While there have been tools that use @hmm::pl they were later outperformed by
@ml models. This is due to phase separation being driven by complex interactions
and not being dependent on particular motifs. @hmm::pl also struggle on low-complexity
and disordered regions, which are common and important in @llps proteins. While
@hmm::pl are usually limited to sequence information alone, @ml models can
incorporate other features like @ptm::pl.

@pspredictors summarizes some of the most important @llps predictors that were
created over the last few years. The predictors PhaSePred and PSPire are
explained more thoroughly afterwards, as they both tackle the difficulty
of @llps prediction in unique ways.

#figure(table(
  columns: 3,
  align: (left, left, center, center),
  "Name",                                               "Information used for prediction",                                                                                                                                                                                                        "Release Article",
  [PLAAC @lancaster_plaac_2014],                        [@hmm to find prion like domains],                                                                                                                                                                                                        "2014",
  [catGRANULE @bolognesi_concentration-dependent_2016], [Scoring Formula that uses RNA bindings, structural disorder propensity, amino acid patterns and polypeptide length],                                                                                                                     "2016",
  [PScore @vernon_pi-pi_2018],                          [Scoring Formula that uses $pi$-$pi$ interaction potential],                                                                                                                                                                              "2018",
  [PSPer @orlando_computational_2019],                  [Modified @hmm to find prion like domains that undergo @llps],                                                                                                                                                                            "2019",
  [FuzDrop @hardenberg_widespread_2020],                [Logistic Regression model using disordered binding modes and the degree of protein disorder],                                                                                                                                            "2020",
  [MaGS @kuechler_distinct_2020],                       [Generalized Linear Model that uses protein abundance, percentage of protein intrinsic disorder, number of annotated phosphorylation cites, PScore, Camsol score, RNA interaction, and percentage of composition of leucine and glycine], "2020",
  [PSAP @mierlo_predicting_2021],                       [Random Forest Classifier that uses amino acid composition, sequence length, isoelectric point, molecular weight, GRAVY, aromaticity, secondary structure content, intrinsic disorder scores, hydrophobicity profiles],                   "2021",
  [PSPredictor @chu_prediction_2022],                   [Gradient Boosting Decision Tree that uses Word2Vec embeddings of sequence k-mers],                                                                                                                                                       "2022",
  [PdPS / SaPS @chen_screening_2022],                   [see @phasepred],                                                                                                                                                                                                                         "2022",
  [PSPire @hou_machine_2024],                           [see @pspire],                                                                                                                                                                                                                            "2024",
  [PSPHunter @sun_precise_2024],                        [Random Forest Model that uses amino acid composition, evolutionary conservation, predicted functional site annotations, word embedding vectors, protein annotation information and protein–protein interaction network features],        "2024",
  [PICNIC @hadarovich_picnic_2024],                     [Catboost classifier that uses disorder scores, sequence complexity, sequence distance-based features, secondary structure based features and Gene Ontology Features (for their GO-model)],                                               "2024",
  [catCRANULE 2.0 @monti_catgranule_2025],              [Multilayer Perceptron that uses a combination of 82 physico-chemical and 46 structural and RNA-binding features],                                                                                                                        "2025",
  table.hline()
), caption: [Summary of recent Liquid-liquid phase separation predictors.]) <pspredictors>

=== PhaSePred - PdPS / SaPS <phasepred>

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

=== PSPire <pspire>

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
@ai has brought new approaches to biological questions. Applications like @ml,
@dl, or natural language processing are already widely used and have
helped to develop powerful bioinformatic tools. These tools are used in a
variety of bioinformatic domains such as genome sequencing, protein structure
prediction, drug discovery, systems biology, personalized medicine, imaging,
signal processing and text mining. Support Vector machines, random forests and
@nn::pl are just a few of the @ai models employed in bioinformatics.
@jamialahmadi_artificial_2024

As the techniques of @ai models differ they are often categorized into @ml, @dl
and @nn as shown in @ai_cat. Although @nn and @dl are
technically subfields of @ml, we distinguish between them in this work for
clarity. Therefore, when we refer to @ml in this text, it explicitly excludes
@nn. @dl are @nn that consist of three or more layers. @noauthor_ai_2023

Comparing @ml models to @dl models, there are some important differences. @ml
models usually perform better on small data sets ($< 10000$). @dl models
tend to over fit if the sample size is not large enough. @ml models also need
less computational resources and can therefore be trained faster and on lower
end devices @noauthor_pdf_2025. They are also easier to interpret. @dl models
on the other hand are able to learn more complex associations than @ml models
and are therefore able to outperform @ml models in complex scenarios were there
is enough training data. The input data for these models also differs. While @ml
models usually require tabular unstructured data, @dl are able to handle
structured data.

Basic models already perform well. If there is only limited data there is no
benefit in making the model more complex. @zeng_convolutional_2016

#figure(cetz.canvas({
  import cetz.draw: *
  circle((3.5, 0), radius: (4, 1), fill: rgb(0, 200, 250, 50))
  content((rel: (2, 0)), [@ai])
  circle((1.5, 0), radius: (2, 0.75), fill: rgb(0, 200, 250, 100))
  content((rel: (1, 0)), [@ml])
  circle((0.5, 0), radius: (1, 0.5), fill: rgb(0, 200, 250, 150))
  content((rel: (0.5, 0)), [@nn])
  circle((0, 0), radius: (0.5, 0.25), fill: rgb(0, 200, 250, 200))
  content((), [@dl])
}), caption: [Categorization of @ai models]) <ai_cat>

One well known tool in bioinformatics that was created using @ai was AlphaFold.
It is a @dl model developed by google that is able to predict the structure of
proteins with almost experimental precision. It uses @msa::pl and is based on
Bidirectional Encoder Representations from Transformers.
@jumper_highly_2021

In the following sections @ai models used during this work will be covered briefly.

=== Neural Networks
@nn are inspired by nature, they consist of an input layer, an arbitrary number
of hidden layers and an output layer. They imitate the complex connected networks of
neurons in a living organism. Both biological and artificial neurons receive
signals from multiple neurons and transmit signals to multiple neurons. An
artificial neuron computes a weighted sum of its inputs, adds a bias term and
then applies an activation function to produce its output, see @neuron.
@han_artificial_2018

#figure(image("figures/neuron.jpg", width: 60%), caption: [Visualization of an
artificial neuron. @han_artificial_2018]) <neuron>

The output of a neuron $i$ with $n$ inputs $x$, the weights $w$ and an activation
function could be described as:

$ "output"_i = "activation function"(sum_j^n w_(i j) x_j + b_i) $

The weights and the biases are the learnable parameters of a @nn. They are
initialised with random values and are updated each training epoch, see
@nn_weights.

#figure(image("figures/nn_weights.jpg", width: 100%), caption: [Visualization of the weights in a @nn @han_artificial_2018]) <nn_weights>

Updating these values uses an algorithm called backpropagation,
see @backpropagation. After the model makes predictions, the difference between
the predicted output and the label is calculated. This value is calculated by
using the loss function and is called loss. Backpropagation then computes how
much each weight and bias contributed to that loss by calculating the
gradients. Rather than calculating an exact solution for all parameters, which
is computationally intractable in deep networks, neural networks use an
optimization technique called gradient descent. This method uses the computed
gradients to update each parameter in the direction that reduces the loss most
efficiently. Over many training epochs, the model gradually learns better
weights and biases that minimize the prediction error. Gradient descent is not
guaranteed to find the global minima of the loss, as it is able to get stuck in
a local minima. @han_artificial_2018

#figure(image("figures/backpropagation.jpg", width: 70%), caption: [Visualization of the backpropagation mechanism. @han_artificial_2018]) <backpropagation>

@af::pl are an important part of a @nn. Their main responsibility is adding
non-linearity to the network. Without these a @nn would just produce the
outputs as a linear function of the inputs. An activation function should also
have some more properties. It should be easy to calculate, it should be
differentiable and it should retain the distribution of data. The first common
@af was the sigmoid function, see @sigmoid.
$ sigma(x)=frac(1, 1 + e^(-x)) $ <sigmoid>
Today the @relu function is commonly used, as it is computational less complex, see @relu.
$ "ReLU"(x) = cases(x ", if" x gt.eq 0, 0 ", otherwise") $ <relu>

There are also other more advanced @af::pl today, that do have some advantages
over the here mentioned ones, but they are not in focus of this work. @dubey_activation_2022

Today there are several types of @nn, each designed to handle different tasks.
Some of these used during this work will be introduced in the following sections.

==== Convolutional Neural Networks
@cnn are part of the @nn models. Along with Recurrent Neural Networks, Generative
Networks and Transformers they excel in tasks like image recognition or natural
language processing. They are mainly known for their success in computer vision,
yet they can also be used in other domains like natural language processing.
@ersavas_novel_2024

==== Long short-term memory
@lstm are able to identify patterns over longer distances.

==== Transformers
Transformers have emerged as a novel @dl model that is great at working with
time series data or sequences. It is able to capture long-term relations.
@ersavas_novel_2024

=== XGBoost

XGBoost eXtreme Gradient Boosted trees is an ensemble method. Boosting models
take many weak learners and add them sequentially to their prediction. Each new
weak learner is trained to correct the errors the previous tree caused. It has
several features that set it apart from other models. It has regularized boosting
to prevent overfitting.

ion
automatically handles missing values, is able to work
in parallel, can cross validate at each iteration and uses tree pruning.

is a powerful gradient tree boosting algorithm that belongs to the @ml
models. It is a tree ensemble model, where the final prediction for a given
sample is the sum of predictions from all trees. @chen_xgboost_2016

the outputs of many weak learners to form a strong predictive model.
Each new tree that is added to the model is trained to correct for the errors of
the last tree.

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
