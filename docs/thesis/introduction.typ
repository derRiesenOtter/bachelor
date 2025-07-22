#import "@preview/glossy:0.8.0": *
#set math.equation(numbering: "(1)")
#import "@preview/cetz:0.4.0"
#import "state.typ": bib_state
#context bib_state.get()
#show: init-glossary.with(yaml("glossary.yaml"))
#set heading(numbering: "1.1")

= Introduction

@llps is a biologically important process that plays key roles in cells. It is
a major contributor to the functional compartmentalization but is also
associated with some diseases @xu_liquid-liquid_2023. Although it is possible
to experimentally determine whether a protein undergoes @llps under specific
conditions, such experiments are expensive and consume resources. As a result,
computational prediction of @llps poses an attractive alternative.

In recent years, a variety of computational tools for @llps prediction have
been developed. These tools have improved over time, incorporating new features
and knowledge as well @ml techniques @hou_machine_2024. However, their
predictive power still leaves room for improvement.

Using @nn::pl could simplify the development of @llps predictors, as they require
less feature engineering and offer potentially greater predictive power
especially as more curated data becomes available @li_deep_2019. Their ability
to use the whole sequence as input should enable them to capture patterns and
relationships between amino acids that relate to @llps. This is something more
traditional @ml models can not do. This work will therefore test if @nn::pl are
suitable for @llps prediction and how they compare to current state-of-the-art
predictors. For the main input two different approaches will be explored. One
will use the raw amino acid sequence, while the other will use multiple block
decompositions of the amino acid sequence that each emphasize different
biochemical properties.

== Liquid-Liquid Phase Separation

@llps is a reversible process where a homogeneous mixture spontaneously separates
into two liquid phases, one with a depleted and one with an increased
concentration of components, see @phase_separation @xu_liquid-liquid_2023. @llps can form colloids
like milk or layers like a mixture of oil and water. It is a widespread
phenomena that is dependent on many factors like temperature, pressure,
differences in polarity and hydrophobicity. Studying @llps in
polymer systems started in the mid of the 20th century. In the late 20th
century the phenomena was recognized as an important process in organisms and
studies began to understand the implications of @llps in an biological context.
@xu_liquid-liquid_2023

#figure(image("figures/phase_separation.png", width: 70%), caption: [Visualization of LLPS.]) <phase_separation>

Cells are capable of conducting various different tasks that involve many
biochemical reactions. As these reactions often need different educts, enzymes
and conditions a spatial organization helps the cell to carry out these reactions
effectively. Organelles are the compartments of the cell. They need a boundary
that separates them from the rest of the cell and the components inside the
compartment have to be able to move freely. Compartments that are confined by a
membrane, like the nucleus or mitochondria, have been well known for a long time. A
more recent discovery was the existence of @mlo:pl. They often form via
@llps. A well known example of one such @mlo is the nucleoli. @xu_liquid-liquid_2023

The macro molecules responsible for @llps in cells are proteins and nucleic acids.
A combination of simultaneous weak and strong interactions between proteins and
proteins or proteins and nucleic acids seems to be the driving force. The presence of
many different binding sites, multi valency, is a crucial parameter. The solubility
also effects the propensity to undergo @llps, as many proteins that undergo @llps
have a poor solubility in water. Other than proteins with multiple domains, @idp::pl
are often involved in @llps. They differ from globular molecules in two ways. They
do not posses a fixed conformation and they only use a small subset of the available
amino acids. Their function is generally less dependent on their exact sequence than
on more general characteristics like charge patterns. @alberti_phase_2017

As proteins that take part in @llps can serve different roles, they were
divided into two groups. One group that is able to undergo @llps on its own and
one group that can only join existing condensates. The proteins of the first
group are called drivers, scaffolds or self-assembling proteins, while the
proteins in the second group are called clients or partner-dependent proteins.
@chen_screening_2022 @pintado-grima_confident_2024

As @llps is extremely sensitive to changes in physico-chemical conditions, it
is possible that it also plays an important role in stress adaptation
@alberti_phase_2017. However, @llps is not always wanted. In some cases
proteins undergo @llps that normally would not. This can be due to several
reasons like mutations or @ptm::pl. These unwanted aggregates are suspected to
lead to diseases like cancer or neurodegenerative diseases.
@xu_liquid-liquid_2023

== Phase Separation Predictors

=== Overview

With the rise of interest on the topic of @llps, tools were developed to
predict @llps propensity for proteins. The first generation of tools were not
specifically developed for @llps prediction. Their goals centered around
finding prion-like domains (PLAAC @lancaster_plaac_2014), finding proteins that
are associated with RNA granules (catGranule
@bolognesi_concentration-dependent_2016) or detecting proteins with high
propensities for $pi-pi$ interactions (PScore @vernon_pi-pi_2018). While these
tools were not intended for @llps prediction, their predictions all correlated
with the prediction of @llps. These tools used @hmm::pl or scoring formulas.
PSPer was one of the first tools specifically designed for @llps prediction, it
still used @hmm::pl @orlando_computational_2019. The tools that followed
started to use @ml models as they were better suited for the complex task of
@llps prediction @hardenberg_widespread_2020. With better understanding of the
driving forces of @llps and the incorporation of according features the
predictors steadily improved. A short summary of some of the current @llps
predictors including a short description is given in @pspredictors. Two
predictors, PdPS / SaPs (@phasepred) and PSPire (@pspire) will be covered in
more detail due to their relevance and performance.

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

PhaSePred is a meta-predictor that integrates multiple existing tools to
predict LLPS propensity. It combines outputs from several established LLPS
predictors listed in @pspredictors (including CatGranule, PLAAC, and PScore),
an @idr predictor (ESpritz @walsh_espritz_2012), a low-complexity region
prediction @wootton_statistics_1993, hydropathy predictions from CIDER
@vedantam_cider_2015, the coiled-coil domain predictor DeepCoil
@ludwiczak_deepcoilfast_2019, the Immunofluorescence-image-based
droplet-forming propensity predictor DeepPhase @noauthor_learning_nodate, and
phosphorylation data from PhosphoSitePlus @hornbeck_phosphositeplus_2015.

PhaSePred has two specialized submodels: PdPS and SaPS, tailored for different
@llps mechanisms. PdPS is trained and evaluated on partner-dependent
@llps proteins, while SaPS is trained and tested on self-assembling
proteins. Each of these has two versions: one for human proteins, which
utilizes the full feature set including phosphorylation frequency and
Immunofluorescence-image-based droplet propensity and one for non-human
proteins, which omits those two features due to limited data availability. By
distinguishing between partner-dependent and self-assembling proteins, the PdPS
and SaPS models achieve significantly improved predictive performance,
particularly for partner-dependent @llps proteins. They chose
XGBoost @chen_xgboost_2016 as their @ml model. @chen_screening_2022.

=== PSPire <pspire>

PSPire was developed due to the ongoing challenge of accurately predicting
@llps in proteins that do not contain @idr::pl. The key innovation in their
approach was to separate protein features into two categories: @idr related and
@ssup related features. Amino acids in @ssup::pl of the protein
were only taken into account if their @rsa value was above 25%, as amino acids
that are on the surface of a protein are more likely to contribute to the
formation of @llps. @hou_machine_2024

The structural information was taken from AlphaFold and the calculation of the
@rsa values was conducted using PSAIA @mihel_psaia_2008. Overall 44 features
were engineered from which
39 were calculated separately for @idr::pl and @ssup::pl. These include
fractions of amino acids, fractions of groups of amino acids, the averaged
scores of the: isoelectric point, molecular weight, hydropathy and polarity as
well as @idr sequence length, fraction of @idr::pl, stickers and sticker pairs
in @ssup::pl and number of phosphorylation sites. They also used XGBoost as
their @ml model and were able to significantly outperform other @llps
predictors when it comes to the prediction of non-@idr proteins.
@hou_machine_2024

=== Problems of Liquid-Liquid Phase Separation Prediction

While the prediction of proteins undergoing @llps has steadily improved,
significant challenges remain. One of the primary difficulties lies in the
complexity of @llps itself. It is driven by multiple factors, including both
strong and weak multivalent interactions @xu_liquid-liquid_2023, $pi -pi$
interactions @vernon_pi-pi_2018, and hydrophobic effects
@xu_liquid-liquid_2023. As a result, capturing the full spectrum of mechanisms
that contribute to @llps is difficult using tabular data like the current
predictors do.

Computational approaches, such as @hmm::pl, have proven useful in many areas of
bioinformatics, particularly for detecting conserved sequence motifs or
structural domains @yoon_hidden_2009. However, their applicability to predict
@llps is limited. This is because @llps is not solely determined by specific
sequence motifs or local features @xu_liquid-liquid_2023.

Another major challenge is the limited availability of experimental data,
particularly for negative samples @pintado-grima_confident_2024. This scarcity
restricts the ability of @ml models to generalize and hinders the development
of balanced training datasets. Since @ml models require a lot of data,
especially deep learning approaches, the lack of it slows down progress.

Furthermore, many existing tools perform poorly in predicting partner-dependent
and non-@idr proteins @hou_machine_2024@chen_screening_2022. As
partner-dependent proteins usually have smaller @idr contents
@zhou_two-task_2024 these problems correlate. Predictors like PhasePred and
PSPire have tried to address this issue, yet their performance on said @llps
proteins is still lacking.

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
and @nn as shown in @ai_cat. Although @nn and @dl are technically subfields of
@ml, we distinguish between them in this work for clarity. Therefore, when we
refer to @ml in this text, it explicitly excludes @nn::pl. @dl models are a
subcategory of @nn::pl that consist of three or more layers. @noauthor_ai_2023

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
}), caption: [Categorization of AI models]) <ai_cat>

Comparing @ml models to @nn::pl, there are some important differences. @ml
models usually perform better on small data sets ($< 10000$). @nn::pl tend to
overfit if the sample size is not large enough. @ml models also need less
computational resources and can therefore be trained faster and on lower end
devices. They are also easier to interpret. @nn::pl on the other hand are able
to learn more complex interactions than @ml models and are therefore able to
outperform @ml models in complex scenarios were there is enough training data.
@elmobark_evaluating_2025 \
The input data for these models also differs. While
@ml models usually require tabular unstructured data, @nn::pl are able to handle
structured data.

One well known tool in bioinformatics that was created using @ai was AlphaFold.
It is a @dl model developed by google that is able to predict the structure of
proteins with almost experimental precision. It uses @msa::pl and is based on
Bidirectional Encoder Representations from Transformers.
@jumper_highly_2021

In the following sections @ai models used during this work will be covered briefly.

=== Neural Networks <sec_nn>
@nn::pl are inspired by nature. They imitate the complex connected networks of
neurons in a living organism. Both biological and artificial neurons receive
signals from multiple neurons and transmit signals to multiple neurons. An
artificial neuron computes a weighted sum of its inputs, adds a bias term and
then applies an activation function to produce its output, see @neuron
@han_artificial_2018. A @nn consists of an input layer, an arbitrary number of
hidden layers and an output layer. In a classical @nn every neuron will have
all neurons from the previous layer as input. Such a layer is called @fcl.
@han_artificial_2018

#figure(image("figures/neuron.jpg", width: 60%), caption: [Visualization of an
artificial neuron.]) <neuron>

The output of a neuron $i$ with $n$ inputs $x$, the weights $w$ and an activation
function could be described using @activation_function.

$ "output"_i = "activation function"(sum_j^n w_(i j) x_j + b_i) $ <activation_function>

The weights and the biases are the learnable parameters of a @nn. They are
initialised with random values and are updated each training epoch, see
@nn_weights @han_artificial_2018.

#figure(image("figures/nn_weights.jpg", width: 100%), caption: [Visualization of the weights in a Neural Network.]) <nn_weights>

Updating these values uses an algorithm called backpropagation, see
@backpropagation @han_artificial_2018. After the model makes predictions, the
difference between the predicted output and the label is calculated. This value
is calculated by using a loss function and is called loss. Backpropagation then
computes how much each weight and bias contributed to that loss by calculating
the gradients. Rather than calculating an exact solution for all parameters,
which is computationally difficult in deep networks, neural networks use an
optimization technique called gradient descent. This method uses the computed
gradients to update each parameter in the direction that reduces the loss the
most. Over many training epochs, the model gradually learns better weights and
biases that minimize the prediction error. Gradient descent is not guaranteed
to find the global minima of the loss, as it is able to get stuck in a local
minima. @han_artificial_2018

#figure(image("figures/backpropagation.jpg", width: 70%), caption: [Visualization of the backpropagation mechanism.]) <backpropagation>

@af::pl are an important part of a @nn. Their main responsibility is adding
non-linearity to the network. Without these a @nn would just produce the
outputs as a linear function of the inputs. An activation function should also
have some more properties. It should be easy to calculate, it should be
differentiable and it should retain the distribution of data. The first common
@af::pl were the sigmoid function, see @sigmoid, and the tanh function, see @tanh.
$ sigma(x)=frac(1, 1 + e^(-x)) $ <sigmoid>
$ f(x) = frac(2, 1+e^(-2x)) -1 $ <tanh>
Today the @relu function is commonly used, as it is computational less complex, see @relu_eq.
$ "ReLU"(x) = cases(x ", if" x gt.eq 0, 0 ", otherwise") $ <relu_eq>

There are also other more advanced @af::pl, that do have some advantages
over the here mentioned ones, but they are not in focus of this work. @dubey_activation_2022

There are several types of @nn::pl, each designed to handle different tasks.
Some of these used during this work will be introduced in the following sections.

==== Convolutional Neural Networks

@cnn::pl are a popular class of @nn::pl that are often used for image
classification, speech recognition and many more tasks. They typically consist of four
components. A convolutional layer, a pooling layer, an activation function and
a fully connected layer. In a convolutional layer neurons of one layer are only
connected to some neurons of the previous layer. These neurons of the previous
layer are called the receptive field of the corresponding neuron in the next
layer. The output of the receptive field is calculated using a weight vector
that is called filter or kernel. This filter is slid over the whole input, see
@convolution @ratan_what_2020. The weights of the filter are the same for the
whole layer. @indolia_conceptual_2018

#figure(image("figures/convolution.png", width: 80%), caption: [Visualization of a convolutional layer with a two dimensional input. ]) <convolution>

The pooling layer typically shrinks its input further. There are different
pooling techniques. Most common are average pooling and max pooling. Similar to
the convolutional layer a window is slid over the input and the pooling
function is applied. What differs is that the window usually does not overlap
with the next window, leading to a considerable reduction in size, see
@maxpooling @noauthor_everything_2020. @indolia_conceptual_2018

#figure(image("figures/maxpooling.png", width: 40%), caption: [Visualization of max pooling. ]) <maxpooling>

Both activation function and fully connected layer were already described in
@sec_nn. The fully connected layer is usually only used for the final prediction.
@cnn::pl are widely used because their shared-weight architecture significantly
reduces the number of trainable parameters, lowering computational costs and
enhancing generalization. Additionally, their convolutional layers enable them
to automatically learn hierarchical feature representations from the input
data. @indolia_conceptual_2018

==== Long short-term memory
@lstm networks are part of the @rnn family. Unlike standard neural networks, @rnn::pl
allow connections across time via feedback loops. This architecture gives the
network a form of "memory". @lstm::pl solve key problems faced by standard @rnn::pl, such as the
vanishing and exploding gradient issues during backpropagation through time.
They achieve this by enforcing a more stable error flow through their cell
state, which enables the model to retain relevant information over longer
sequences. @hochreiter_long_1997

A @lstm unit consists of three sections called gates, the forget gate,
the input gate and the output gate. It also carries two memories, the short-term memory
and the long term memory. The forget gate determines what percentage of the
long-term memory will be remembered. The input gate updates the long-term memory and
the output gate updates the short-term memory. The short-term memory of the last unit
is the output of the @lstm. Each step takes the short-term memory, the long-term memory
and the current input into consideration. Two activation functions are used. The sigmoid and the
tanh function, see @lstm @noauthor_deep_nodate. @noauthor_deep_nodate

#figure(image("figures/lstm.jpeg", width: 60%), caption: [Visualization of a Long Short Term Memory unit.]) <lstm>

@bilstm::pl are an extension of @lstm::pl that process sequential data in forward and backward
directions. This allows the model to capture context from past and future states
simultaneously. This ability is important for tasks like natural language processing,
where understanding both past and future context is crucial. @noauthor_bidirectional_nodate

==== Transformers

The introduction of the Transformer architecture has had a big impact that is
not limited to scientific research. Large Language Models like Open AIs ChatGPT
used this technology to revolutionize chat bots @ray_chatgpt_2023. It could be
seen as an advanced version of @rnn::pl. Compared to these, it does enable
parallel processing and has improved long-range dependence
@vaswani_attention_2017.

The original Transformer consisted of an encoder and a decoder and was created
for the purpose of machine translation @vaswani_attention_2017. Today there are
also Transformers that only use one of theses two. ChatGPT for example only uses
the decoder as it only needs to generate text, while BERT uses only the encoder
as it only needs to process the input sequence @vaswani_attention_2017. Here we
will cover the original transcoder architecture used for machine translation.

As for most @ai models, the input needs to be converted into a numerical
representation. Every token, which corresponds to a word or subword, is mapped to
a dense vector using a learned embedding matrix. To help the model understand the
position of each token in the sequence, positional encoding is added to the
embeddings. This encoding uses sine and cosine functions at different
frequencies to generate a unique pattern for each position. The result is added
to the embedded input. @vaswani_attention_2017

Next comes the self-attention mechanism. For each token, three vectors are
computed: a query, a key, and a value. These are obtained by multiplying the
input (embedding + positional encoding) with learned weight matrices. Then, the
attention score for each token is calculated by taking the dot product between
the query of the current token and the keys of all tokens. These scores are
passed through a SoftMax function to normalize them into probabilities,
determining how much of the value of each token should contribute to the
current one. This process is done multiple times in parallel with different
sets of weights, a technique known as Multi-Head Attention. The
result is then added back to the original input. @vaswani_attention_2017

After attention, the output passes through a feed-forward neural network, which
consists of two linear layers with a non-linearity in between. This step is
applied to each position separately and identically. Again, a residual
connection is used, and layer normalization is applied before or after each
sub-block, depending on the implementation, to stabilize training and improve
performance. @vaswani_attention_2017

In the decoder, Masked Multi-Head Attention is used instead of regular
self-attention. This ensures that each token can only attend to earlier
positions in the sequence. Another important component is encoder-decoder
attention, which allows the decoder to focus on relevant parts of the encoded
input. It works similarly to self-attention: a query is computed for the
current token being generated, and dot products are taken with the keys from
the encoder’s output. After applying SoftMax, the decoder learns which encoded
tokens are most important for generating the next word. Like other attention
mechanisms, this can also be stacked. @vaswani_attention_2017

Finally, the output from the decoder is passed through a linear projection
layer followed by a SoftMax function, producing a probability distribution over
the entire vocabulary. The most likely next token is selected based on this
distribution. Layer normalization and residual connections are applied
throughout the model to ensure stable learning and better generalization, see
@transformer @nyandwi_transformer_2023 for a detailed overview. @vaswani_attention_2017

#figure(image("figures/transformer.png", width: 80%), caption: [Visual overview of the
architecture of an encoder-decoder Transformer. ]) <transformer>

=== XGBoost

XGBoost (eXtreme Gradient Boosting) is a powerful and widely used @ml algorithm
based on the concept of gradient boosted decision trees. Boosting models build an
ensemble by sequentially adding many weak learners where each new learner
is trained to correct the errors made by the previous learners.
It has several features that set it apart from other
boosting implementations: It incorporates regularization to prevent overfitting
and improve generalization. It can automatically handle missing values and
is able to work in parallel. It employs tree pruning via a post-pruning process
that starts with deep trees and removes branches that do not contribute to
reducing the loss, thereby preventing overfitting. A schematic of the
XGBoost algorithm is shown in @xgboost @yao_short-term_2022. @chen_xgboost_2016

#figure(image("figures/xgboost.png", width: 50%), caption: [Schematic of the XGBoost algorithm. ]) <xgboost>

=== Categorical Embeddings

As already mentioned, most @ai models need numerical data as input. As inputs
do not always fulfill this several embedding techniques have been developed. In
this section we will focus on static word embeddings that were used in this
work. Static here means, that each word in the vocabulary only has one vector
representation.

Words are, in a way, categorical data. Representing them as numerical values
could be accomplished by using something like a one-hot encoding. In a one-hot
encoding a vector of the size of the number of categories is created for each
element. The position that corresponds to the category of this element is
filled with a one, while all other positions contain a zero. However, this
approach does have two drawbacks. First, these vectors can become very large
and second, all words are treated the same. This means there is no way
similarities between words can be expressed. @noauthor_word_nodate

A better representation for words is to use dense vectors of real numbers. Such
a vector can be significantly shorter than a one-hot encoded vector and is
filled with real numbers that each represent a property of the element. In
a biological context, if we assume that the elements are for example amino acids,
those properties could be thought of as chemical properties. A representation
of valine (@valine) and threonine (@threonine) could look like this: @noauthor_word_nodate

$ q_("valine") = [attach(limits(2.4), t: "Acidity"), attach(limits(1.2), t: "Size"), ... ] $ <valine>

$ q_("threonine ") = [attach(limits(6.5), t: "Acidity"), attach(limits(1.3), t: "Size"), ... ] $ <threonine>

This way similarities between words can be expressed and calculated, see @similarities. $Phi$
is the angle between the two words. That way similar words will result in a value close to
one while dissimilar words will result in a value close to zero. @noauthor_word_nodate

$ "Similarity"("valine", "threonine") = frac(q_"valine" dot q_"threonine", ||q_"valine"|| ||q_"threonine"||) = cos(Phi) $ <similarities>

In an actual word embedding these vectors do not contain real world properties like in
this example. Instead a model learns the values that result in the smallest loss.
@noauthor_word_nodate

=== Block Decomposition of Protein Sequences

The block decomposition algorithm by Martin Girard, a researcher at the Max Planck
Institute for Polymer Research, was used. This algorithm stems from an
unpublished article where a surrogate model for low complexity protein
sequences was created. The model itself is based on combinatorics on words,
particular sturmian words and their generalizations. It was able to show that
low complexity protein sequences have similar properties to homopolymers with
equivalent parameters. For example the radius of gyration. Changes to the
radius of gyration are strongly correlated to changes in the @llps propensity
of a protein. The algorithm was kindly provided to test it as feature in this
work. @noauthor_files_2024

The block decomposition algorithm itself uses word balance as a measure of
homogeneity. It finds the longest segments of the sequence, that have a word
balance below a certain threshold. As the algorithm originates from the field
of polymer physics these segments are called blocks. A word $w$ is $q$-balanced
if, for any two equal-length substrings $u$ and $v$ within $w$, the count of
each character differs by no more than $q$. Once the largest homogeneous block
is identified, the algorithm recursively searches for the largest blocks to the
left and right of it. Segments shorter than a predefined length threshold are
discarded. Before applying the block decomposition algorithm, a mapping is
applied to be less sensitive to mutations. @noauthor_files_2024

#pagebreak()
