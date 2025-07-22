#import "@preview/glossy:0.8.0": *
#import "state.typ": bib_state
#context bib_state.get()
#show: init-glossary.with(yaml("glossary.yaml"))

= Discussion

== Block Decomposition vs. Raw Sequence

The comparison between simple models using either block decomposition or the
raw sequence input revealed that using the block decomposition does not benefit
the models. In the case of @nn::pl, which are capable of automatically learning and
extracting complex features from raw data, applying block decomposition
effectively limits the available information. This reduction in input detail
likely explains the slight performance decrease observed with these models.

Similarly, using the block decomposition with the @ml model XGBoost
did not yield strong results either. This suggests that the simplified representation
introduced by the block decomposition algorithm is not suitable for @llps prediction.

== Choosing the best Model

After the simple two layer @cnn performed the best in the first tests, some
more complex alternatives, including a three layer @cnn, a @bilstm model and a Transformer, were
tested. Although some of these models achieved results comparable to the two
layer model, none was able to surpass its performance. A previous study that
focused on DNA-protein binding also came to the conclusion that the performance
of @cnn::pl decreases with network complexity when there is insufficient
training data. In their case the more complex models only started to outperform
the simple models when they used a training set of around 40,000 samples
@zeng_convolutional_2016. This reasoning is probably applicable for @bilstm models
and Transformers too, as they are also more complex. The datasets used in this
study contained only around 10,000 entries at most. This limited size is likely
insufficient for effectively training deeper or more complex models.

It should also be noted that the alternative models were only evaluated using a
single set of hyperparameters. It is possible that with more extensive tuning,
some of these architectures could achieve better performance. However, given
the small dataset and the greater data requirements of these models, a more
exhaustive evaluation was not conducted in this study.

== Optimizing the final Model

The two-layer model was selected based on its strong performance. Several
experiments were conducted to further enhance it. As the models exhibited a
tendency to overfit, seen by the train loss, the dropout rate was increased. A
higher dropout rate reduces the model’s reliance on individual neurons,
encouraging it to learn more generalizable features and thereby mitigating
overfitting.

Since the features contributing to @llps in proteins often differ between
proteins with and without @idr::pl, the positive dataset was split accordingly
@hou_machine_2024. Separate models were then trained on each subset. On the
PPMC-lab dataset, this separation had minimal impact on performance. However,
for the PSPire dataset, the performance of the non-@idr model improved notably.
As a result, the split-model approach was adopted as the new baseline, and
subsequent experiments were built upon this version.

Both models incorporating @rsa values demonstrated improved performance.
However, the model that used @rsa values as weights applied to the embedded
sequence outperformed the one that simply concatenated the @rsa values with the
embeddings. These performance gains were observed on both datasets, though they
were more pronounced on the PSPire dataset. This improvement is intuitive, as
amino acids located on the protein’s surface, which is represented by the @rsa
values, are primarily responsible for molecular interactions. Applying @rsa
values as weights integrates this information directly into the model.

The addition of @bn yielded mixed results. On the PSPire dataset, it improved
the performance of the non-@idr model, but led to performance declines when
applied to the @idr models. A likely explanation for
this inconsistency lies in the use of sequence padding. Since protein sequences
vary in length, all sequences were padded to a uniform length of 2700. As seen
in the distribution of sequence length, most proteins in this study were smaller
than 1000 residues, which leads to many sequences consisting mostly of padded values.
@bn calculates the mean and standard deviation per channel across the
batch, including the padded values. When batches include many short sequences,
these padding values disproportionately affect the normalization statistics,
skewing the model’s behavior. Batches with longer sequences are less affected.
This imbalance can result in unstable or degraded performance.

The inclusion of @ptm values did not yield significant improvements on the
PPMC-lab dataset. On the PSPire dataset, the performance of the @idr model
decreased slightly, while the non-@idr model experienced a minor improvement.
Although @ptm::pl are known to play a role in @llps @li_post-translational_2022
and their inclusion should theoretically enhance the models performance, the
current databases for @ptm::pl are incomplete. Many proteins likely contain
@ptm::pl that have not yet been experimentally identified. The absence of this
data introduces noise, which can interfere with model training and prediction
accuracy.

In the final stage, the different enhancements were tested in various
combinations. For the @idr model, no combination outperformed the two-layer
model with @rsa values used as weights. Consequently, this
configuration was selected as the final model for @idr proteins. In contrast,
for the non-@idr model, the combination of @rsa values as weights, @bn, and the
inclusion of @ptm values produced the best overall performance. This model was
therefore chosen as the final version for non-@idr proteins.

== Evaluation of the final Model

To further assess and compare the performance of the final models, they were
trained and tested across multiple datasets. The first comparison was conducted
using the PSPire dataset. In this setting, the @idr model was unable to match
the performance of the PSPire model, though it achieved results that were
comparable to those of the PdPS model. Notably, the non-@idr model outperformed
both the PSPire and PdPS models, although its advantage over PSPire was
marginal. A similar pattern emerged when the models trained on the PSPire
dataset were evaluated using the @mlo datasets. Once again, the @idr model
underperformed relative to both the PSPire and PdPS models. Meanwhile, the
non-@idr model surpassed the PdPS model and performed at a level comparable to
the PSPire model.

One possible explanation for the weaker performance of the @idr model lies in
the intrinsic characteristics of @idr::pl. Unlike @cnn::pl, which are capable
of identifying recurring patterns, @idr::pl are not constrained by any fixed or
consistent structural motifs. Two @idr sequences that lead to @llps can appear
entirely different from each other. As a result, relying solely on structural
similarity or pattern recognition may not be sufficient for accurate modeling
or requires more data. In contrast, using only the fractional composition of
amino acids, combined with additional scalar features, appears to be a more
effective strategy at present @hou_machine_2024. The non-@idr model, already performs
competitively with existing models, though there is still considerable room for
improvement in its overall performance.

Using the models trained on the PPMC-lab dataset to evaluate their performance
on the @mlo data yielded unexpected results. Compared to the models trained
on the PSPire dataset, a significant drop in performance was observed. One
plausible reason for this drop could be the smaller size of the PPMC-lab
dataset, particularly its limited number of negative samples. With only a third
of total samples the models were probably not able to learn the features in a
general manner. For the @mlo evaluation, the negative test set from PSPire was
reused. To investigate whether this influenced performance, one evaluation was
conducted using the negative test set from the PPMC-lab dataset instead. This
did not change the observation. It would be interesting to see how the more
traditional @ml predictors would perform using the PPMC-lab dataset as they
require fewer data and as the PPMC-lab dataset does feature higher quality
selection of non @llps proteins.

Lastly, the evaluation using the catGranule 2.0 dataset did again show that
the model created in this work does yield a competitive performance. It
was able to slightly outperform all models that were evaluated in the
catGranule 2.0 paper. It is important to mention that due to lack of time,
the dataset was not split into @idp::pl and non-@idp::pl. This split
could lead to further improvements as it did on the PSPire dataset.

The visualization of the saliency scores provided a preliminary look at which
regions of a sequence might influence classification. Comparing these plots
with the @llps propensity profiles from catGranule 2.0 showed that the regions
with high saliency are similar to the regions catGranule 2.0 identified to have
high @llps propensity. Scaling up the analysis of the saliency scores may holds
potential for uncovering meaningful patterns or sequence features associated
with @llps. Another interesting insight was to compare the saliency between the
@idr model and the non-@idr model. It revealed, that both models relied on very
different areas of the proteins to form their prediction. This could be seen as
supportive to create different models for the different @llps proteins.
Comparing the results from the saliency scores with experimental data could
further confirm if the learned features actually play important roles in @llps.
Having the ability to visualize the important areas of the sequences also is an
advantage over the some of the other @llps predictors that do not have a similar
representation like PSPire.

Due to time constraints, several promising ideas could not be explored in this
study. One such is the optimization of hyperparameters, for example
through five-fold cross-validation. This technique would likely improve the
robustness and generalization ability of the final models and might lead to
measurable performance gains. Additionally, alternative strategies for dividing
the training data should be investigated, for instance, by distinguishing
between self-assembling proteins and those that depend on interaction partners,
like the PhasePred study did. Comparing these two grouping strategies
could help further studies on @llps prediction to choose the most impactful. \
Another worthwhile extension would be the construction of a new dataset that
combines the larger positive data from the catGranule 2.0 study with the
large negative set used in the PSPire dataset. Lastly, making the developed tool
accessible for external users, for example, as a web application or
command-line utility, would increase its practical value and
enable its use by other researchers.

#pagebreak()

== Conclusion

This work has shown that a relatively simple, sequence-based @cnn model can
perform competitively and sometimes even outperform traditional @ml models.
While both the @cnn models and the @ml models yielded similar performance,
they offer different advantages and limitations.

One of the clearest benefits of using @nn models lies in the significant
reduction of feature engineering. The final models in this study relied on only
three main inputs: the raw amino acid sequence, @ptm annotations, and @rsa
values. Of these, both the sequence and @ptm::pl information are readily
accessible via databases like UniProt, although the @ptm features required some
minor preprocessing. The @rsa values, while effective in improving performance,
do require structural predictions and additional processing using tools like
AlphaFold and DSSP. In contrast, PSPire used 44 features, many of which are
derived from simple amino acid compositions but also include complex
features requiring external computation. Despite this feature richness, the
@cnn model achieved comparable results.

A second major benefit of @nn models is their ability to process whole
sequences, identifying complex patterns that are inaccessible to models
restricted to flat, tabular inputs. This is particularly relevant in the
context of @llps, where interactions between sequence motifs, disorder, and
post-translational modifications play significant roles. However, there are
also certain drawbacks to these model types. Neural networks typically demand
more computational resources and longer training times. Processing sequences
of different lengths requires padding which can introduce noise.

When it comes to interpretability both @ml models and @nn models have their
benefits. Traditional @ml models are generally more transparent, with
well-established methods that allow for clear feature attribution. Neural
networks, on the other hand, are inherently more complex due to their depth and
non-linear structure. In this study, it was experimented with saliency-based
visualization techniques to inspect model behavior. The comparison of these
saliency maps showed similarities to @llps propensity profiles created by
catGranule 2.0. This approach may hold potential for further
analysis that could be used to find relevant sequence sections for @llps.

An important theme throughout this work is the limited availability of curated
data. The success of the @cnn model, despite being relatively lightweight, is
encouraging, but likely bounded by the current size and quality of existing
datasets. The absence of a well-defined negative dataset also remains a bottleneck.
Efforts, such as the PPMC-lab dataset, are therefore important contributions
and will help to mitigate this problem. Despite the small dataset itself, the
incompleteness of @ptm::pl may also limit current models.

Another major insight was the value of dividing the model to specific protein
characteristics. Dividing the data into proteins with and without @idr::pl
and developing specialized models for each group led to consistent improvements
in predictive performance. The saliency maps further confirmed that the two
models focused on different regions of the protein sequences. The addition of
@rsa values as attention weights also proved beneficial. While @bn and @ptm
features showed mixed impact, likely due to incomplete annotations and the
variability introduced by padding, the overall framework remains extensible and
can be adapted as better data and annotations become available.

In summary, this study contributes a practical @llps predictor showing that @nn
approaches can match or outperform traditional models while simplifying the
feature engineering process. The strong performance achieved with relatively
few features and a compact architecture suggests that there is significant
potential in neural network-based methods, especially as datasets grow.

#pagebreak()
