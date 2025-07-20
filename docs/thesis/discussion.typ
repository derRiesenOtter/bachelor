#import "@preview/glossy:0.8.0": *
#import "state.typ": bib_state
#context bib_state.get()
#show: init-glossary.with(yaml("glossary.yaml"))

= Discussion

== Block Decomposition vs. Raw Sequence

The comparison between simple models using either block decomposition or the
raw sequence input revealed that using the block decomposition does not benefit
the models. In the case of @nn, which are capable of automatically learning and
extracting complex features from raw data, applying block decomposition
effectively limits the available information. This reduction in input detail
likely explains the slight performance decrease observed with these models.

Similarly, applying block decomposition to the traditional @ml model XGBoost
did not yield strong results. This suggests that the simplified representation
introduced by block decomposition may omit critical patterns or contextual
information necessary for accurate predictions, particularly in complex tasks
such as modeling @llps.

== Choosing the best Model

After the simple two layer @cnn performed the best in the first tests, some
more complex alternatives, including a three layer @cnn, a @bilstm model and a Transformer, were
tested. Although some of these models achieved results comparable to the two
layer model, none was able to surpass its performance. A previous study that
focused on DNA-protein binding also came to the conclusion that the performance
of @cnn::pl decreases with network complexity when there is insufficient
training data. In their case the more complex models only started to outperform
the simple models when they used a training set of 40,000 samples
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
tendency to overfit, the dropout rate was increased. A higher dropout rate
reduces the model’s reliance on individual neurons, encouraging it to learn
more generalizable features and thereby mitigating overfitting.

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
values as weights integrates this information directly into the model in.

The addition of @bn yielded mixed results. On the PSPire dataset, it improved
the performance of the non-@idr model, but led to performance declines when
applied to the @idr model or on the PPMC-lab dataset. A likely explanation for
this inconsistency lies in the use of sequence padding. Since protein sequences
vary in length, all sequences were padded to a uniform length of 2700. As seen
in the distribution o sequence length, most proteins in this study were smaller
than 1000 residues, which leads many sequences consisting mostly of padded values.
@bn calculates the mean and standard deviation per channel across the
batch, including the padded values. When batches include many short sequences,
these padding values disproportionately affect the normalization statistics,
skewing the model’s behavior. Batches with longer sequences are less affected.
This imbalance can result in unstable or degraded performance.

The inclusion of @ptm values did not yield significant improvements on the
PPMC-lab dataset. On the PSPire dataset, the performance of the @idr model
decreased slightly, while the non-@idr model experienced a minor improvement.
Although @ptm::pl are known to play a role in @llps @li_post-translational_2022
and their inclusion should theoretically enhance the models, the current
databases for @ptm::pl are incomplete. Many proteins likely contain unannotated
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
on the @mlo data yielded unexpected results. Compared to the the models trained
on the PSPire dataset, a significant drop in performance was observed. One
plausible reason for this drop could be the smaller size of the PPMC-lab
dataset, particularly its limited number of negative samples. With only a third
of total samples the models were probably not able to learn the features in a
general manner. For the @mlo evaluation, the negative test set from PSPire was
reused. To investigate whether this influenced performance, one evaluation was
conducted using the negative test set from the PPMC-lab dataset instead. This
did not change the observation. It would be interesting to see how the more
traditional @ml predictors would perform using the PPMC-lab dataset, as they
require fewer data.

Lastly, the evaluation using the catGranule 2.0 dataset did again show that
the model created in this work does yield a competitive performance. It
was able to slightly outperform all models that were evaluated in the
catGranule 2.0 paper. It is important to mention that due to lack of time,
the dataset was not split into @idp::pl and non-@idp::pl. This split
could lead to further improvements as it did on the PSPire dataset.

The visualization of the saliency provided some insights on which features of a
sequence were important during classification. Investigating the saliency scores
at a larger scale could unveil important patterns or sequences for @llps.
For now the information they
provide is limited, but further investigations of these could help identifying
patterns or sequences that contribute to @llps. The comparison of the protein
P42766 showed the difference of feature importance between the @idr model and
the non-@idr model. The visualizations of the @idr proteins was usually more
subtle and did not involve strong bands like for the non-@idr proteins.

== Comparing @nn to @ml models

Given the similar performance, there are some benefits as well as downsides to
using @nn compared to traditional @ml approaches. \
One benefit is the reduction of feature engineering. The final models contained
three different features at most. These being the sequence, the @rsa values and
the @ptm::pl. Of these three features the sequence and the @ptm::pl are
directly available for use on UniProt, even though the @ptm values underwent
some feature engineering by mapping them into similar groups. The inclusion of
the @rsa values did require obtaining the structure files from AlphaFold and
using DSSP to calculate them. In comparison, the PSPire model used 44 different
features. Most of these features were easy to calculate as 20 were only
fractions of single amino acids and 19 were fractions of groups of amino acids.
Others were more complex and required external tools for calculation like the
isoelectric points, the polarity scores or their computational approach to
identify charged stickers that included the use of @rsa values. \
The second benefit may be the ability to process the whole sequence and search
it for patterns. @ml models are limited to the tabular representation of a
protein sequence. While this is a legitime approach, especially if the datasets
are small, these features are limited and are not enough to model complex
interactions like @llps appropriately. @nn are more powerful when it comes to
understanding more complex interactions especially with sequence like data.
What holds them back for now is the limited availability of curated data.

Both types of models offer some degree of interpretability regarding the roles
of input features. Traditional @ml models are generally more transparent in
this regard. Feature importance can be directly quantified using techniques
such as permutation importance, SHAP values, or model coefficients, making it
easier to understand which features contribute most to the predictions.

In contrast, interpreting @nn models is more complex due to their higher
dimensionality and non-linear structure. However, interpretability is still
possible through techniques such as saliency maps, integrated gradients, or
attention mechanisms. These methods allow the visualization of feature
importance on a per-sample basis, highlighting specific input regions that
influence the model’s decision. This can support deeper biological analysis,
for example by identifying critical amino acid regions or structural patterns,
and may contribute to a better understanding of underlying mechanisms such as
those driving @llps.

There are, however, some notable downsides to using @nn models. Although not
explicitly measured in this study, training neural networks typically requires
significantly more computational resources and time compared to traditional @ml
models. This can be a limiting factor, especially when working with large
datasets or limited hardware.

Another challenge is the need to pad sequences to a fixed length, due to the
varying sizes of protein sequences. Padding can introduce noise and interfere
with model performance, particularly if a large portion of the input consists
of non-informative padded positions. This issue can distort intermediate
representations—especially in techniques like batch normalization—and may
reduce the model’s ability to generalize effectively.

== The remaining challenge

The main challenge in @llps prediction will continue to be the insufficient
curated data. While datasets are growing it will still take some time till
they reach appropriate sizes for training more complex models. Another
challenge is the absence of a database for proteins that do not partake in
@llps. Creating such a database itself would be difficult as @llps does only
occur if the right physicochemical conditions are met.

Furthermore, new @llps prediction tools should strive to be more comparable.
One goal of the people behind the PPMC-lab dataset is to help with this problem
by providing a dataset that can be used to benchmark models. In fact, part of their
work focuses on comparing the currently available tools using their dataset.

== Conclusion

This work has demonstrated that an @llps predictor based on a relatively simple
two-layer @cnn architecture can perform on par with, and in some cases surpass,
state-of-the-art models based on more traditional @ml methods. While the
initial idea of using block decomposition as input proved viable for @llps
prediction, it consistently underperformed compared to using the raw protein
sequence. As a result, the focus of this study shifted toward developing a
sequence-based @llps predictor.

Several enhancements contributed to the model’s success, including the
integration of @rsa values, @ptm features, and batch normalization (@bn), as
well as the division of the dataset into proteins with and without @idr::pl.
This stratification enabled the model to better capture differences in the
underlying sequence characteristics of these two protein groups.

The final model achieved results that were comparable or superior to those of
PSPire, particularly for proteins containing @idr::pl, on both the PSPire test
data and the independent @mlo datasets. Notably, this performance was achieved
with less reliance on handcrafted features compared to other models. It also
outperformed several additional @llps predictors not covered in the PSPire
publication, including CatGranule 2.0, which relies on a feature-rich input of
128 descriptors.

Despite these promising results, one of the ongoing challenges in @llps
prediction is the limited amount of high-quality, curated data. Although the
dataset has grown in recent years, it remains insufficient to capture the full
complexity of the biological processes involved. The fact that a relatively
lightweight @cnn model is already able to compete with more established @ml
approaches is encouraging, especially considering the potential of neural
networks when applied to larger datasets.

The experiments underscore that tailoring prediction models to specific protein
characteristics—particularly by separating @idr and non-@idr proteins—can lead
to meaningful improvements in performance. The consistent benefit of including
@rsa values as attention weights further highlights the importance of surface
accessibility in modeling interactions relevant to @llps. While @bn and @ptm
features showed mixed effects, this likely reflects issues such as incomplete
annotation databases and technical limitations like sequence padding.

In conclusion, the final models—one optimized for @idr proteins using @rsa
weighting, and another for non-@idr proteins that combines @rsa weighting with
@bn and @ptm features—strike a strong balance between model complexity and
generalization. These results lay a solid foundation for future improvements.
Further progress may come from more comprehensive data curation, advanced
handling of variable sequence lengths, and deeper integration of functional and
structural protein annotations.

#pagebreak()
