#import "@preview/glossy:0.8.0": *
#import "state.typ": bib_state
#context bib_state.get()
#show: init-glossary.with(yaml("glossary.yaml"))

= Discussion

== Block Decomposition vs. Raw Sequence

The comparison of simple models using the block decomposition and sequence as
input showed, that while the block decomposition yielded
comparable results on one dataset it performed significantly worse on the
other. As @nn are able to extract features by themselves applying the block
decomposition is basically reducing the information they get. This would
explain why these models performed slightly worse.

Using the block decomposition with the @ml model XGBoost did not provide great results
either.

== Choosing the best Model

Different models were tested in the second phase. The tested @nn all were more
complex and theoretically more capable than the two layer @cnn that was used as
the baseline. While some models showed comparable results, none outperformed
the two layer model. This could be due to two reasons. For one, more complex
models are more reliant on sufficient data, especially transformers. As the
tested dataset was only around ten thousand entries large and only contained
around 2000 positive samples the conditions were not in favor for the more
complex models. The other reason may be @cnn's efficiency in capturing local
sequence patterns that could be beneficial in finding sequence motifs involved
in @llps, even if they are not preserved well. The other models were only
tested with one set of parameters it is possible that they would be able to
perform better after optimizing them, but due to the little data that is
available and the fact that these models would generally require more data, they
were not tested more thoroughly.

== Optimizing the final Model

The two layer model was chosen due to its performance. Several tests were
conducted trying to improve it. As the models showed a tendency to overfit the
dropout value was increased. Increasing the dropout makes the model less dependent on
single neurons, which leads to the model learning more general features and therefore
counteracting overfitting. As the features that lead to @llps in proteins are
sometimes divided into ones that are important in proteins with @idr::pl and
without @idr::pl, the positive dataset was divided into those and independent
models were created. For the PPMC-lab dataset the performance was barely
affected. The prediction of the PSPire dataset however showed a performance
increase for the non-@idr model. Therefore, the split model was taken as a new
baseline. Starting from this model a new set of approaches were tested.

Both models that included the @rsa values improved the performance. However,
the model that used the @rsa values as weights for the embedded sequence
performed better than the model that concatenated them with the embedded
sequence. These improvements were seen on both datasets, although they were
stronger on the PSPire dataset. Seeing a better performance when including the
@rsa values makes sense as mainly the amino acid sequence that is on the
surface of a protein is responsible for interactions with other molecules.
Using the @rsa values as weights is an intuitive way integrate this information into the model.

The addition of @bn did improve the performance of the non-@idr model on the
PSPire data but showed performance decreases when added to the other model and
on the PPMC-lab dataset. A probable cause for this inconsistency is the use of
sequence padding. Since the protein sequences vary in length, they were padded
to a fixed length of 2700. As a result, many sequences contain more padded
positions than actual amino acids. @bn computes the mean and standard deviation
per channel across the batch, including the padded positions. When many short
sequences are present in a batch, the normalization statistics are
disproportionately influenced by the padding values, skewing the normalization
process. In contrast, batches dominated by longer sequences would not suffer
this effect. This behavior may lead to unstable or degraded model performance.

Adding the @ptm values to the models also did not show any significant changes
on the PPMC-lab dataset. For the PSPire dataset, the @idr models performance
decreased while the non-@idr models slightly increased. @ptm::pl are important
factors for @llps and the inclusion should theoretically benefit the models.
The problem here probably lies in the fact, that the databases for @ptm::pl do
only contain a small amount of the @ptm::pl that are actually present.
Therefore proteins that are missing their annotations, because there were not
experimentally found yet, interfere with the models learning and prediction
capabilities.

== Evaluation of the final Model

== Conclusion

This work has shown, that an @llps predictor based on a relatively simple @cnn
structure is able to compete with state-of-the-art competitors that are based
on @ml models. While the initial idea of using a block decomposition of a
sequence as input did work for @llps prediction, it was inferior to using the
bare sequence. Therefore, the focus of this work changed to developing an @llps
predictor based on the protein sequence. The addition of @bn, @rsa, @ptm and
the division into proteins containing an @idr and proteins containing no @idr
contributed to the performance of the model.

The model created in this work did manage to deliver slightly worse, but still
comparable results on proteins that do not contain an @idr::pl and comparable
or better results on proteins that do contain @idr::pl compared to the well
performing tool PSPire on both the PSPire testing data and the @mlo datasets.
It did that requiring less feature engineering.

It also outperformed a set of other modern @llps predictors that were not included
in the PSPire paper. Especially the CatGranule 2.0 model that TODO.

One major problem of @llps predictors remains the small amount of curated data.
Although it has increased over the last years there is still not enough data to
learn all the complex interactions that lead to @llps. The fact, that the model
created during this work is already able to compete with @ml models shows is
promising for the usage of @nn based models for this task, as they tend to really shine
when it comes to datasets that are larger than 10,000 entries.

#pagebreak()
