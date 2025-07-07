#import "@preview/glossy:0.8.0": *
#import "state.typ": bib_state
#context bib_state.get()
#show: init-glossary.with(yaml("glossary.yaml"))

= Discussion

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
