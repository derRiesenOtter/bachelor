#import "@preview/glossy:0.8.0": *
#import "state.typ": bib_state
#context bib_state.get()
#show: init-glossary.with(yaml("glossary.yaml"))

= Introduction

== Phase Separation

The organization of the eukaryotic cell has classically been associated with
organelles that are surrounded by lipid membranes. However, over the past few
years phase transitions gained interest, as they have proven their role in
organizing cells. The term phase transition describes the process in which a
substance changes it physical state. @alberti_phase_2017

An especially important form of phase transition is @llps, where a homogeneous
solution separates into two phases. One phase is enriched in molecules, the
other is depleted. @alberti_phase_2017
- interface forms boundary that allows selective passage of some molecules but
  not others
- they can adopt different physical states, for example harden into gel- or
  glass-like states or turn into solid crystals
- consist of macromolecules such as RNAs and proteins
- some examples of this are P bodies, nucleolus
- organelles can be very large and complex and multilayered

- the availability of many different binding sites on a molecule is an important
  parameter to predict if phase separation will happen multi domain proteins
- a mixture of weak and strong interactions
- many phase separating proteins have poor solubility in water

- Intrinsically Disordered Proteins have been implicated in promoting phase separation
- do not have a fixed conformation
- often characterized by low sequence complexity (IDPs use only a small subset of the
  20 available amino acids)
- exact sequence of IDPs is often not important
- what matters for phase separation are simple charge patterns and the overall
  sequence composition.
- different types of IDPs
  - prion-like mostly composed of polar amino acids such as serine, tyrosine, glutamine, asparagine, and glycine
  - second class positively and/or negatively charged residues arranged in characteristic charge patterns. Such charged IDPs undergo electrostatic interactions that are highly sensitive to the pH and ionic strength of the solution.
  - the interaction of arginine/glycine-rich RNA-binding proteins with RNA. In this case, polyvalent interactions between the positively charged arginine residues and the negatively charged RNA drive the phase separation process.

- One function may be to concentrate biomolecules in a confined space and thus promote biochemical reactions
- Phase separation has also been shown to organize and facilitate signaling that phase separation can repress biochemical reactions One example is provided by ribonucleoprotein (RNP) granules. RNP granules often store mRNAs, which are then transported in a silenced state to diverse locations. By doing so, RNP granules promote the distribution of information in cells. In neurons, this allows the local synthesis of proteins in synapses and in dendrites upon demand.
- Phase separation is extremely sensitive to changes in physico-chemical conditions, suggesting that phase separation could play an important role in stress adaptation
@alberti_phase_2017

== Phase Separation Predictors

To know if a protein will take part in @llps can help to understand its use or
help planning experiments with it. As proving this property experimentally is
difficult and resource consuming, a tool to predict @llps helps to do so in an
efficient manner. In recent years many tools were developed with this task in
mind. In the following section we will highlight the functionality of some
representative @llps predictors:

- in-silico screening of proteomes
- often machine learning models used (the newer ones)
@hou_machine_2024

PSAP
based solely on amino acid content of the proteins. Uses Random Forest.
Important features were Fraction of AS C and L, percentage of IDRs and Low
Complexity scores.
@mierlo_predicting_2021

PSPire
As current models were biased towards proteins that contained @idrs this model
also included structural information to better estimate proteins lacking @idrs.
Previous models only relied on the amino acid sequence. This model outperformed
the other models significantly for proteins without @idrs and yielded comparable
results for proteins containing no @idrs. Uses XGBoost model.
@hou_machine_2024

== Block Decomposition of Protein Sequences

The functionality of proteins stems from small groups of amino acids that are
called motifs or active sites. These motifs are often integrated into larger protein
domains which are structurally independent units that perform distinct
functions. @embl-ebi_what_nodate

Finding conserved motifs or domains in proteins has become quiet easy. While
there are also motifs within @idrs, these motifs alone are not able to explain
the full functionality of these regions.

- block decomposition trying to find blocks with a certain functionality
- mapping of the sequence to a smaller alphabet that represents this
  functionality

The original Block Decomposition algorithm was created in an

It uses word balance as the parameter for homogeneity. Word balance for one word
is defined as the maximum difference in occurrences of a single letter between
any two factors of the word with the same length.

== Block Decomposition for Phase Separation Predictor

The goal of this work is to investigate the capabilities multidimensional block
decompositions based on word balance as predictor for phase separation.

#pagebreak()
