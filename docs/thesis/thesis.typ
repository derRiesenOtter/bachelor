#import "@preview/glossy:0.8.0": *

#set page(paper: "a4")
#set par(justify: true)
#set text(size: 11pt)
#set table(stroke: (_, y) => if y == 0{ (bottom: 0.5pt, top: 1pt) })

#show outline.entry.where(level: 1): header => {
  set text(weight: "bold")
  set block(above: 1.3em)
  header
  the
}
#show figure.where(kind: table): set figure.caption(position: top)
#show: init-glossary.with(yaml("glossary.yaml"))

#import "state.typ": bib_state
#bib_state.update(none)

#include "titlepage.typ"

#set page(numbering: "I")

#outline(title: "Table of Contents")
#show outline: set heading(outlined: true)
#pagebreak()
#outline(title: "List of Figures", target: figure.where(kind: image))
#pagebreak()
#outline(title: "List of Tables", target: figure.where(kind: table))
#pagebreak()
#glossary()
#set heading(numbering: "1.")
#pagebreak() <page-anchor>

#set page(numbering: "1")
#counter(page).update(1)

#include "introduction.typ"
#include "material.typ"
#include "methods.typ"
#include "results.typ"
#include "discussion.typ"

#set page(numbering: "I")
#context counter(page).update(counter(page).at(<page-anchor>).first())

#bibliography("bachelor.bib")
