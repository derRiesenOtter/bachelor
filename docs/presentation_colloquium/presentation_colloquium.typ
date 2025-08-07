#import "@preview/touying:0.6.1": *
#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import "@preview/cetz:0.4.0"
#import themes.simple: *

#show: simple-theme.with(aspect-ratio: "16-9")

= Prediction of Liquid-Liquid Phase Separation of Proteins using Neural Networks

#place(right + bottom)[
Datum: 07.08.2025
]

#place(left + bottom)[
Bachelorarbeit Robin Ender
]

#place(right + top)[#image("./figures/th_bingen_logo.jpg", width: 25%)]

#place(left + top)[#image("./figures/cbdm_logo.png", width: 60%)]

== Liquid-Liquid Phase Separation (LLPS)
#show: magic.bibliography-as-footnote.with(bibliography("./Kolloquium.bib", title: none))
// Immer noch recht neues Gebiet, dass 
// den Vorgang von spontaner Phasenseparation einer fluessigen Phase 
// in zwei unterschiedliche konzentrierte beschreibt
// In biologischen Zellen geht man davon aus, dass LLPS auch der hauptmechanismus 
// hinter der Entstehung von membranlosen Organellen ist
- Bildung von Membranlosen Organellen
  // Diese haben im Vergleich zu den von membran umgebenen Organellen 
  // weniger Limitation im Austausch von Stoffen mit der Umgebung
#place(right,figure(image("./figures/membraneless_organelles.jpg", width: 60%)))
  // Die Grafik zeigt in A das Prinzip von LLPS und mit dem Protein FUS, dass zunaechst 
  // durch das Maltose Binding Protein getagt in Loesung bleibt und nach dem Abspalten 
  // LLPS durchfuehrt und Troepfchen bildet (aehnlich wie oel in wasser)
  // In B sind einige bekannte membranlose Organelle gezeigt:
  // Stress Granules: mRNA Lager und Regulation von Translation 
  // P body: mRNA Abbau und Silencing
  // PML bodies: Apoptose Signal, anti-virale Verteidigung and 
  // Regulation von Transkription
- Pathologisches Wirken: \ Demenz, Krebs, \ Infektionen 
  // Mutationen oder veraenderte Post Translationale Modifikationen koennen zu
  // Protein Aggregaten fuehren, die nicht reversibel sind oder negative
  // Auswirkungen haben, diese werden als Ursachen fuer Neurodegenerative
  // Erkrankungen sowie auch Krebs gesehen 
  // Auch Viren koennen LLPS nutzen, um sich effizient in menschlichen Zellen zu 
  // vervielfaeltigen. Sars-CoV-2 etwa nutzt wahrscheinlich LLPS in menschlichen Zellen zur 
  // Transkription, Replikation und Verpackung.
#text(white)[ @gomes_molecular_2019 @wang_liquidliquid_2021]

== LLPS Predictor

  // ueber die letzten Jahre wurden einige LLPS Predictors entwickelt
- Erste Generation: Hidden Markov Modelle, \ 
  Formel basiert 
  // die erste Generation basierte auf Hidden Markov Modellen und Modellen 
  // die ueber Formeln einen Score berechnet haben
- Zweite Generation: Machine Learning \ 
  (die skalare Werte) 
  // Die zweite Generation fing an machine learning algorithmen zu nutzen 
  // Diese nutzen Feature Engineering um aus der Sequenz Werte zu erstellen wie 
  // etwa der Anteil an Aminosauren, der Anteil an IDRs und viele mehr
- Einteilung der LLPS Proteine nach \ 
  Selbst-Assemblierend | Partner-Abhängig \ und Ungeordnet | Geordnet
  // Neuere LLPS Predictors zeigten auch, dass eine Unterscheidung der LLPS 
  // Proteine nach ihrer Rolle von Vorteil sein kann 
  // Ein weiteres Tool zeigte zudem auch, dass innerhalb eines Proteins die Aminosaueren 
  // abhaengig von ihrer geordnetheit und ihrer Kontaktmoeglichkeit mit 
  // der oberflaeche unterschiedlich in die Modelle einlaufen sollten
#place(right + horizon, image("./figures/homefig.png", width: 35%))
#text(white)[@chen_screening_2022]

== Hypothesen

- Neuronale Netze könnten besser geeignet für LLPS Prediction sein, als heutige 
  tools, weil sie:
  - Reihenfolge und Anordnung von Aminosäuren berücksichtigen, 
    dadurch komplexere Zusammenhänge verstehen können
  - Feature Engineering reduzieren 
- Input: Block Decomposition | Sequenz
- Ein neuer Datensatz des PPMC-lab, der speziell für die Entwicklung von LLPS 
  Predictors entwickelt wurde, sollte auch getestet werden
  // Gruppe, die neues dataset aus experimentell bestimmten positiv und negativ datenset 

== Datensatz und Evaluation

Hauptdatensatz: PSPire 
- Etwa 10.000 negativ gelabelte Proteine und etwa 500 positiv gelabelte Proteine

Evaluation durch: 
- Vergleich der ROCAUC und PRAUC Werte zu denen aus anderen Studien
  - Vor allem die PRAUC ist eine wichtige Metrik für unausgeglichene Datensätze

== Modellfindung I

Modellfindung über drei Phasen: 

+ Vergleich Block Decomposition gegen Sequenz + \ Test ob Neuronale Netze lernfähig sind
+ Testen weiterer komplexerer Modelle 
+ Optimierung des besten Modells und Zugabe weiterer biologischer Information an die Modelle

== Modellfindung II

#let g = rgb(80, 150, 200, 100)
#let y = rgb(180, 230, 0, 100)
#let f = rgb(240, 100, 0, 100)
#[#set text(size: 18pt)
#figure(
  diagram(node-corner-radius: 2pt, 
  spacing: (2.5em, 1em), 
  node((0, 0), [Start], fill: g, name: <a>), 
  edge(<a>, <b>, "-"), 
  node((-1.5, 1), [Block Decomposition], fill: g, name: <b>), 
  edge(<b>, <d>, "-"), 
  node((-2.5, 2), [XGBoost], fill: y, name: <d>), 
  edge(<b>, <e>, "-"), 
  node((-1.5, 2), [1L CNN], fill: g, name: <e>), 
  edge(<b>, <f>, "-"), 
  node((-0.5, 2), [2L CNN], fill: g, name: <f>), 
  edge(<a>, <c>, "=>"), 
  node((1.5, 1), [Sequence], fill: g, name: <c>), 
  edge(<c>, <g>, "-"), 
  node((0.5, 2), [1L CNN], fill: g, name: <g>), 
  edge(<c>, <h>, "-"), 
  node((1.5, 3), [3L CNN], fill: y, name: <h>), 
  edge(<c>, <i>, "-"), 
  node((2.5, 2), [BiLSTM], fill: y, name: <i>), 
  edge(<c>, <j>, "=>"), 
  node((0.5, 3), [2L CNN], fill: g, name: <j>), 
  edge(<c>, <k>, "-"), 
  node((2.5, 3), [Transformer], fill: y, name: <k>), 
  node((0.5, 5), [Batch Norm. \*], fill: f, name: <m>), 
  node((-0.5, 5), [RSA Weights], fill: f, name: <o>), 
  node((-1.5, 5), [RSA], fill: f, name: <p>), 
  node((0, 4), [Split Ungeordnet | Geordnet + \ higher Dropout], fill: f, name: <q>), 
  node((1.5, 5), [PTM \*], fill: f, name: <r>), 
  node((0, 6), [Final Model], fill: f, name: <s>), 
  edge(<j>, <q>, "=>"), 
  edge(<q>, <o>, "=>"), 
  edge(<q>, <p>), 
  edge(<q>, <o>, "=>"), 
  edge(<q>, <m>, "=>"),
  edge(<q>, <r>, "=>"), 
  edge(<r>, <s>, "=>"), 
  edge(<m>, <s>, "=>"), 
  edge(<o>, <s>, "=>"))
)]


// == Ergebnisse
//
// Phase 1: 
// - Block Decomposition bringt keinen Mehrwert in die Modelle
// - Zwei Layer CNNs erzielen eine leicht bessere Performance als 
//   die ein layer CNNs
//
// Phase 2: 
//   - Die komplexeren Modelle bringen alle keinen Mehrwert gegenueber 
//     dem zwei Layer CNN
//
// == Ergebnisse
//
// Phase 3: 
// - Erhoehung Dropout und Aufteilung in IDP und non-IDP verbesserte die Modelle 
// - Einbettung der Relative Solvent Accessibility (RSA) als Gewicht war ein Gewinn
// - Batch Normalization und integration der Post Translationalen Modifikationen (PTM)
//   brachten nur in dem non-IDP Modell Verbesserungen

== Finales Modell
// wo kommt rsa und ptm und so her
#[#set text(size: 18pt)
#cetz.canvas({
  import cetz.draw: *
  let heading = 4.5
  let dim = -4.5
  let dist = 3
  let start = 0

  content((0, heading), [Input])
  content((0, 0), text(size: 7pt)[MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTD...], angle: 270deg)
  content((0, dim), [2700 x 1])
  content((start + 0.45 * dist, 0), [$==>$])

  let start = start + dist
  content((start, heading), [Embedding])
  let bottom = -3.5
  let top = 3.5
  let width = 0.3

  for z in (0, 0.1, 0.2) {
    rect((start - z, bottom - z), (start - z + width, top - z), fill: white)
    let x = bottom
    while x < top {
      line((start - z, x - z), (start - z + width, x - z))
      x = x + 0.1
    }
  }
  content((start, dim), [2700 x 10])
  content((start + 0.45 * dist, 0), [$==>$])

  let start = start + dist
  content((start, heading), [RSA])
  let bottom = -3.5
  let top = 3.5
  let width = 0.3

  for z in (0, 0.1, 0.2) {
    rect((start - z - .3, bottom - z), (start - z - .3 + width, top - z), fill: white)
    let x = bottom
    while x < top {
      line((start - z - .3, x - z), (start - z - .3 + width, x - z))
      x = x + 0.1
    }
  }

  content((start + .15, 0), [$dot$])

  rect((start + .3, bottom), (start + .3 + width, top), fill: white)
  let x = bottom
  while x < top {
    line((start + .3, x), (start + .3 + width, x))
    x = x + 0.1
  }

  content((start, dim), [2700 x 10])

  content((start + 0.5 * dist, 0), [$==>$])

  let start = start + dist
  content((start, heading), [PTM])
  let bottom = -3.5
  let top = 3.5
  let width = 0.3

  for z in (0, 0.1, 0.2) {
    rect((start - z - .35, bottom - z), (start - z - .35 + width, top - z), fill: white)
    let x = bottom
    while x < top {
      line((start - z - .35, x - z), (start - z - .35 + width, x - z))
      x = x + 0.1
    }
  }

  content((start + .15, 0), [$+$])

  for z in (0, 0.1) {
    rect((start + .45 - z, bottom - z), (start + .45 - z + width, top - z), fill: white)
    let x = bottom
    while x < top {
      line((start + .45 - z, x - z), (start + .45 - z + width, x - z))
      x = x + 0.1
    }
  }
  content((start + 0.6 * dist, 0), [$==>$])

  content((start, dim), [2700 x 18])

  let start = start + dist
  content((start, heading), [CL 1])
  let bottom = -3.3
  let top = 3.3
  let width = 0.3

  for z in (0, 0.1, 0.2, 0.3, 0.4) {
    rect((start - z, bottom - z), (start - z + width, top - z), fill: white)
    let x = bottom
    while x < top {
      line((start - z, x - z), (start - z + width, x - z))
      x = x + 0.1
    }
  }
  content((start, dim), [2691 x 70])

  content((start + 0.5 * dist, 0), [$==>$])
  content((start + 0.5 * dist, .8), [BN, \
  ReLU])

  let start = start + dist
  content((start, heading), [MPL])
  let bottom = -1.5
  let top = 1.5
  let width = 0.3

  for z in (0, 0.1, 0.2, 0.3, 0.4) {
    rect((start - z, bottom - z), (start - z + width, top - z), fill: white)
    let x = bottom
    while x < top {
      line((start - z, x - z), (start - z + width, x - z))
      x = x + 0.1
    }
  }
  content((start, dim), [1347 x 70])

  content((start + 0.45 * dist, 0), [$==>$])

  let start = start + dist
  content((start, heading), [CL 2])
  let bottom = -1.5
  let top = 1.5
  let width = 0.3

  for z in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6) {
    rect((start - z, bottom - z), (start - z + width, top - z), fill: white)
    let x = bottom
    while x < top {
      line((start - z, x - z), (start - z + width, x - z))
      x = x + 0.1
    }
  }
  content((start, dim), [1338 x 140])

  content((start + 0.45 * dist, 0), [$==>$])
  content((start + 0.5 * dist, 0.8), [BN, \
  ReLU])

  let start = start + dist
  content((start, heading), [AMPL])
  let bottom = 0
  let top = 0.1
  let width = 0.3

  for z in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6) {
    rect((start - z, bottom - z), (start - z + width, top - z), fill: white)
    let x = bottom
    while x < top {
      line((start - z, x - z), (start - z + width, x - z))
      x = x + 0.1
    }
  }
  content((start, dim), [1 x 140])

  content((start + 0.55 * dist, 0), [$==>$])
  content((start + 0.5 * dist, 0.55), [Dropout])

  let start = start + dist
  content((start, heading), [Fully \ Connected \ Layer (p)])
  let bottom = -0.1
  let top = 0.1
  let width = 0.3

  rect((start, bottom), (start + width, top), fill: white)
  let x = bottom
  while x < top {
    line((start, x), (start + width, x))
    x = x + 0.1
  }
  content((start, dim), [2])

})]

== Ergebnisse PSPire Datensatz

Vergleich mit PSPire @hou_machine_2024 und PdPS @chen_screening_2022: 
#align(center,image("./figures/Screenshot 2025-08-05 at 21.28.18.png", width: 50%))

Vergleich mit PSPire und PdPS auf MLO Daten: 
- ähnlich wie PSPire Datensatz

== Ergebnisse catGranule / PPMC-lab Datensatz 

Vergleich catGranule 2.0 @monti_catgranule_2025: 
- Leicht bessere Performance als die dort beschriebenen Tools
  - catGranule 2.0: 0.76 ROCAUC
  - non-IDP Modell: 0.80 ROCAUC

PPMC-lab Datensatz: 
- Evaluation auf MLO Daten zeigte deutlich schlechtere Performance

== Ergebnisse Saliency

#place(left + top, dy: 10%)[Saliency map und LLPS \ 
Profil von Protein P04264]

#align(right)[
  #image("./figures/captum_idr_P04264_1.0_0.9834264516830444_ID-PSP.png", width: 55%)]
#place(right, dx: -9%,image("./figures/sp|P04264|K2C1_HUMAN_Profile.png", width: 50%))

== Fazit
#pause
- Block Decomposition als Input nicht besser als die Sequenz 
#pause
- Trotz relativ kleinen Datensätzen haben die CNNs bereits 
  vergleichbare Leistung zu anderen State of the Art Tools gezeigt -> CNNs haben ggf. das Potenzial bei besserer Datenlage jetzige Tools 
  abzulösen (mehr und besser annotiert - PTMs)
#pause
- Saliency ermöglicht Einblicke in Entscheidungsfindung, interpretierbar
  wie LLPS Profil anderer Tools

== Outlook
#pause
- Saliency Maps weiter analysieren 
#pause
- Weitere Optimierung des Modells durch systematische Anpassung der Parameter und 
  five-fold Validation
#pause
- Experimente mit grösserem Datensatz wiederholen
#pause
- Ein fertiges Tool (Web Anwendung / CLI-Tool) entwickeln

= Vielen Dank! 
Fragen?
