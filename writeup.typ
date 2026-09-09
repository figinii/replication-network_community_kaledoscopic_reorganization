#import "@preview/zebraw:0.6.3": *

#let conf(doc) = {
  set page(columns: 2)
  set page(
    paper: "a4",
    margin: 0.5in,
  )

  set text(
    font: "New Computer Modern",
    size: 11pt,
    lang: "en",
  )

  show: zebraw

  align(center)[
    #text(24pt, "Writeup")
  ]

  doc
}

The goal of the paper is to characterize the organization of communities formed using a modularity maximization algorithm, Louvain algorithm in particular, at the change of parameter $gamma$.

now let's start with a little digression over the modularity. 
- a null model comparison: $sum_(g in G) frac(1,2M) sum_(i,j in g) (A_(i j) - gamma frac(k_i k_j,2M)) = frac(1, 2M) sum_(g in G) L_g - gamma frac(K_g^2, 2M)$
- a intra-clustering density with a regularization over the volume: $$



