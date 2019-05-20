---
layout: poster
title: Markov Random Filed (MRF)
date: 2019-04-16 23:54:54
categories: ML
mathjax: true
---

###### Markov Random Field (MRF) <=> Markov Network <=> Conditional Random Field

### Pairwise Markov Networks

A pairwise Markov network is an undirected graph whose nodes are $X_1, X_2, \dots, X_n​$ and each edge $X_i-X_j​$ is associated with a factor $(potential)\phi_{ij}(X_i-X_j)​$.

<!-- more -->

### General Gibbs Distribution

![image-20190417015557619](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190417015557619.png)

Which is much smaller than probability distribution over n random variables ($O(d^n)$), which means **pairwise is not expressive enough** for all the probability distribution.

So here is Gibbs Distribution:

- General factors $\phi_{ij}(D_i)$
  - $D_i$ Can be any size $(2\le |D_i|\le n)$
- $\Phi = \{\phi(D_1),\dots,\phi(X_k)\}​$
- $\tilde{P}_{\Phi} (X_1, \dots, X_n)=\prod \limits_{I=1}^k \phi_i(D_i)​$ —> factor product
  - Unnormalized measure
- $Z_{\phi} = \sum \limits_{X_i,\dots,X_n}\tilde{P_\Phi} (X_i,\dots,X_n)​$ 
  - Partition function
- $P_{\Phi} (X_1, \dots, X_n)=\frac{1}{Z_\Phi}\tilde{P}_{\Phi} (X_1, \dots, X_n)​$

For the graph, there can be various trails/factorizations:
![image-20190417095829676](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190417095829676.png)

##### Active Trails

A trails $(X_1-\dots- X_n)​$ is active given $Z​$ if $X_I​$ is in $Z​$.

But if $X_i​$ is observed, the trail is no longer active.

![image-20190417101220770](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190417101220770.png)

### Conditional Random Field

**$P(Y|X)$**

##### CRF Representation

$\Phi = \{\phi(D_1),\dots,\phi(X_k)\}​$

$\tilde{P}_{\Phi} (X, Y)=\prod \limits_{I=1}^k \phi_i(D_i)$

$Z_{\phi} (X)= \sum \limits_{Y}\tilde{P_\Phi} (X,Y)$

 $P_{\Phi} (Y|X)=\frac{1}{Z_\Phi(X)}\tilde{P}_{\Phi} (X,Y)$

- This may seem like Bayes Model, but the difference is :
  - Bayes Model uses independent $X_i$
  - CRF uses $X_i$ come together to see their contributions. ==> **Thus, the correlations of these superpixels just don't matter.**

##### CRFs and Logistic Model

![image-20190417113051319](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190417113051319.png)

Where $1(x)$ means:

![{\displaystyle \mathbf {1} _{A}(/Users/yola/myGit/myblog/source/images/99183313e173826627fb905a9cdeec2727861484.png):={\begin{cases}1&{\text{if }}x\in A,\\0&{\text{if }}x\notin A.\end{cases}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/99183313e173826627fb905a9cdeec2727861484)

##### Applications

![image-20190417141430358](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190417141430358.png)

![image-20190417141631518](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190417143250112.png)

![image-20190417142149002](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190417142149002.png)