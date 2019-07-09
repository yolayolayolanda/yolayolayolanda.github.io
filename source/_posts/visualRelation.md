---
title: Visual Relation 
date: 2019-04-19 19:13:46
categories: CV
mathjax: true
---

## Visual Relationship Detection with Language Priors

##### Visual Genome dataset

They designed a dataset directly for connecting language and vision using crowdsourced dense image annotations.

<!-- more -->

- 33K object categories
- 42K relationship categories

##### Difficulties in Visual Relationship

1. Visual Phrases = Objects + Interactions ==> complicated meaning

2. From object detection —> visual relationships: 

   1. Interactions are various
   2. even slight changes in the picture makes the meaning completely different

   ![image-20190420133216891](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190420133216891.png)

3. Previous work attempted to detect only a handful of visual relations and do not scale: one detector for one relationship

   ![image-20190420134154239](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190420134154239.png)

4. Training suffers from the **long tail distribution** of relationships of VG dataset. Some relationships are far more frequent than others

   ![image-20190420140244094](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190420140244094.png)

5. **Quadratic explosion** of N objects and K relationships leads to $N^2K$ detectors

##### Model

![image-20190420140707375](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190420140707375.png)

- Visual Model + Language Model
- Localize the objects with bounding box and label with ***<object1 - predicate - object2>***

###### Visual Model

***==> Visual Model can tackle the Quadratic Explosion***.   Learn objects and relationships independently.

![image-20190420143345924](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190420143345924.png)

- Region proposal to generate object proposals: **selective search**

- Detectors to detect 

  - Object detector: pre-trained CNN(VGG) 
  - Relationship detector: pre-trained CNN(VGG).
    -  To detect [on, in, ride, ...]

- Combine the three output scores and define the **Ranking loss** by:
  $V(R_{ ⟨i, k, j⟩ },Θ|⟨O1,O2⟩) = P_i (O1)(z^T_k CNN(O1,O2) + s_k)P_j (O2)$

  , where where $Θ$ is the parameter set of {$z_k,s_k$}. $z_k$ and $s_k$ are the parameters learnt to convert our CNN features to relationship likelihoods

  - Get a final score describing how well the $<object_1-predicate-object_2>$ triple is.

- Fine-tune both detectors jointly on a bigger dataset



###### Language Model

***==> Language Model can tackle the Long tail distribution***.  

For aiding strange triples

- ***Projection Function***
  - Pre-trained *word2vec* to cast objects into a word embedding space

  - Relationship projection function
    - $f(R_{ ⟨I, k, j⟩ },W) = w_k^T [word2vec(t_i), word2vec(t_j)] + b_k$

      , where $t_j$ is the word (in text) of the $j_{th}$ object category. $w_k$ is a 600 dim. vector and $b_k$ is a bias term. W is the set of {$w_1,b_1\}, … ,\{w_k,b_k$}, where each row presents one of our K predicates.

- ***Training Projection Function***

  - Heuristic function that projects similar relationships closer:
    $$
    \frac{[f(R, W)− f(R′, W)]^2} {d(R,R′)} = constant, ∀R,R'
    $$
    , where *d(R, R’)* is the sum of cosine distances between two relationships

    ![image-20190420172803979](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190420172803979.png)

  - Minimize the variance of sampled pairs(500K) of relationships
    $$
    K(W) = var(\frac{[f(R, W) − f(R′, W)]^2 }{ d(R, R′)} \ \ \ ∀R,R' )
    $$
    , where var() is a variance function.

- ***Likelihood of a Relationship***

  - Intuition: encourage correct ranking with a small margin.

  - If relationship *R* occurs more frequently than relationship *R*′ in the
    training data, then it should have a higher likelihood of occurring
    again.
    $$
    L(W) = 􏰃 \sum_{R,R'}\max{f(R′,W) − f(R,W) + 1,0}
    $$

  - Projection function *f()* should generalize for all relationship combinations even if not appearing in the training data. 

  - Minimize *L(W)* enforces that a relationship with a lower likelihood of occurring has a lower projection function *f()* score. 

###### Training Both Modules

- Ranking loss function:

  $C(Θ,W)= 􏰃 \max\{1−V(R,Θ|⟨O1,O2⟩)f(R,W)+\max_\limits{⟨O1',O2'⟩ \ne ⟨O1,O2⟩,R'\ne R} V (R′ , Θ|⟨O1′ , O2′ ⟩)f (R′ , W), 0\} $

  ![image-20190420174426670](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190420174426670.png)

- Total loss:

  $\min_{Θ,W}{C(Θ, W) + λ_1L(W) + λ_2K(W)}$

- Hold one train another strategy

##### Model Details 

![image-20190420174829225](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190420174829225.png)

##### Result

![image-20190420175117613](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190420175117613.png)

![image-20190420175451967](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190420175451967.png)

![image-20190420175505162](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190420175505162.png)

##### Drawbacks

No higher level rationale.

##### Strengthes

Good at novel relationships.

## Detecting Visual Relationships with Deep Relational Networks [DR-Net]

![image-20190420180955386](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190420180955386.png)

##### Overall pipeline

###### Object detection

Faster R-CNN

###### Pair filtering 

Generate $N(N-1)$ pairs. Then use a shallow neuron network to filter the obviously meaningless pairs. This filter takes into account both the spatial configurations (*e.g*. objects too far away are unlikely
to be related) and object categories(*e.g*. certain objects are
unlikely to form a meaningful relationship). Then fed them to the joint recognition module.

###### Joint recognition

- Appearance — feature maps of the objects
- Spatial Configurations — *dual spatial masks* relative positions and relative sizes
- Statistical Relations — similar to language prior
- Integrated Prediction — get the compressed pair feature

##### DR Net (Deep Relation)

**Input:** Appearance of two objects and subjects + compressed pair feature

*Waiting to be finished….*

$x_s,x_o,x_r$ are appearance features of subject and object in Joint Recognition, and compressed pair feature. $q_r$ be a vector of the posterior probabilities for r. 

$W_{r, s}(r,s)=𝞅_{r,s}(r,s)$

##### Results

![image-20190420182107032](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190420182107032.png)

##### Drawbacks

Because of the use of softmax at the last layer of DR-Net, only one relation is determined. So the existing problem is that it cannot figure out the multi-relationships.



