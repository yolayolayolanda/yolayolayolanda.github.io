---
layout: poster
title: What I Read about Data Science and Deep Learning [Scrappy Collection]
date: 2019-04-13 22:12:57
categories: DS&DL
mathjax: true
---

#### Activation Function

###### [Activation Functions in Neural Networks](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)

- Sigmoid, tanh, Softmax, ReLU, Leaky ReLU EXPLAINED !!!
- By [SAGAR SHARMA](https://towardsdatascience.com/@sagarsharma4244)

<!-- more -->

###### [Understanding Activation Functions in Neural Networks](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)

- By [Avinash Sharma V](https://medium.com/@avinashsharmav91)

###### [Stanford-CS231n](http://cs231n.github.io/neural-networks-1/)

- High Recommendation for this one

##### Sigmoid:

-  $σ(x)=\frac{1}{1+e^{−x}}$
- [0,1]
- For RNN, LSTM
- Kind of out of fashion, because of the **drawbacks**:
  - *Sigmoids saturate and kill gradients*. It gets harder and harder to learn because the gradient at these regions is almost zero, which is bad for backpropagation. What's more, too large or too small initial weights will cause network barely to learn.
  - *Sigmoid outputs are not zero-centered*. This is undesirable since neurons in later layers of processing in a Neural Network (more on this soon) would be receiving data that is not zero-centered. This could introduce undesirable zig-zagging dynamics in the gradient updates for the weights. But it can be migrated by patch learning.

##### Tanh

- $\tanh(x) = 2 \sigma(2x) -1$
- [-1,1]
- Like the sigmoid neuron, its activations saturate and used most in LSTM
- But unlike the sigmoid neuron its output is zero-centered, which is better.

##### ReLU

- $f(x) = \max(0, x)$
- [0,$\infty​$]
- Good:
  - Simplicity. Just thresholding.
  - Better performance for convergence.
- Bad:
  - Dying ReLU: ReLU units can be fragile during training and can “die”. If learning rate is too high, it is probable that too much (40%) of the neurons can be dead(0). 

##### Leaky ReLU

- $f(x) = \mathbb{1}(x < 0) (\alpha x) + \mathbb{1}(x>=0) (x)$
- [$-\infty,\infty$]
- It can fix the “dying ReLU” problem.
- **PReLU**: The slope in the negative region can also be made into a parameter of each neuron. (Customized ReLU)

##### Maxout

- $\max(w_1^Tx+b_1, w_2^Tx + b_2)​$
- Other types of units have been proposed that do not have the functional form $f(w^Tx+b)$ where a non-linearity is applied on the dot product between the weights and the data.
- Good:
  - All the benefits of ReLU
  - But no dying ReLU problem
- Bad:
  - Doubled number of parameters for every neuron 

#### Vanishing & Exploding gradient

###### [Vanishing/Exploding Gradients (C2W1L10) - Andrew Ng](https://youtu.be/qhXZsFVxGKo)

Because of "deep", >1 becomes bigger and bigger along the network(exploding) and <1 becomes smaller and smaller(vanishing). [A function of the number of the layers ]

#### LSTM

###### [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

- From [colah's blog](http://colah.github.io/)

![LSTM](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190414150821124.png)

###### [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)



 