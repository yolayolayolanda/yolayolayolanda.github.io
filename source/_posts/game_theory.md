---
layout: poster
title: Game Theory
date: 2019-05-19 21:41:53
categories: network
toc: true
marked: true
---

*I'm back. This time the topic is game theory. And it's a blog mainly for course **Introduction to Network and Society**, but it's quite coincidental for me to have contacted to this topic several times before. Once when I was watching the film **A Beautiful Mind** and the scene of Nash striking on the game theory behind chasing a girl really impressed me. Once when I used it to solve the problem during MCM/ICM contest and once in the course **Randomized Algorithm**. So I pick this topic and want to explore a bit more.*

<!-- more -->

In the book ***[Networks, Crowds, and Markets: Reasoning about a Highly Connected World](http://www.cs.cornell.edu/home/kleinber/networks-book/)***, the authors emphasized two main things of "connectedness": one is underlying structure of the network and the other is the interconnectedness at the level of behavior. 

Human decisions are necessary and ubiquitous. As a tool to address situations when decisions are coming, game theory is widely used in many fields such as economics, political science, politics, and computer science, and can be used to model many real-world scenarios.

Here's an outline of this blog:

<!-- toc -->

#### Definitions

##### Game

A game is any situation with the following three aspects.

1. There is a set of participants, whom we call the players. In our example, you and your partner are the two players. 

2. Each player has a set of options for how to behave; we will refer to these as the player’s possible strategies. In the example, you and your partner each have two possible strategies: to prepare for the presentation, or to study for the exam. 

3. For each choice of strategies, each player receives a payoff that can depend on the strategies selected by everyone. The payoffs will generally be numbers, with each player preferring larger payoffs to smaller payoffs. We will generally write the payoffs in a payoff matrix like:

| You\Your Partner | Decision A | Decision B |
| - | - | - |
| Decision A       | [payoffs]  | [payoffs]  |
| Decision B       | [payoffs]  | [payoffs]  |

##### Game Theory

Generally speaking, game theory models the behavior of logical decision-makers in a mathematical way so that we can reasoning about behaviors in a game.

###### Underlying Assumptions

1. **Reflect all in the payoff** - Everything that a player cares about is summarized in the player’s payoffs.
2. **Consciousness** - Each player knows everything about the structure of the game.
3. **Rationality** - Each individual chooses a strategy to maximize her own payoff, given her beliefs about the strategy used by the other player.

###### Reasoning Approach

Considering each of your partner’s options separately and measure your payoff according to your options.

#### Responses and Strategies

Suppose that Player 1 chooses a strategy S and Player 2 chooses a strategy T.

***Best Responses*** - the best choice of one player to maximize their payoff, given a belief about what the other player will do.
$$
P_1(S,T) ≥ P_1(S′,T)
$$
***Strict Best Responses*** - the best choice of one player to maximize their payoff, given a belief about what the other player will do.
$$
P_1(S,T) > P_1(S′,T)
$$
***Dominant Strategy*** - the best choice of one player to maximize their payoff, no matter what the other player will do.
$$
P_i(S,T) > P_i(S′,T)
$$
***Strictly Dominant Strategy*** - the best choice of one player to maximize their payoff, no matter what the other player will do.
$$
P_i(S,T) > P_i(S′,T)
$$
***Nash Equilibrium*** - even when there are no dominant strategies, we should expect players to use strategies that are best responses to each other. That is, strategy (S,T) is a *Nash equilibrium* if S is a best response to T, and T is a best response to S.

***Mixed Strategy*** - when there is no Nash equilibria at all, we can make predictions about players’ behavior by enlarging the set of strategies to include the **possibility** of randomization; once players are allowed to behave randomly, one of John Nash’s main results establishes that equilibria always exist. For Player 1 choose a pure strategy and Player 2 choose with a probability q:
$$
\begin{align}
E[P_H] = (−1)(q)+(1)(1−q) = 1−2q\\
E[P_T] = (−1)(q)+(1)(1−q) = 1−2q
\end{align}
$$
***Social Optimality*** - A choice of strategies — one by each player — is a social welfare maximizer (or socially optimal) if it maximizes the sum of the players’ payoffs. 

#### Game Models with Strategies

##### Prisoner’s Dilemma

The prisoner’s dilemma is a hypothetical situation in which there are two prisoners (call them prisoner A and B) and they have no way of communicating. They are each given the following offer. If they each betray the other, then they will each get 4 years in prison. If prisoner A betrays prisoner B, then prisoner A goes free and prisoner B gets 10 years in prison. The same goes for prisoner B betraying prisoner A. If they both stay silent then each gets 1 years in prison. To formalize this story as a game we need to identify the players, the possible strategies and the payoffs. The two suspects are the players, and each has to choose between two possible strategies — $Confess (C)$ or $Not-Confess (NC)$. Finally, the payoffs can be summarized from the story above as a payoff matrix:
![image-20190520001550214](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190520001550214.png)

By reasoning, it is obvious that no matter what decision will your partner make, $Comfess$ is always a better choice.

In this situation, both the players have a **strictly dominant strategy**.

##### Marketing Strategy

Suppose there are two firms that are each planning to produce and market a new product; these two products will directly compete with each other. Let’s imagine that the population of consumers can be cleanly divided into two market segments: people who would only buy a low-priced version of the product, and people who would only buy an upscale version. Let’s also assume that the profit any firm makes on a sale of either a low price or an upscale product is the same. So to keep track of profits it’s good enough to keep track of sales. Each firm wants to maximize its profit, or equivalently its sales, and in order to do this it has to decide whether its new product will be low-priced or upscale.

The corresponding payoff matrix is:

![image-20190520004734903](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190520004734903.png)

Notice that, for Firm 1, *Low-Priced* is a **strictly dominant strategy**. While for Firm 2 neither option holds the **best response**. 

So in this situation, only one player has a **strictly dominant strategy**. 

##### A Three-Client Game

Suppose there are two firms that each hope to do business with one of three large clients, A, B, and C. Each firm has three possible strategies: whether to approach A, B, or C.

We can work out the payoff matrix.

![image-20190520005509216](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190520005509216.png)

We can see that no one has a **dominant strategy** in this model. So we use **Nash Equilibrium** to handle this.

Then, we can figure out that $(A,A)$ is a **Nash Equilibrium** for two firms. And under this circumstance, we only have one single **Nash Equilibrium**.

##### Exam-or-Presentation Game - A Coordination Game

Suppose you and a partner are each preparing slides for a joint project presentation; you can’t reach your partner by phone,
and need to start working on the slides now. You have to decide whether to prepare your half of the slides in PowerPoint or in Apple’s Keynote software. Either would be fine, but it will be much easier to merge your slides together with your partner’s if you use the same software.

The payoff matrix is:

![image-20190520011956683](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190520011956683.png)

This is called a Coordination Game because the two playersThis is called a Coordination Game because the two players’ shared goal is really to coordinate on the same strategy. And there are two **Nash equilibria** in this game which makes the decisions of two players ambiguous.

This remains a subject of considerable discussion and research, but some proposals have received attention in the literature. Thomas Schelling [364] introduced the idea of a **focal point** as a way to resolve this difficulty. He noted that in some games there are natural reasons (possibly outside the payoff structure of the game) that cause the players to focus on one of the Nash equilibria.

Also, if we take **social optimum** into consideration, both you and your partner will prepare for the presentation, which produces a combined payoff of 90 + 90 = 180.

##### The Hawk-Dove Game

Suppose two animals are engaged in a contest to decide how a piece of food will be divided between them. Each animal can choose to behave aggressively *(the Hawk strategy)* or passively *(the Dove strategy)*. If the two animals both behave passively, they divide the food evenly, and each get a payoff of 3. If one behaves aggressively while the other behaves passively, then the aggressor gets most of the food, obtaining a payoff of 5, while the passive one only gets a payoff of 1. But if both animals behave aggressively, then they destroy the food (and possibly injure each other), each getting a payoff of 0. Thus we have the payoff matrix:

![image-20190520013011391](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190520013011391.png)

This game has two **Nash equilibria**: (D, H) and (H, D). Without knowing more about the animals we cannot predict which of these equilibria will be played. So as in the coordination games we looked at earlier, the concept of Nash equilibrium helps to narrow down the set of reasonable predictions, but it does not provide a unique prediction.

##### Matching Pennies

A simple attack-defense game is called Matching Pennies, and is based on a game in which two people each hold a penny, and simultaneously choose whether to show heads (H) or tails (T) on their penny. Player 1 loses his penny to player 2 if they match, and wins player 2’s penny if they don’t match. This produces a payoff matrix as:
![image-20190520013757646](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190520013757646.png)

There is no Nash equilibrium for this game. This is not so surprising if we consider how Matching Pennies works. A pair of strategies, one for each player, forms a Nash equilibrium if even given knowledge of each other’s strategies, neither player would have an incentive to switch to an alternate strategy. But in Matching Pennies, if Player 1 knows that Player 2 is going to play a particular choice of H or T, then Player 1 can exploit this by choosing the opposite and receiving a payoff of +1. Analogous reasoning holds for Player 2.

##### Run-Pass Game

First, let’s consider a streamlined version of the problem faced by two American football teams as they plan their next play in a football game. The offense can choose either to run or to pass, and the defense can choose either to defend against the run or to defend against the pass. Here is how the payoffs work. 

- If the defense correctly matches the offense’s play, then the offense gains 0 yards. 
- If the offense runs while the defense defends against the pass, the offense gains 5 yards. 
- If the offense passes while the defense defends against the run, the offense gains 10 yards. 

Hence we have the payoff matrix .

![image-20190520015151724](https://raw.githubusercontent.com/yolayolayolanda/yolayolayolanda.github.io/master/images/image-20190520015151724.png)

Obviously, there is no **Nash Equilibrium**.

However, we can use **Mixed Strategy** for this.

- First, suppose the defense chooses a probability of q for defending against the pass. Then the expected payoff to the offense from passing is 
  $$
  (0)(q) + (10)(1 − q) = 10 − 10q
  $$
  , while the expected payoff to the offense from running is 
  $$
  (5)(q) + (0)(1 − q) = 5q.
  $$

  To make the offense indifferent between its two strategies, we need to set $10−10q = 5q$, and hence $q = 2/3$. 

- Next, suppose the offense chooses a probability of p for passing. Then the expected 

  payoff to the defense from defending against the pass is 
  $$
  (0)(p) + (−5)(1 − p) = 5p − 5,
  $$
  with the expected payoff to the defense from defending against the run is 
  $$
  (−10)(p) + (0)(1 − p) = −10p.
  $$
  To make the defense indifferent between its two strategies, we need to set $5p−5 = −10p$, and hence $p = 1/3$. 

Thus, the only possible probability values that can appear in a mixed-strategy equilibrium are $p = 1/3$ for the offense, and $q = 2/3$ for the defense, and this in fact forms an equilibrium. 
