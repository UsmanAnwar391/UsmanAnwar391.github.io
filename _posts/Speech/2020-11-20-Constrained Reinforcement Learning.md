---
layout: post
title: Constrained Reinforcement Learning
categories: [Reinforcement Learning]
mathjax: true
---

> This post reviews some papers submitted to ICLR 2021 on constrained reinforcement learning.

## Robust Constrained Reinforcement Learning For Continuous Control
This paper considers a problem where there is some sort of model misspecification present **which results in uncertain dynamics** e.g. one leg of the agent has different length than the 'real world' length. Mathematically, this is equivalent to assuming an uncertain dynamics model $P(s'|s,a) \sim \mathcal{P}(s,a)$.


This work considers this setting in context of CMDP and shows that robust constrained Bellman Operator holds in this setting and provide experimental validation on RealWorld-RL suite of deepmind.

---

## Explicit Pareto Front Optimization For Constrained RL

This work considers a CMDP with a primary reward function that is to be maximized subject to some auxillary funcions being kept less than some threshold value.
$$\max_\pi \mathbb{E}[R(\tau)] \ \ \text{s.t.} \ \mathbb{E}[\vec C(\tau) - \vec \epsilon] \leq 0$$ where $\vec C(\tau)$ is a vector of measurements corresponding to $m$ constraint functions. 

Note that corresponding to this CMDP, we can define a Multi-Objective MDP (MO-MDP) with reward function $\hat R(\tau) = [R(\tau),\ -\vec C(\tau)]$ and use $\vec \epsilon$; note that solution of a CMDP lies on the Pareto Front of the corresponding MO-MDP. 

This paper uses Hierarchical RL to iteratively learn the the preference vector $\vec \epsilon$ and 


---

## Reconnaissance For Reinforcement Learning With Safety Constraints
The idea in this paper is to decompose the CMDP into *Reconnaissance MDP (R-MDP)* and *Planning MDP (P-MDP)*. Constraint or cost function is minimized in R-MDP and this results in  a safe baseline policy $\pi_{\text{safe}}$.

For a given state, all the actions deemed safe by $\pi_{\text{safe}}$ form a set of safe actions $\mathcal{A}_{\text{safe}} \subseteq \mathcal{A} $ and all the states that have at least one safe action make a set of safe states $ \mathcal{S}_{\text{safe}}$. 

Then we can define a space of safe policies with respect to $\pi_{text{safe}}$ as follows:

$$\Pi^{\pi_{\text{safe}}} = \{\pi | \ \text{for} \ s \ \in \mathcal{S}_{\text{safe}}, \text{support}(\pi(\cdot|s)) \subseteq \mathcal{A}_{\text{safe}}, \\ \hspace{50pt} \text{otherwise} \ \pi(\cdot|s) = \text{argmin}_a Q^c_{\pi_\text{safe}}(s,a) \}$$

P-MDP then searches for optimal policy in $\Pi^{\pi_\text{safe}}$.

The paper is not well written (lots of useless mathiness) but contains some interesting ideas. Worth a second read if this project goes forward.

---

## Learning What Not To Model

This paper considers a Gaussian Process Regression where not only positive examples are available (to which Gaussian Process must fit), but also negative examples which should not be fit. 

This paper models these negative data pints as blobs of Gaussian $\mathcal{N}^-$ and *maximizes* the divergence of the $\mathcal{GP}$ with respect to the negative data points distribution $\mathcal{N}^-$.

:::success
There is no approach currently which tries to build a neural network based world model with such constraints. This may be a direction worth trying out.
:::

---

## Safe Reinforcement Learning With Natural Language Constraints

:::info
This paper categorizes constraints as to be of three types: 
1. Budget Constraints: You can do thing X but only Y times.
2. Relation Constraints: You must keep away from walls by 4 units.
3. Sequential Constraints (Non-Markovian): You can do thing X only if you have done thing Y before.

Will be interesting to add a similar discussion in our ICRL paper.
:::

This paper proposes to use "Natural Language" to specify constraints. A "constraint interpreter" module is introduced which takes current observation embedding and natural text embedding as input and produces a "constraint mask" and "budget mask" which are then concatenated with observation embedding and fed to policy network which produces an action. 

Constraint interpreter network is pre-trained by maximizing a binary cross entropy loss on a dataset collected by executing a random policy.

Policy is then trained by Projection Based Constrained Policy Optimization (PCPO) where budget mask values are used as approximation of true cost.

:::warning
I am unsure about where the cost estimate comes from, my guess is this comes from budget mask but this is only weakly alluded to in second last paragraph of page 4 (line 5).
:::

:::danger
**My Concerns**
1. It seems like that in constraint interpreter module pre-training (eq 2 and eq 3 in paper and paragraph above Experiments section), authors assume knowledge of true cost mask and true budget threshold. Is this a good thing? If you already know these then why even bother with natural language constraints? 

2. They use budget mask values as an approximation of true cost, but this mask quality is dependent on constraint interpreter module and expected to evolve with training? Will this not result in optimization problems similar to the ones faced by us in CIRL? Plus is the cost not associated with 

:::

---

## A Primal Approach To Constrained Policy Optimization
The paper considers setting of constrained RL (hard constraints) where there is one primary reward function that is to be maximized subject to the fact that auxilary cost (constraint) functions are below some threshold value.

This paper proposes that at any update step, if there is any constraint function that is being violated (if multiple constraint functions are being violated, then choose any of the violated function at random), then policy minimizes this cost function else primary reward function is maximized. 

There is extensive analysis which shows that this approach converges at a sub linear rate.

---

## Balancing Constraints and Rewards With Meta-Gradient D4PG

**Meta Gradients**
Meta gradients are used to adjust hyperparameters during training. We assume that there are two losses: $L_{inner}$ and $L_{outer}$. $L_{inner}$ is used to make usual updates to the learning parameters $\theta$ and then $L_{outer}$ is used to make updates to a selected set of hyperparameters caller meta-parameters.

**RCPO-D4PG**
This paper proposes to treat learning rate of Lagrangian multiplier in RCPO as a meta-parameter and proposes to treat critic loss $L_{outer} = L_{critic} = ||r(s,a) + \gamma V(s') - V(s)||^2$ and derives a gradient to update learning rate $\alpha_\lambda$ based on this loss.

---

## Learning Safe Policies With Cost Sensitive Advantage Estimation

The paper generalizes '[Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)' to constrained RL with hard constraints. Note that

$$\hat A_t^{\text{GAE}(\gamma, \lambda)} := \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}$$ where $\lambda$ and $\gamma$ are hyperparameters with domain $[0,1]$ and $\delta_n$ is n-step temporal difference return.

This paper proposes following estimator for constrained RL with hard constraints.

$$\hat A_t^{\text{CGAE}(\gamma, \lambda)} := \sum_{l=0}^\infty (\gamma \lambda)^l \alpha_{t+l}\delta_{t+l}$$

where $\alpha_t$ is a binary variable denoting whether  a transition $(s_t, a_t, s_{t+1})$ is safe ($\alpha_t=1$) or not safe ($\alpha_t=0$) i.e. $\alpha_t = \mathbb{I}[C(s_t, a_t, s_{t+1}) > 0]$. 

This is equivalent to the following reward shaping: 
$$\begin{equation}
\bar R(s_t, a_t, s_{t+1}) = 
\begin{cases}
R(s_t, a_t, s_{t+1}) \hspace{43pt} if \hspace{10pt} \alpha_t = 1 \\
\mathbb{E}_{a, s' \sim \tau} [R(s_t, a, s')] \hspace{23pt} if \hspace{10pt} \alpha_t = 0
\end{cases}
\end{equation}$$

This is equivalent to setting a reward of constrained state equal to the average reward of states that an agent can transition to from the said constrained state. Intuitively, this serves as incentive for the agent to get out of that state quickly.

:::info
We do not want to go into a constrained state, hence, should we not try to ensure that (effective) reward of going into a constrained state $s^c$ from current state $s$ should be less than of all possible safe states $s'$ we could transition to from $s$?

Alternatively, what if the expectation is not over the next possible states, but the previous states (states from which a transistion to constrained state could have been made)?
:::


:::success
Consider a one step MDP where reward of constrained transition is higher than all other transitions; in this even where this reward is replaced with expected one step reward; it remains higher than the other transitions and does not dissuade agent from carrying out that transition.

Instead of the proposed reward shaping, I (Usman Anwar) propose the following reward shaping (Shehryar Malik desires dissociation with this):
$$\begin{equation}
\bar R(s_t, a_t, s_{t+1}) = 
\begin{cases}
\mathbb{E}_{a, s' \sim \tau} [R(s_t, a_t, s')] + R(s_t, a_t, s_{t+1}) \hspace{10pt} if \hspace{10pt} \alpha_t = 1 \\
\mathbb{E}_{a, s' \sim \tau} [R(s_t, a_t, s')] \hspace{80pt} if \hspace{10pt} \alpha_t = 0
\end{cases}
\end{equation}$$

Under the assumption that rewards are lower bounded by 0; i.e. there is no negative reward, this reward shaping intuitively sets the reward of constrained transitions equal to average one step reward and reward of unconstrained transitions equal to sum of average one step reward plus reward for that transition; thus *disincentivizing* constrained transition. 

Let $\hat r= \mathbb{E}_{a, s' \sim \tau} [R(s_t, a_t, s')] = V(s_t) - \gamma V(s_{t+1})$ (See Appendix (7.4) of this paper for this identity.)

Then our shaped reward at any time step $t$ is $\bar r_t = \hat r_t + \alpha_t r_t$. 

Then, with symbols $\hat{A}_t^{(k)}$ and $\hat{A}_t^{\text{GAE}(\gamma, \lambda)}$ having the same meaning as GAE paper, 


$$\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \delta_{t+l}^V =\sum_{l=0}^{k-1}\gamma \alpha_{t+l}r_{t+l}$$

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty}(\gamma \lambda)^l\delta_{t+l}^V = \sum_{l=0}^{\infty}(\gamma \lambda)^l \alpha_{t+l}r_{t+l}$$

This is easy to follow as $\delta_t^V = \bar r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD return using shaped reward and $\bar r_t$ is the shaped reward. Putting in value of $\bar r_t$ gives $\delta_t^V = (V(s_t) - \gamma V(s_{t+1}) + \alpha_t r_t) + \gamma V(s_{t+1}) - V(s_t) = \alpha_t r_t$. 

:::
---

## Density Constrained Reinforcement Learning

The paper posits that defining a cost function is non-intuitive and prone to error. As opposed to that, constraints *on states* can be effectively expressed in the form of how *often* particular states should be visited.

Typcially, state density function $\rho: [0, \infty) \rightarrow \mathbb{R}_{\geq0}$ is defined as the unnormalized accumulative discounted probability of being in a particular state $s$ throughout the length of one rollout of stationary policy $\pi$ where the initial state distribution is $\phi(s)$.

$$\rho^\pi(s) = \sum_{t=0}^\infty \gamma^t P(s^t=s|\pi, s_0 \sim \phi)$$

It is easy to show that we can rewrite $\rho^\pi(s)$ as follows:
$$\rho^\pi(s) = \phi(s) + \gamma \int_\mathcal{S} \int_\mathcal{A} \pi(a|s') P(s'|s, a) \rho^\pi(s') da \ ds'$$

:::success
**Density Constrained RL (DCRL) Problem**: Given a MDP $\mathcal{M} = <\mathcal{S}, \mathcal{A}, P, R, \gamma>$ and an initial state distribution $\phi(s)$, DCRL finds the optimal policy $\pi^*$ that maximizes the expectation of reward signal $\int_\mathcal{S} \phi(s) V^\pi(s) ds$ subject to constraints on stationary density function represented as $\rho_\min(s) \leq \rho^{\pi^*}(s) \leq \rho_\max(s) \ \forall s \in \mathcal{S}.$
:::

Note that in stoachastic dynamics setting, $r(s,a) = \int_\mathcal{S}P(s'|s,a)R(s', a, s) ds$ is the expected reward attained when action $a$ is executed in state $s$.  

Then the paper claims (in theorem 1) that following objectives $J_d$ (optimization of density function) and $J_p$ (Optimization of Q Function) are dual to each other and if they are both feasible, then $J_d=J_p$ and they both share the same optimal policy $\pi^*$.

$$J_d = \max_{\rho, \pi} \int_\mathcal{S} \int_\mathcal{A} \rho^\pi(s,a)r(s,a) da ds \hspace{10pt} \\ \text{where} \hspace{10pt} \rho^\pi(s,a) = \pi(a|s)\rho(s)$$

$$J_p = \max_{Q, \pi} \int_\mathcal{S} \phi(s) \int_\mathcal{A} Q^\pi(s,a) \pi(a|s) da ds \hspace{10pt} \\ \text{where} \hspace{10pt} Q^\pi(s,a) = r(s,a) + \gamma \int_\mathcal{S}P(s'|a,s) \int_\mathcal{A} \pi(a'|s') Q^\pi(s',a') da'ds'$$


The paper goes on to show (in theorem 2) that constrained version of the above objectives are also dual to each other and if both constrained versions have a feasible solution and KKT conditions hold, then they have same optimal policy.

$$J^c_d = \max_{\rho, \pi} \int_\mathcal{S} \int_\mathcal{A} \rho^\pi(s,a)r(s,a) da ds \hspace{10pt} \\ \text{s.t.} \hspace{10pt} \rho_\min(s) \leq \rho^\pi(s) \leq \rho_\max (s)$$

If we define Lagrange multipliers $\sigma_-(s): \mathcal{S}\rightarrow \mathbb{R}_{\geq 0}$ and $\sigma_+(s): \mathcal{S}\rightarrow \mathbb{R}_{\geq 0}$ corresponding to $\rho_\min$ and $\rho_\max$ respectively; then constrained version of primal problem can be formulated as:

$$J_p = \max_{Q, \pi} \int_\mathcal{S} \phi(s) \int_\mathcal{A} Q^\pi(s,a) \pi(a|s) da ds \hspace{10pt} \\ \text{where} \hspace{10pt} Q^\pi(s,a) = r(s,a) +\sigma_-(s) - \sigma_+(s) + \gamma \int_\mathcal{S}P(s'|a,s) \int_\mathcal{A} \pi(a'|s') Q^\pi(s',a') da'ds'$$

Note that in constrained version of primal problem, reward is adjusted by a factor equivalent to Lagrange Multipliers.

**Primal Dual Algorithm For DCRL**
Based on the observation above, paper proposes to solve DCRL problem by updating the policy in primal domain (by any typical unconstrained RL algorithm like TRPO or DDPG) by using reward $r(s,a) = r(s,a) + \sigma_-(s) - \sigma_+(s)$ and then in dual domain, evaluate the state density function of the current policy $\pi$. If the KKT conditions, $\sigma_+ \cdot (\rho^\pi - \rho_\max)=0$, $\sigma_- \cdot (\rho_\min - \rho^\pi)=0$ and $\rho_\min \leq \rho^\pi \leq \rho_\max$ are not satisfied, then Lagrange multipliers are updated in the following way: 

$$\sigma_+ = \max(0, \sigma_+ + \alpha(\rho^\pi - \rho_\max)) \\
  \sigma_- = \max(0, \sigma_+ + \alpha(\rho_\min - \rho^\pi))$$

:::warning
My best guess here is that $\alpha$ here is learning rate, however, the paper also uses $\alpha$ to denote budget. There is lack of clarity on this in the paper.
:::

Section 3.3 in the paper text is relevant for implementation and includes details on how to compute state density function and Lagrange Multiplier updates.

:::danger
**My Concerns**
Not obvious how to set $\rho_\max$ and $\rho_\min$; especially the fact that these are unnormalized and discounted densities makes it counter intuitive to select a value for them easily. Details on this may perhaps be available in this paper: https://arxiv.org/pdf/1902.09583.pdf.
:::
