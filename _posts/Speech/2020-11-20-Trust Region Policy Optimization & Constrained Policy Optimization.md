---
layout: post
title: Trust Region Policy Optimization & Constrained Policy Optimization
categories: [Research Papers, Reinforcement Learning]
mathjax: true
---

# Trust Region Policy Optimization
Due to Kakade and Langford 2002, we have the following result comparing the return of two policies $\pi$ and $\pi'$. 

$$
\begin{align}
J(\pi') &=  J(\pi) + \mathbb{E}_{\substack{s \sim d^{\pi'} \\ a \sim \pi'}}[A^\pi(s, a)] \\ 
J(\pi') - J(\pi) &=  \mathbb{E}_{\substack{s \sim d^{\pi'} \\ a \sim \pi'}}[A^\pi(s, a)] 
\label{eq1}
\end{align}$$

Intuitively, this update tells us that policy update $\pi \rightarrow \pi'$ will result in policy improvement, if the new policy chooses action which had positive advantage under the current policy. Unfortunately, using this identity directly to search for $\pi'$ is not feasible as evaluating RHS would require off-policy evaluation (determine $A^\pi$ via samples from $\pi'$.)

TRPO proposes to get around this by using samples from $\pi$.

$$
J(\pi') \approx L_\pi(\pi') = J(\pi) + \mathbb{E}_{\substack{s \sim d^{\pi} \\ a \sim \pi'}}[A^\pi(s, a)]
$$

<!-- :::info
### Why this approximation is valid?
Theorem 1 of TRPO paper shows that approximation error is bounded by the KL divergence between the current and new policy.

$$L_\pi(\pi') - J(\pi') \leq C D_{KL}^{\text{max}}(\pi, \pi')$$ where $$D_{KL}^{\text{max}}(\pi, \pi') = \max _s D_{KL}(\pi(\cdot|s)|\pi'(\cdot|s)) \\ 
C = \frac{4\epsilon \gamma}{(1-\gamma)^2} \hspace{10pt},  \hspace{30pt} \epsilon = \max_s |\mathbb{E}_{a \sim \pi'}[A^\pi (s, a)]|
$$ 
::: -->

This gives us an alternative objective to maximize. Further note that we have policies paramterized with $\theta$ and we denote current policy paramters as $\theta_{old}$. $$\max_\theta [L_{\theta_{old}}(\theta) - C D_{KL}^{\text{max}}(\pi, \pi')]$$. This objective has two problems 

(1) Step size $C$ suggested by theory is very small; making this objective impractical.
(2) Further, evaluation of $D_{KL}^{\text{max}}(\theta_{old}, \theta)$ is intractable for high dimensional state space.

To fix (1), we convert the penalty by KL divergence into an explicit constraint. For (2), we use average KL divergence between the two policies instead of the maximum KL divergence. This gives us the following form of objective:

$$
\max_\theta L_{\theta_{old}}(\theta) \hspace{10pt} \text{s.t.} \hspace{10pt} \bar D_{KL}(\theta_{old}, \theta) < \delta$$ where $$\bar D_{KL}(\theta_{old}, \theta) = \mathbb{E}_{s \sim \theta_{old}}[D_{KL}(\theta_{old}(\cdot|s)|\theta(\cdot|s))]$$. We modify the objective $L_{\theta_{old}}(\theta) = \sum_s d^{\theta_{old}}(s) \sum_a \pi(a|s) A^{\theta_{old}}(a,s)$ by replacing the sums over states and ations with expectations. 

$$L_{\theta_{old}}(\theta) = \mathbb{E}_{s \sim d^{\theta_{old}}(s), a \sim \theta(a|s)} [A^{\theta_{old}}(a,s)] \hspace{10pt} \text{s.t.} \hspace{10pt} \bar D_{KL}(\theta_{old}, \theta) < \delta$$. Note that we do not yet know $\theta(a|s)$ making it difficult to sample from this distribution. Hence, instead of using expectation over $\theta(a|s)$, we instead use an importance sampling estimator with $\theta_{old}(a|s)$ as the sampling distribution. 

$$L_{\theta_{old}}(\theta) = \mathbb{E}_{s \sim d^{\theta_{old}}(s), a \sim \theta_{old}(a|s)} \left[\frac{\theta(a|s)}{\theta_{old}(a|s)}A^{\theta_{old}}(a,s)\right] \hspace{10pt} \text{s.t.} \hspace{10pt} \bar D_{KL}(\theta_{old}, \theta) < \delta$$. Even in this form, we can not yet optimize this loss efficiently. Hence, we instead perform a linear approximation of $L_{\theta_{old}}$ and quadratic approximation of KL divergence constraint. Luckily for us, these approximations have nice closed forms.

**LInear Approximation of $L_{\theta_{old}}(\theta)$**

By Taylor series expansion of $L_{\theta_{old}}(\theta)$ around $\theta_{old}$.

$$
\begin{align}
L_{\theta_{old}}(\theta) \approx L_{\theta_{old}}(\theta_{old}) + \nabla_\theta L_{\theta_{old}}(\theta)\Big |_{\theta=\theta_{old}} (\theta - \theta_{old})
\end{align}
$$ $$ \begin{align}L_{\theta_{old}}(\theta_{old}) &= \mathbb{E}_{s,a \sim \theta_{old}}[A^{\theta_{old}}(a,s)] 
\\ &= \mathbb{E}_{s,a \sim \theta_{old}}[Q^{\theta_{old}}(a,s) - V^{\theta_{old}}(s)]
\\ &= \mathbb{E}_{s}[\mathbb{E}_{a \sim \theta_{old}}[Q^{\theta_{old}}(a,s)]] - \mathbb{E}_{s,a \sim \theta_{old}}[V^{\theta_{old}}(s)] 
\\ &= \mathbb{E}_{s}[V^{\theta_{old}}(s)] - \mathbb{E}_{s}[V^{\theta_{old}}(s)] 
\\ &= 0
\\ \nabla_\theta L_{\theta_{old}}(\theta) &= \nabla_\theta \mathbb{E}_{s \sim d^{\theta_{old}(s)},a \sim \theta_{old}}\left[\frac{\theta(a|s)}{\theta_{old}(a|s)}A^{\theta_{old}}(a,s)\right] 
\\ &= \nabla_\theta \int_{\mathcal{S}} d^{\theta_{old}}(s)\int_{\mathcal{A}} \theta_{old}(a|s) 
\left[\frac{\theta(a|s)}{\theta_{old}(a|s)}A^{\theta_{old}}(a,s)\right] da \ ds
\\ &= \int_{\mathcal{S}} d^{\theta_{old}}(s)\int_{\mathcal{A}} \theta_{old}(a|s) 
\left[\frac{\nabla_\theta \theta(a|s)}{\theta_{old}(a|s)}A^{\theta_{old}}(a,s)\right] da \ ds
\\ &= \int_{\mathcal{S}} d^{\theta_{old}}(s)\int_{\mathcal{A}} \theta_{old}(a|s) 
\left[\frac{\theta(a|s)}{\theta_{old}(a|s)}  \nabla_\theta \log \theta(a|s) A^{\theta_{old}}(a,s)\right] da \ ds
\\&= \mathbb{E}_{s \sim d^{\theta_{old}(s)},a \sim \theta_{old}}\left[\frac{\theta(a|s)}{\theta_{old}(a|s)} \nabla_\theta \log \theta(a|s) A^{\theta_{old}}(a,s)\right] 
\\ \nabla_\theta L_{\theta_{old}}(\theta) \Big |_{\theta=\theta_{old}} &= \mathbb{E}_{s \sim d^{\theta_{old}(s)},a \sim \theta_{old}}\left[\nabla_\theta \log \theta_{old}(a|s) A^{\theta_{old}}(a,s)\right]
\end{align}
$$

Note that the final expression of $\nabla_\theta L_{\theta_{old}}(\theta) \Big |_{\theta=\theta_{old}}$ has the same form as policy gradient. Now if we assume $g = \nabla_\theta L_{\theta_{old}}(\theta) \Big |_{\theta=\theta_{old}}$, then we can compactly write:
$$L_{\theta_{old}}(\theta) \approx g^T (\theta - \theta_{old})$$

**Second Order Approximation of Average KL Divergence**
Note that
$$
\begin{align}
\bar D_{KL}(\theta_{old}, \theta) &= \mathbb{E}_{s \sim \theta_{old}}[D_{KL}(\theta_{old}(\cdot|s)|\theta(\cdot|s))] 
\\&= \mathbb{E}_{s \sim \theta_{old}} \int_{\mathcal{A}} \theta_{old}(a|s) \log \left[\frac{\theta_{old}(a|s)}{\theta(a|s)}\right]da
\\ &= \mathbb{E}_{s,a \sim \theta_{old}}\left[\theta_{old}(a|s) \log \left(\frac{\theta_{old}(a|s)}{\theta(a|s)}\right)\right] 
\\ \text{Then} \hspace{10pt} \nabla_{\theta}\bar D_{KL}(\theta_{old}, \theta)  &= \nabla_\theta \mathbb{E}_{s,a \sim \theta_{old}}\left[\theta_{old}(a|s) \log \left(\frac{\theta_{old}(a|s)}{\theta(a|s)}\right)\right] 
\\ &= \mathbb{E}_{s,a \sim \theta_{old}}\left[\theta_{old}(a|s) \nabla_\theta \log \left(\frac{\theta_{old}(a|s)}{\theta(a|s)}\right)\right] 
\\ &= \mathbb{E}_{s,a \sim \theta_{old}}\left[\theta_{old}(a|s) \left(\frac{\theta(a|s)}{\theta_{old}(a|s)}\right) \left(- \frac{\theta_{old}(a|s)}{\theta^2(a|s)}\right) \right] 
\\ &= - \mathbb{E}_{s,a \sim \theta_{old}}\left[ \left(\frac{\theta_{old}(a|s)}{\theta(a|s)}\right) \right] 
\end{align}
$$

It is easy to see that $\bar D_{KL}(\theta_{old}, \theta_{old}) = 0$ and $\nabla_\theta \bar D_{KL}(\theta_{old}, \theta) \Big |_{\theta=\theta_{old}} =0$
Hence, Taylor expansion till second order terms of $\bar D_{KL}(\theta_{old}, \theta)$ reduces to following.

$$\bar D_{KL}(\theta_{old}, \theta) \approx \frac{1}{2}(\theta - \theta_{old})^TH(\theta - \theta_{old})$$ where $H$ is the Hessian matrix of average KL divergence.

**Final Form Of Objective**
This gives us following final form of objective 
$$\max_\theta g^T(\theta - \theta_{old}) \hspace{10pt} \text{s.t.} \hspace{10pt} \frac{1}{2}(\theta - \theta_{old})^TH(\theta - \theta_{old}) < \delta$$

Let $x = -(\theta - \theta_{old})$

Then our objective takes the form

$$\min_x g^Tx \hspace{10pt} \text{s.t.} \hspace{10pt} \frac{1}{2}x^THx < \delta$$ This is a linear program with quadratic constraints. We can thus use Lagrangian method to solve this by levaraging the fact that this problem is convex and strong duality holds.
$$
\begin{align}
p^* &= \min_x \max_{\lambda\geq 0} g^Tx + \frac{\lambda}{2}\left(x^THx -\delta\right)
\\ d^* &= \max_{\lambda\geq 0} \min_x g^Tx + \frac{\lambda}{2} x^THx -\frac{\lambda}{2} \delta
\\ \mathcal{L}(x, \lambda) &= g^Tx + \frac{\lambda}{2} x^THx -\frac{\lambda}{2} \delta \\
\\ \nabla_x \mathcal{L}(x,\lambda) &= 0 \hspace{130pt} \text{By KKT conditions} \\
\\ \nabla_x \mathcal{L}(x,\lambda) &= \nabla_x \left(g^Tx + \frac{\lambda}{2} x^THx -\frac{\lambda}{2} \delta\right) \\
0 &= g + \frac{\lambda}{2}(2Hx) \\
\implies x^* &= - \frac{1}{\lambda}H^{-1}g \\
\\ \mathcal{L}(x, \lambda) &= g^T\left(- \frac{1}{\lambda}H^{-1}g\right) + \frac{\lambda}{2} \left(- \frac{1}{\lambda}H^{-1}g\right)^TH\left(- \frac{1}{\lambda}H^{-1}g\right) -\frac{\lambda}{2} \delta \\
\\ \nabla_\lambda \mathcal{L}(x,\lambda) &= 0 \hspace{130pt} \text{By KKT conditions} \\
\\ \nabla_\lambda \mathcal{L}(x,\lambda) &= \nabla_\lambda \left(- \frac{1}{\lambda}g^TH^{-1}g + \frac{1}{2\lambda}g^TH^{-1}g -\frac{\lambda}{2} \delta\right) \\
0 &= \nabla_\lambda\left(- \frac{1}{2\lambda}g^TH^{-1}g -\frac{\lambda}{2} \delta\right) \\
0 &= \frac{-1}{2} \left( -\frac{1}{\lambda^2}g^TH^{-1}g + \delta\right) \\
\frac{1}{2} g^TH^{-1}g &= \lambda^2 \delta \\
\lambda &= \sqrt\frac{g^TH^{-1}g}{2\delta}
\\ \implies x^* &= - \frac{1}{\lambda}H^{-1}g = - \sqrt\frac{2\delta}{g^TH^{-1}g}H^{-1}g
\\ \implies -(\theta - \theta_{old}) &= - \sqrt\frac{2\delta}{g^TH^{-1}g}H^{-1}g
\\ \theta &= \theta_{old} + \sqrt\frac{2\delta}{g^TH^{-1}g}H^{-1}g
\end{align}
$$

If there were no approximation errors, last equation above would give TRPO update. However, due to approximation errors, there is a chance that this update may violate constraints on KL divergence. Hence, in order to avoid this violation, backtracking line search is used.

$$\theta = \theta_{old} + \alpha^j\sqrt\frac{2\delta}{g^TH^{-1}g}H^{-1}g$$ where $\alpha \in (0, 1)$ is the backtracking co-efficient and $j$ is the smallest non-negative integer such that new policy does not violate KL divergence constraint. 


# Constrained Policy Optimization
Constrained Policy Optimization (CPO) includes cost function in addition to reward function. The average value of this cost function ought to be kept below some threshold value $d_i$ for $i$th cost function. 

Building on TRPO, CPO introduces three key modifications to allow use of trust region for constrained optimization.

1. Similar to how objective for reward function was linearized, CPO linearizes the cost function as well giving the following final form of objective function.
$$\begin{align}
\max_\theta \ &g^T(\theta - \theta_{old}) 
\\\text{s.t.} \hspace{5pt} & c_i + b_i^T(\theta - \theta_{old}) \leq 0 \hspace{50pt} \forall \ \ \ i=\{1,2,...,m\}
\\\text{s.t.} \hspace{5pt} &\frac{1}{2}(\theta - \theta_{old})^TH(\theta - \theta_{old}) < \delta 
\end{align}$$ where $c_i=J_{C_i}(\theta_{old}) - d_i$ i.e. remaining budget.
Following similar steps as in case of TRPO, we can derive the following solution to the proposed optimization program above.
$$\theta = \theta_{old} + \frac{1}{\lambda^*}H^{-1}(g - B\nu^*)$$ where $\lambda^*$ and $\nu^*$ are the solution to the dual of the above program and $B = [b_1, ..., b_m]$.
2. In case the above update produces an infeasible iterate due to approximation errors; CPO proposes to *purely* minimize cost function. Update for this is similar to TRPO but instead of gradient ascent, we perform gradient descent. This is termed *feasibility recovery*.
3. CPO uses cost shaping and constrains upper bound on cost. $$C_i^+(s,a,s') = C(s,a,s') + \Delta_t(s,a,s')$$ where $\Delta_t$ is the probability that policy will enter an unsafe state after $t$ time steps. A neural network is trained to approximate this function.


# Lagrangian Policy Gradient
In policy optimization step of Lagrangian Policy Gradient, we have the following objective where policy $\pi$ is parametrized by parameters $\theta$.

$$J(\theta) = \mathbb{E}_{\tau \sim \pi} [R(\tau)] - \nu (\mathbb{E}_{\tau \sim \pi}[C(\tau)] - \alpha)$$

$$
\begin{align}
\nabla_\theta J(\theta) &= \nabla_\theta \int_\tau P(\tau|\theta)R(\tau) d\tau - \nabla_\theta \nu \left (\int_\tau P(\tau|\theta) C(\tau) - \alpha\right) \\
&=  \int_\tau \nabla_\theta P(\tau|\theta)R(\tau) d\tau - \int_\tau \nabla_\theta P(\tau|\theta) \nu C(\tau) - \nabla_\theta \nu \alpha \\
&\stackrel{1}{=} \mathbb{E}_{\tau\sim \pi}[\sum_{t=0}^T \nabla_\theta \log \pi(a_t|s_t) (R(\tau) - \nu C(\tau))] \\
&\stackrel{2}{=} \mathbb{E}_{\tau\sim \pi}[\sum_{t=0}^T \nabla_\theta \log \pi(a_t|s_t) R(\tau)] - \nu \mathbb{E}_{\tau\sim \pi}[\sum_{t=0}^T \nabla_\theta \log \pi(a_t|s_t) C(\tau)] 
\end{align}
$$

(1) shows that you can use one critic to reduce variance.
(2) shows that you can separate critics to reduce variance. 

Can you use cost to go? Yes, any trick that you can use with $R(\tau)$ to reduce variance, you can use that with $C(\tau)$ as well.

### Update of $\nu$

$$
\begin{align}
\nabla_\nu J(\theta) &= - \mathbb{E}_{\tau \sim \pi}[C(\tau)] - \alpha
\end{align}
$$

Is it possible to use tricks here to reduce variance here as in policy optimization update? Unfortunately no, those tricks will bias the estimator here and are not useful. 

