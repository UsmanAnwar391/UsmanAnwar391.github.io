---
layout: post
title: Notes On RL Papers
categories: [Research Papers]
mathjax: true
---

> This post contains notes on different RL papers that I have gone through. Please note that these notes are primarily meant for my understanding and I bear no responsibility of their correctness.

## [Time Limits In Reinforcement Learning](https://arxiv.org/abs/1712.00378)

### Metadata

Paper was published at ICML 2019. 

### Motivation

Practically, all infinite horizon and finite horizon tasks in RL are dealt with as 'fixed' time horizon tasks. What are the implications of this? 

### Paper Summary

Goal of agent in RL is to maximize discounted sum of future rewards. In case of finite horizon tasks, the following form is valid by assuming that $R_t = 0 \ \ \forall \ \ t>T$ where $T$ is the length of horizon.  
$$
G_t = R_{t+1} + \gamma R_{t+2}+... = \sum_{k=1}^{\infty} \gamma^{k-1}R_{t+k}
$$
However, often maximum length of episode is fixed (e.g. Atari games are only played till 128 steps maximum in PPO2 implementation of baseline) to a certain number for ease of training. In such cases, we can rewrite return in a computationally feasible form as below: 
$$
G_{t:T} = R_{t+1} + \gamma R_{t+2} + ...+\gamma^{T-t-1} R_{T} = \sum_{k=1}^{T-t} \gamma^{k-1}
$$
Now the authors note that task of the RL agent may be to either 

1. Maximize its reward over the fixed time period $[0,T]$ i.e. **time limited task.**

2. Maximize its reward over an indefinite time period $[0, \infty]$ i.e. **time unlimited task.**

Authors note that for time limited tasks, **Markov state must contain time index or in other words, stationary policy does not exist for time limited tasks.** Hence, the only solution is to learn policies that are dependent on time. So, time left, that is $T-t$ is provided to the agent as input after normalizing it in the range $[-1,1]$. Authors give two examples which elaborate on this point. 

- Further, they show that openAI gym environments (walker, hopper, reacher, swimmer etc.), though philosophically time unlimited, are actually time limited (Gym's TimeLimit wrapper is included by default in all environments). Here, they show that using time aware PPO helps achieve superior performance and even allows training with $\gamma = 1$. <u>Results of time aware PPO with $\gamma=1$ are quite impressive and much better than standard PPO.</u>

- The authors note that terminations due to time limit (i.e. in our Atari game example termination due to reacing 128 steps) is fundamentally different than environment termination. However, bcz they are treated alike this results in state aliasing (btw actual terminal state and states that become terminal due to time limit) and sub-optimal policies. 
  - They hence propose 'partial-bootstrapping' for episodes which are terminated artificially. That is if state is terminated artificially (that is environment has not provided 'done' signal but episode is deemed terminated bcz it reached max_time_steps limit), they propose bootstrapping return. That is temporal difference targets for value function should be 
    - ​	$r$ at environment termination. 
    - $r + \gamma V_{\pi}(s)$ for all other states, including artificial termination. 
  - They show that with this kind of bootstrapping, agents trained with small time limit are able to perform well for significantly larger time limits at evaluation time. For example, hopper trained with bootstrapped reward and time limit $T=200$ reaches time limit of $T = 10^6$ often at test time. 

### What to read?

See this [poster](https://fabiopardo.github.io/posters/time_limits_in_rl.pdf). Read discussion section and experiments (2.4 onwards).

