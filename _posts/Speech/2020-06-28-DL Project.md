---
layout: post
title: Imitation Learning On Atari Games Using GAIL
categories: [Projects]
mathjax: true
---

[[Code](..)] [[Report]()]

<h2>
    <center>Abstract</center>
</h2>
<p align="justify">
      <em>Specification of a reward function which aligns with the intentions of human users is a difficult task in reinforcement learning. To circumvent this issue, various methods have been proposed in the literature with the objective of implicitly inferring the reward function from the examples of expert behaviour. However, these methods often suffer from drawbacks such as lack of robustness, difficulty in optimization and high computational burden. Generative Adversarial Imitation Learning (GAIL) subverts these issues by posing the task of learning from demonstrations as an adversarial game between a generator policy network (which learns to imitate expert loss) and a discriminator network (which learns to differentiate between the samples from expert policy and generator policy and hence implicitly capture the reward function of expert). We use GAIL to learn to play two Atari games, Breakout and Pong. Our results are competitive with the state of the art. Further, we use gradient based class activation mapping to interpret the actions chosen by the policy network.</em>
  </p>
<h2>
    <center>Results</center>
</h2>
<h3>
    <center>Video Demos</center>
</h3>



<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/UgKxI8EfBf4" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/P1BICZbX25U" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

<h3>
<center>Average Reward Over Time</center></h3>

Breakout             |  Pong
:-------------------------:|:-------------------------:
![](/images/GAIL/breakout.png)  |  ![](/images/GAIL/pong.png)

<h2>
    <center>Interpretations Using Grad-CAM</center>
</h2>


![alt-text-1](/images/GAIL/map15.jpg  "title-1") ![alt-text-2](/images/GAIL/map15.jpg  "title-2")

![](/images/GAIL/map15.jpg)  |  ![](/images/GAIL/map16.jpg)  |  ![](/images/GAIL/map20.jpg)

