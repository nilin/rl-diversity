---
layout: default
title: RL Diversity with Soft-Muon
---

# RL Diversity with Soft-Muon

2026/5/30

## Muon diversifies input-output relationships

The 2D weights `W` in a neural network or transformer represent linear mappings from input activations to outputs. Let `G` be the estimated gradient of the loss with respect to such a weight matrix. The SVD `sum_i s_i output_i input_i^T` of a matrix `G` decomposes it into weighted input-output relationships 

`s_i output_i input_i^T`

where `s_i>0` is the corresponding singular value. [Muon](https://kellerjordan.github.io/posts/muon/) approximates the transformation of the gradient `G` where all the `s_i` are scaled to the same value. We can think of this as increasing the diversity of the learning signal instead of letting it be dominated by a few relationships `output_i input_i^T` with large singular value.

## Diversity collapse in RL

In RL we only get one reward signal per reponse (or a few in the case of VPO). Therefore the estimated policy gradient may not have enough information to estimate 1000s of input-output relationships, so the Muon update becomes dominated by noise. On the other hand collapse of diversity and exploration is a well-known issue that can arise in RL finetuning of LLMs, and methods such as [vector policy optimization (VPO)](https://arxiv.org/abs/2605.22817) seek to ameliorate this. 


## Soft Muon

We explore whether [Soft-Muon](https://nilin.github.io/contra-muon-and-soft-muon/) provides an alternative way to encourage diversity and exploration during RL, while being more robust to noisy policy gradients than full Muon.

![Cumulative p=0.4 approximation built from stacked Newton-Schulz iterates.](assets/soft-muon-p04-cumulative-fit.png)

_Fixed p=0.4 Soft-Muon approximation built as a cumulative stack of standard Muon's Newton-Schulz iterates._

## Experiment

We test Soft-Muon for RL in a setting from the [VPO paper](https://arxiv.org/abs/2605.22817). We use a 7x7 version of the Maze benchmark described in the paper. We compare AdamW GRPO baseline vs Muon GRPO vs AdamW VPO. Like the VPO paper we train on multi-answer chains: one model completion contains m=3 candidate answers (routes). We use multi-answer chains for all methods (AdamW baseline, Muon, and AdamW VPO). For the AdamW and Muon GRPO methods, each rollout is scored by the best route among the 3 following the VPO paper (probably not ideal for these methods for late training). VPO additionally receives a vector reward r(x, y) representing a decomposition of the reward into [completion, gold, diamond, avoid_lava], whereas the other methods get the averaged vector reward.

We run 50 steps and swept learning rates 1e-6, 3e-6, 1e-5 and picked 3e-6 based on mean best-route score@30 for the AdamW GRPO baseline.

Please see the full experiment settings at [this repo](https://github.com/nilin/rl-diversity).

The plot below reports mean best-route score@k for 7x7 Maze RLVR runs at learning rate 3e-6, excluding k=1. This is not pass@k: each route gets a continuous scalar reward from `[completion, gold, diamond, avoid_lava]`, then the evaluator takes the best route score in each k-route pool and averages those best scores.

![Mean best-route score@k curve comparing AdamW Multi-RLVR, Soft-Muon p=0.4, and VPO at learning rate 3e-6.](assets/best-at-k-lr3e-6-with-vpo.png)

<!---
[View the version including AdamW VPO at LR=3e-6](assets/best-at-k-lr3e-6-with-vpo.png).
![Mean best-route score@k curve comparing AdamW Multi-RLVR and Soft-Muon p=0.4 at learning rate 3e-6.](assets/best-at-k-lr3e-6.png)
--->

_Mean best-route score@k over route pools for AdamW Multi-RLVR, AdamW VPO, and fixed-coefficient Soft-Muon p=0.4._
