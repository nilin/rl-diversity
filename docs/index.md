---
layout: default
title: RL Diversity Soft-Muon
---

# RL Diversity Soft-Muon

Best@k comparison for 7x7 Maze RLVR runs at learning rate 3e-6, excluding VPO and k=1.

[View the version including AdamW VPO at LR=3e-6](assets/best-at-k-lr3e-6-with-vpo.png).

![Best@k curve comparing AdamW Multi-RLVR and Soft-Muon p=0.4 at learning rate 3e-6.](assets/best-at-k-lr3e-6.png)

_Mean best@k over route pools for AdamW Multi-RLVR and fixed-coefficient Soft-Muon p=0.4._

![Cumulative p=0.4 approximation built from stacked Newton-Schulz iterates.](assets/soft-muon-p04-cumulative-fit.png)

_Fixed p=0.4 Soft-Muon approximation built as a cumulative stack of Newton-Schulz iterates, staying below the target power curve._
