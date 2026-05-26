from __future__ import annotations

import torch

from diversity_muon.optim import SoftMuon, build_optimizer


def test_soft_muon_step_updates_2d_parameter() -> None:
    param = torch.nn.Parameter(torch.randn(4, 3))
    param.grad = torch.randn_like(param)
    before = param.detach().clone()

    optimizer = SoftMuon([param], lr=1e-3, weight_decay=0.01, power=0.2, mix=0.8)
    optimizer.step()

    assert not torch.equal(param.detach(), before)


def test_build_optimizer_accepts_soft_muon() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
    optimizer = build_optimizer(
        model,
        optimizer_name="soft_muon",
        learning_rate=1e-3,
        weight_decay=0.01,
    )

    assert optimizer.param_groups
