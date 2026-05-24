from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import torch
from torch import nn
from transformers import get_scheduler


OptimizerName = Literal["adamw", "muon"]


class CombinedOptimizer(torch.optim.Optimizer):
    """Thin wrapper for stepping Muon hidden matrices and AdamW auxiliary params together."""

    def __init__(self, optimizers: Iterable[torch.optim.Optimizer]):
        self.optimizers = list(optimizers)
        if not self.optimizers:
            raise ValueError("CombinedOptimizer requires at least one optimizer")
        params = [
            param
            for optimizer in self.optimizers
            for group in optimizer.param_groups
            for param in group["params"]
        ]
        super().__init__(params, {})
        self.param_groups = [group for opt in self.optimizers for group in opt.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):  # type: ignore[override]
        loss = None
        for optimizer in self.optimizers:
            maybe_loss = optimizer.step(closure=closure)
            if maybe_loss is not None:
                loss = maybe_loss
        return loss

    def state_dict(self):  # type: ignore[override]
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state_dict):  # type: ignore[override]
        for optimizer, child_state in zip(self.optimizers, state_dict["optimizers"], strict=True):
            optimizer.load_state_dict(child_state)


def build_optimizer(
    model: nn.Module,
    *,
    optimizer_name: OptimizerName,
    learning_rate: float,
    weight_decay: float,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    muon_momentum: float = 0.95,
) -> torch.optim.Optimizer:
    if optimizer_name == "adamw":
        decay, no_decay = split_weight_decay_params(model)
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
        )

    if optimizer_name == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError(
                "torch.optim.Muon is unavailable. Install PyTorch with Muon support or add a "
                "third-party Muon implementation."
            )
        muon_params, adam_decay, adam_no_decay = split_muon_aux_params(model)
        optimizers: list[torch.optim.Optimizer] = []
        if muon_params:
            optimizers.append(
                torch.optim.Muon(  # type: ignore[attr-defined]
                    muon_params,
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    momentum=muon_momentum,
                    nesterov=True,
                    adjust_lr_fn="match_rms_adamw",
                )
            )
        adam_groups = []
        if adam_decay:
            adam_groups.append({"params": adam_decay, "weight_decay": weight_decay})
        if adam_no_decay:
            adam_groups.append({"params": adam_no_decay, "weight_decay": 0.0})
        if adam_groups:
            optimizers.append(
                torch.optim.AdamW(adam_groups, lr=learning_rate, betas=(adam_beta1, adam_beta2))
            )
        return CombinedOptimizer(optimizers)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(optimizer: torch.optim.Optimizer, *, scheduler_type: str, num_training_steps: int):
    return get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )


def split_weight_decay_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or any(key in name.lower() for key in ("bias", "norm", "ln")):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def split_muon_aux_params(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
    muon_params: list[nn.Parameter] = []
    adam_decay: list[nn.Parameter] = []
    adam_no_decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lowered = name.lower()
        is_aux = (
            param.ndim != 2
            or "embed" in lowered
            or "lm_head" in lowered
            or "norm" in lowered
            or "ln" in lowered
            or "bias" in lowered
        )
        if not is_aux:
            muon_params.append(param)
        elif param.ndim >= 2 and "embed" not in lowered and "lm_head" not in lowered:
            adam_decay.append(param)
        else:
            adam_no_decay.append(param)
    return muon_params, adam_decay, adam_no_decay
