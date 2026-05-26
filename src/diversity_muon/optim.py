from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import torch
from torch import nn
from transformers import get_scheduler

OptimizerName = Literal["adamw", "muon", "soft_muon"]


class SoftMuon(torch.optim.Optimizer):
    """Muon variant using a soft singular-value map instead of hard orthogonalization."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        power: float = 0.2,
        mix: float = 0.8,
        eps: float = 1e-7,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not 0.0 <= power <= 1.0:
            raise ValueError(f"soft_muon_power must be in [0, 1], got {power}")
        if not 0.0 <= mix <= 1.0:
            raise ValueError(f"soft_muon_mix must be in [0, 1], got {mix}")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            power=power,
            mix=mix,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            power = group["power"]
            mix = group["mix"]
            eps = group["eps"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.ndim != 2:
                    raise RuntimeError("SoftMuon only supports 2D parameters")

                grad = param.grad
                state = self.state[param]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.lerp_(grad, 1 - momentum)
                update = grad.lerp(buf, momentum) if nesterov else buf

                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)
                update = _soft_muon_update(update, power=power, mix=mix, eps=eps)
                lr_adjustment = _match_rms_adamw_lr_adjustment(update.shape)
                param.add_(update.to(dtype=param.dtype), alpha=-lr * lr_adjustment)

        return loss


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
    soft_muon_power: float = 0.2,
    soft_muon_mix: float = 0.8,
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

    if optimizer_name in ("muon", "soft_muon"):
        if optimizer_name == "muon" and not hasattr(torch.optim, "Muon"):
            raise RuntimeError(
                "torch.optim.Muon is unavailable. Install PyTorch with Muon support or add a "
                "third-party Muon implementation."
            )
        muon_params, adam_decay, adam_no_decay = split_muon_aux_params(model)
        optimizers: list[torch.optim.Optimizer] = []
        if muon_params:
            if optimizer_name == "muon":
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
            else:
                optimizers.append(
                    SoftMuon(
                        muon_params,
                        lr=learning_rate,
                        weight_decay=weight_decay,
                        momentum=muon_momentum,
                        nesterov=True,
                        power=soft_muon_power,
                        mix=soft_muon_mix,
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


def build_scheduler(
    optimizer: torch.optim.Optimizer, *, scheduler_type: str, num_training_steps: int
):
    return get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )


def _soft_muon_update(
    update: torch.Tensor, *, power: float, mix: float, eps: float
) -> torch.Tensor:
    update_float = update.float()
    left, singular_values, right_h = torch.linalg.svd(update_float, full_matrices=False)
    sign_update = left @ right_h
    if power == 0.0:
        soft_update = sign_update
    else:
        soft_singular_values = singular_values.clamp_min(eps).pow(power)
        soft_update = (left * soft_singular_values.unsqueeze(0)) @ right_h
    mixed = torch.lerp(sign_update, soft_update, mix)

    target_rms = sign_update.square().mean().sqrt().clamp_min(eps)
    mixed_rms = mixed.square().mean().sqrt().clamp_min(eps)
    return mixed * (target_rms / mixed_rms)


def _match_rms_adamw_lr_adjustment(shape: torch.Size) -> float:
    fan_out, fan_in = shape
    return 0.2 * max(fan_out, fan_in) ** 0.5


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
