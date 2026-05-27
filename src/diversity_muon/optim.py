from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

import torch
from torch import nn
from transformers import get_scheduler

OptimizerName = Literal["adamw", "muon", "soft_muon"]
DEFAULT_SOFT_MUON_NS_COEFFICIENTS = (2.0, -1.5, 0.5)
DEFAULT_SOFT_MUON_P05_COEFFICIENTS = (
    0.6077251254,
    0.04001601401,
    0.02944381009,
    0.02329789693,
    0.01912933357,
    0.01638279099,
    0.01449631237,
    0.01302904148,
    0.01167778429,
    0.01020522013,
    0.008369069534,
    0.006016417332,
)
DEFAULT_SOFT_MUON_P05_TAIL_COEFFICIENT = 0.2002111839


class SoftMuon(torch.optim.Optimizer):
    """Muon variant using stacked Newton-Schulz iterates for a soft singular-value map."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        power: float = 0.5,
        mix: float = 1.0,
        ns_iterations: int = 12,
        ns_coefficients: Sequence[float] = DEFAULT_SOFT_MUON_NS_COEFFICIENTS,
        soft_coefficients: Sequence[float] = DEFAULT_SOFT_MUON_P05_COEFFICIENTS,
        soft_tail_coefficient: float = DEFAULT_SOFT_MUON_P05_TAIL_COEFFICIENT,
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
        if ns_iterations < 1:
            raise ValueError(f"soft_muon_ns_iterations must be >= 1, got {ns_iterations}")
        ns_coefficients = tuple(float(value) for value in ns_coefficients)
        if len(ns_coefficients) != 3:
            raise ValueError("soft_muon_ns_coefficients must contain exactly 3 values")
        soft_coefficients = tuple(float(value) for value in soft_coefficients)
        if len(soft_coefficients) != ns_iterations:
            raise ValueError(
                "soft_muon_coefficients length must match soft_muon_ns_iterations: "
                f"{len(soft_coefficients)} != {ns_iterations}"
            )
        if any(value < 0.0 for value in soft_coefficients) or soft_tail_coefficient < 0.0:
            raise ValueError("soft_muon coefficients must be nonnegative")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            power=power,
            mix=mix,
            ns_iterations=ns_iterations,
            ns_coefficients=ns_coefficients,
            soft_coefficients=soft_coefficients,
            soft_tail_coefficient=float(soft_tail_coefficient),
            eps=eps,
        )
        super().__init__(params, defaults)
        self._logged_soft_muon_step = False

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
            ns_iterations = group["ns_iterations"]
            ns_coefficients = group["ns_coefficients"]
            soft_coefficients = group["soft_coefficients"]
            soft_tail_coefficient = group["soft_tail_coefficient"]
            eps = group["eps"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.ndim != 2:
                    raise RuntimeError("SoftMuon only supports 2D parameters")
                if not self._logged_soft_muon_step:
                    print(
                        "SoftMuon.step: using custom Newton-Schulz SoftMuon update path "
                        f"power={power} mix={mix} ns_iterations={ns_iterations} "
                        f"ns_coefficients={ns_coefficients} "
                        f"soft_coefficients={soft_coefficients} "
                        f"soft_tail_coefficient={soft_tail_coefficient}",
                        flush=True,
                    )
                    self._logged_soft_muon_step = True

                grad = param.grad
                state = self.state[param]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.lerp_(grad, 1 - momentum)
                update = grad.lerp(buf, momentum) if nesterov else buf

                if weight_decay != 0:
                    param.mul_(1 - lr * weight_decay)
                update = _soft_muon_update(
                    update,
                    power=power,
                    mix=mix,
                    ns_iterations=ns_iterations,
                    ns_coefficients=ns_coefficients,
                    soft_coefficients=soft_coefficients,
                    soft_tail_coefficient=soft_tail_coefficient,
                    eps=eps,
                )
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
    soft_muon_power: float = 0.5,
    soft_muon_mix: float = 1.0,
    soft_muon_ns_iterations: int = 12,
    soft_muon_ns_coefficients: Sequence[float] = DEFAULT_SOFT_MUON_NS_COEFFICIENTS,
    soft_muon_coefficients: Sequence[float] = DEFAULT_SOFT_MUON_P05_COEFFICIENTS,
    soft_muon_tail_coefficient: float = DEFAULT_SOFT_MUON_P05_TAIL_COEFFICIENT,
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
                print(
                    "build_optimizer: constructing custom SoftMuon optimizer "
                    f"muon_param_count={len(muon_params)} lr={learning_rate} "
                    f"weight_decay={weight_decay} momentum={muon_momentum} "
                    f"power={soft_muon_power} mix={soft_muon_mix} "
                    f"ns_iterations={soft_muon_ns_iterations} "
                    f"ns_coefficients={tuple(soft_muon_ns_coefficients)} "
                    f"soft_coefficients={tuple(soft_muon_coefficients)} "
                    f"soft_tail_coefficient={soft_muon_tail_coefficient}",
                    flush=True,
                )
                optimizers.append(
                    SoftMuon(
                        muon_params,
                        lr=learning_rate,
                        weight_decay=weight_decay,
                        momentum=muon_momentum,
                        nesterov=True,
                        power=soft_muon_power,
                        mix=soft_muon_mix,
                        ns_iterations=soft_muon_ns_iterations,
                        ns_coefficients=soft_muon_ns_coefficients,
                        soft_coefficients=soft_muon_coefficients,
                        soft_tail_coefficient=soft_muon_tail_coefficient,
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
    update: torch.Tensor,
    *,
    power: float,
    mix: float,
    ns_iterations: int,
    ns_coefficients: Sequence[float],
    soft_coefficients: Sequence[float],
    soft_tail_coefficient: float,
    eps: float,
) -> torch.Tensor:
    del power  # The configured coefficients define the approximation to x**power.
    sign_update = _zeropower_via_newtonschulz(
        update, ns_iterations=ns_iterations, ns_coefficients=ns_coefficients, eps=eps
    )
    soft_update = _softpower_via_newtonschulz(
        update,
        ns_iterations=ns_iterations,
        ns_coefficients=ns_coefficients,
        soft_coefficients=soft_coefficients,
        soft_tail_coefficient=soft_tail_coefficient,
        eps=eps,
    )

    target_norm = _gram_frobenius_norm_estimate(sign_update, eps=eps)
    soft_norm = _gram_frobenius_norm_estimate(soft_update, eps=eps)
    soft_update = soft_update * (target_norm / soft_norm).to(soft_update.dtype)
    mixed = torch.lerp(sign_update, soft_update, mix)
    mixed_norm = _gram_frobenius_norm_estimate(mixed, eps=eps)
    return mixed * (target_norm / mixed_norm).to(mixed.dtype)


def _gram_frobenius_norm_estimate(
    update: torch.Tensor, *, keepdim: bool = False, eps: float = 1e-10
) -> torch.Tensor:
    update_float = update.float()
    gram = (
        update_float.mT @ update_float
        if update_float.size(-2) > update_float.size(-1)
        else update_float @ update_float.mT
    )
    return gram.norm(dim=(-2, -1), keepdim=keepdim).sqrt().clamp_min(eps)


def _newton_schulz_input(
    update: torch.Tensor, *, eps: float
) -> tuple[torch.Tensor, bool]:
    transposed = update.size(-2) > update.size(-1)
    dtype = torch.bfloat16 if update.device.type == "cuda" else torch.float32
    ns_update = update.to(dtype=dtype)
    if transposed:
        ns_update = ns_update.mT
    ns_update = ns_update / _gram_frobenius_norm_estimate(
        ns_update, keepdim=True, eps=eps
    ).to(ns_update.dtype)
    return ns_update, transposed


def _newton_schulz_step(
    update: torch.Tensor, ns_coefficients: Sequence[float]
) -> torch.Tensor:
    a, b, c = ns_coefficients
    gram = update @ update.mT
    basis = b * gram + c * gram @ gram
    return a * update + basis @ update


def _zeropower_via_newtonschulz(
    update: torch.Tensor,
    *,
    ns_iterations: int,
    ns_coefficients: Sequence[float],
    eps: float,
) -> torch.Tensor:
    ns_update, transposed = _newton_schulz_input(update, eps=eps)
    for _ in range(ns_iterations):
        ns_update = _newton_schulz_step(ns_update, ns_coefficients)
    if transposed:
        ns_update = ns_update.mT
    return ns_update


def _softpower_via_newtonschulz(
    update: torch.Tensor,
    *,
    ns_iterations: int,
    ns_coefficients: Sequence[float],
    soft_coefficients: Sequence[float],
    soft_tail_coefficient: float,
    eps: float,
) -> torch.Tensor:
    ns_update, transposed = _newton_schulz_input(update, eps=eps)
    basis = [ns_update]
    for _ in range(ns_iterations):
        ns_update = _newton_schulz_step(ns_update, ns_coefficients)
        basis.append(ns_update)

    out = soft_tail_coefficient * basis[-1]
    for coefficient, basis_term in zip(soft_coefficients, basis[:-1], strict=True):
        out = out + coefficient * basis_term

    if transposed:
        out = out.mT
    return out


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
