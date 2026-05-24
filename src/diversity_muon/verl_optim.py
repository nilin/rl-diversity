from __future__ import annotations

from collections.abc import Iterable

import torch


class MuonNdMatWithAdamW(torch.optim.Optimizer):
    """veRL-compatible Muon wrapper.

    veRL's FSDP optimizer builder passes an unnamed parameter iterable. That means
    this wrapper can split by tensor rank but cannot exclude embeddings/lm_head by
    name like the TRL path does. Treat this as an experimental veRL Muon path.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-6,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        momentum: float = 0.95,
        **_: object,
    ):
        params = list(params)
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("torch.optim.Muon is unavailable in this PyTorch build.")
        muon_params = [param for param in params if param.requires_grad and param.ndim == 2]
        adam_params = [param for param in params if param.requires_grad and param.ndim != 2]
        optimizers: list[torch.optim.Optimizer] = []
        if muon_params:
            optimizers.append(
                torch.optim.Muon(  # type: ignore[attr-defined]
                    muon_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    nesterov=True,
                    adjust_lr_fn="match_rms_adamw",
                )
            )
        if adam_params:
            optimizers.append(
                torch.optim.AdamW(adam_params, lr=lr, weight_decay=0.0, betas=betas)
            )
        if not optimizers:
            raise ValueError("No trainable parameters were provided.")
        self.optimizers = optimizers
        super().__init__(params, {})
        self.param_groups = [group for optimizer in optimizers for group in optimizer.param_groups]

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
