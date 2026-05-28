from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SOFT_MUON_NS_COEFFICIENTS = (2.0, -1.5, 0.5)
DEFAULT_SOFT_MUON_P04_COEFFICIENTS = (
    0.427359225629,
    0.16510668279,
    0.0950365524083,
    0.0794622422344,
    0.0546397807059,
    0.0442774112372,
    0.0318743547215,
    0.0251008327807,
    0.0184953624306,
    0.014245458414,
    0.0137481403409,
    0.0,
)
DEFAULT_SOFT_MUON_P04_TAIL_COEFFICIENT = 0.0306539563075


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    model: str = "Qwen/Qwen3-0.6B"
    objective: str = "multirlvr"
    optimizer: str = "adamw"
    output_dir: str = "outputs/run"
    train_seed_start: int = 42
    train_size: int = 256
    eval_seed_start: int = 4242
    eval_size: int = 64
    maze_size: int = 9
    max_steps: int = 40
    global_train_batch_size: int | None = 64
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_generations: int = 8
    max_prompt_length: int = 1024
    max_completion_length: int = 256
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    muon_momentum: float = 0.95
    soft_muon_power: float = 0.4
    soft_muon_mix: float = 1.0
    soft_muon_ns_iterations: int = 12
    soft_muon_ns_coefficients: tuple[float, ...] = DEFAULT_SOFT_MUON_NS_COEFFICIENTS
    soft_muon_coefficients: tuple[float, ...] = DEFAULT_SOFT_MUON_P04_COEFFICIENTS
    soft_muon_tail_coefficient: float = DEFAULT_SOFT_MUON_P04_TAIL_COEFFICIENT
    beta: float = 1e-3
    epsilon: float = 0.2
    logging_steps: int = 1
    save_steps: int = 20
    use_vllm: bool = False
    bf16: bool = True
    report_to: str = "none"
    seed: int = 0
    vpo_weight_samples: int = 16
    diversity_eval_steps: int = 0
    diversity_eval_prompts: int = 16
    diversity_eval_samples_per_prompt: int = 1
    diversity_eval_temperature: float = 0.7
    diversity_eval_top_p: float = 1.0
    trace_rollout_examples: int = 1


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}
    return ExperimentConfig(**data)


def resolve_batch_config(cfg: ExperimentConfig, *, world_size: int) -> ExperimentConfig:
    """Keep GRPO's effective train batch stable across data-parallel world sizes."""
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    target = cfg.global_train_batch_size
    if target is None:
        return cfg
    if target < 1:
        raise ValueError(f"global_train_batch_size must be >= 1, got {target}")
    if target % world_size != 0:
        raise ValueError(
            "global_train_batch_size must be divisible by the number of processes: "
            f"{target} % {world_size} != 0"
        )
    if target % cfg.num_generations != 0:
        raise ValueError(
            "global_train_batch_size must be divisible by num_generations: "
            f"{target} % {cfg.num_generations} != 0"
        )

    per_process_batch = target // world_size
    max_per_device = min(cfg.per_device_train_batch_size, per_process_batch)
    divisors = [
        per_device
        for per_device in range(max_per_device, 0, -1)
        if per_process_batch % per_device == 0
    ]
    if not divisors:
        raise ValueError(
            f"Could not resolve per-device batch for per-process batch {per_process_batch}"
        )
    per_device_train_batch_size = divisors[0]
    gradient_accumulation_steps = per_process_batch // per_device_train_batch_size
    return replace(
        cfg,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
