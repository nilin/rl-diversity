from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_generations: int = 8
    max_prompt_length: int = 1024
    max_completion_length: int = 256
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    muon_momentum: float = 0.95
    soft_muon_power: float = 0.2
    soft_muon_mix: float = 0.8
    beta: float = 1e-3
    epsilon: float = 0.2
    logging_steps: int = 1
    save_steps: int = 20
    use_vllm: bool = False
    bf16: bool = True
    report_to: str = "none"
    seed: int = 0
    vpo_weight_samples: int = 16


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle) or {}
    return ExperimentConfig(**data)
