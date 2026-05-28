from __future__ import annotations

from diversity_muon.config import ExperimentConfig, resolve_batch_config


def test_resolve_batch_config_keeps_effective_batch_fixed() -> None:
    cfg = ExperimentConfig(
        run_name="test",
        global_train_batch_size=64,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=8,
    )

    one_gpu = resolve_batch_config(cfg, world_size=1)
    four_gpu = resolve_batch_config(cfg, world_size=4)
    eight_gpu = resolve_batch_config(cfg, world_size=8)

    assert one_gpu.per_device_train_batch_size == 4
    assert one_gpu.gradient_accumulation_steps == 16
    assert four_gpu.per_device_train_batch_size == 4
    assert four_gpu.gradient_accumulation_steps == 4
    assert eight_gpu.per_device_train_batch_size == 4
    assert eight_gpu.gradient_accumulation_steps == 2
    assert 1 * one_gpu.per_device_train_batch_size * one_gpu.gradient_accumulation_steps == 64
    assert 4 * four_gpu.per_device_train_batch_size * four_gpu.gradient_accumulation_steps == 64
    assert 8 * eight_gpu.per_device_train_batch_size * eight_gpu.gradient_accumulation_steps == 64


def test_resolve_batch_config_rejects_world_size_that_changes_batch() -> None:
    cfg = ExperimentConfig(run_name="test", global_train_batch_size=64, num_generations=8)

    try:
        resolve_batch_config(cfg, world_size=3)
    except ValueError as error:
        assert "divisible by the number of processes" in str(error)
    else:
        raise AssertionError("Expected resolve_batch_config to reject world_size=3")
