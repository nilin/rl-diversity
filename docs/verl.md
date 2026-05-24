# veRL Path

The TRL path is the quick prototype. The veRL path is for a more tuned GRPO stack and closer comparability with the VPO paper, which reports using veRL.

## Data and Reward

Export Maze data in veRL's parquet format:

```bash
python -m diversity_muon.export_verl_maze \
  --output-dir data/maze_verl \
  --train-size 128 \
  --val-size 32
```

The custom reward function is [verl_reward.py](../src/diversity_muon/verl_reward.py). It reads `DIVERSITY_OBJECTIVE`:

- `multirlvr`
- `vpo`

## Minimum veRL Run

Requires veRL to be installed in the active environment.

```bash
./scripts/run_verl_minimum_comparison.sh
```

This launches:

- AdamW Multi-RLVR
- experimental Muon Multi-RLVR

Optional VPO:

```bash
RUN_VPO=1 ./scripts/run_verl_minimum_comparison.sh
```

## Muon Caveat

veRL's FSDP optimizer builder receives unnamed parameters. The experimental wrapper in [verl_optim.py](../src/diversity_muon/verl_optim.py) can split by tensor rank but cannot exclude embeddings or `lm_head` by name. The TRL path has a cleaner Muon parameter split.

For a serious veRL Muon run, patch veRL's optimizer construction to pass `named_parameters()` and reuse the same name-aware split as `diversity_muon.optim`.
