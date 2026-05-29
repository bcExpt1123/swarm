# Training configs

All original defaults are unchanged. Level-specific configs mirror `RL/learning.md`.

## Default configs (unchanged)

| File | Use |
|------|-----|
| `pretrain_default.yaml` | Generic pretrain / starting point |
| `finetune_default.yaml` | Generic fine-tune |
| `airl_default.yaml` | Generic AIRL |
| `rewards_shaping.yaml` | Optional reward weights reference |
| `optuna_default.yaml` | Hyperparameter search |

## Level configs (simple → real hard)

| Level | Pretrain | AIRL | Fine-tune |
|-------|----------|------|-----------|
| **Simple** | `pretrain_simple.yaml` | `airl_simple.yaml` | `finetune_simple.yaml` |
| **Medium** | `pretrain_medium.yaml` | `airl_medium.yaml` | `finetune_medium.yaml` |
| **Hard** | `pretrain_hard.yaml` | `airl_hard.yaml` | `finetune_hard.yaml` |
| **Real hard** | `pretrain_real_hard.yaml` | `airl_real_hard.yaml` | `finetune_real_hard.yaml` |

Checkpoints and logs are isolated per level under `RL/train_scripts/checkpoints/<level>/` and `logs/<level>/`.

## Pipeline per level

From repo root:

```bash
# 1) BC + PPO warm-start
python -u RL/train_scripts/train_pretrain.py --config RL/train_scripts/configs/pretrain_<level>.yaml

# 2) AIRL (optional but recommended for medium+)
# Expert .pkl uses pickle (Dict obs); imitation.serialize.save is not supported.
python -u RL/train_scripts/collect_expert_data.py --config RL/train_scripts/configs/airl_<level>.yaml
python -u RL/train_scripts/train_airl.py --config RL/train_scripts/configs/airl_<level>.yaml

# 3) Fine-tune for validator score (update resume_path to latest AIRL policy if needed)
python -u RL/train_scripts/train_finetune.py --config RL/train_scripts/configs/finetune_<level>.yaml

# 4) Evaluate
python RL/train_scripts/evaluate.py --model swarm/submission_template/ppo_policy.zip --config RL/train_scripts/configs/finetune_<level>.yaml
```

Replace `<level>` with: `simple`, `medium`, `hard`, `real_hard`.

## TensorBoard (AIRL)

AIRL logs live under `logs/<level>/airl/summary/` (see YAML `logging.tensorboard_dir`).

```bash
# From repository root:
python RL/train_scripts/launch_tensorboard.py --logdir RL/train_scripts/logs/medium/airl
# equivalent:
tensorboard --logdir RL/train_scripts/logs/medium/airl/summary
```

## Resume paths

Fine-tune YAML `resume_path` points to the **last AIRL round** by convention (`ppo_policy_N.zip`). After AIRL, set `--resume` to your latest file:

```bash
python -u RL/train_scripts/train_finetune.py \
  --config RL/train_scripts/configs/finetune_hard.yaml \
  --resume RL/train_scripts/checkpoints/hard/airl/policy/ppo_policy_50.zip
```

## Timesteps summary

| Level | Pretrain PPO | Fine-tune PPO | AIRL rounds |
|-------|--------------|---------------|-------------|
| Simple | 300k | 300k | 20 |
| Medium | 800k | 1.5M | 35 |
| Hard | 1.5M | 5M | 50 |
| Real hard | 2M | 15M | 60 |
