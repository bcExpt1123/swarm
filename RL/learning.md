## 🔥 Training

### 1. Pretrain

```bash
python RL/pretrain_ppo.py --collect_episodes 48 --output ./model/bc_pretrain_48_1e8/pretrain_model_48_1e8.zip
```
### 2. Extract reward funcion

```bash
python RL/airl_train.py --pretrain_path model/bc_pretrain_48/pretrain_model_48.zip --reward_dir model/airl/reward_48 --policy_dir model/airl/policy_48
```
### 3. Evaluate reward function

```bash
python RL/ppo_train.py --policy-type custom --resume model/bc_pretrain/pretrain_model_240.zip --timesteps 2000000 --no-norm-obs --no-norm-reward

python RL/ppo_train.py --policy-type custom --resume model/bc_pretrain/pretrain_model_240.zip --no-norm-obs --no-norm-reward --timesteps 5000000 --learning-rate 3e-4 --lr-end 1e-5 --ent-coef 0.003 --n-epochs 5 --clip-range 0.1 --max-grad-norm 0.3

python RL/ppo_train.py --policy-type custom --resume model/bc_pretrain/pretrain_model_240.zip --no-norm-obs --no-norm-reward --timesteps 2500000 --n_steps 512 --n-envs 8 --learning-rate 1.5e-4 --lr-end 1e-5 --seed 2 --checkpoint-freq 500000 --tensorboard ./ppo_logs/
```
- Simple:
```bash
python RL/pretrain_ppo.py --seed 2 --collect_episodes 40 --max_steps_per_episode 2048 --train_step 80 --bc_lr 2e-4 --batch_size 512 --learning_rate 3e-4 --n_steps 2048 --ent_coef 0.01 --max_grad_norm 0.3 --output ./model/bc_pretrain_simple/simple_drone.zip

python RL/airl_train.py --pretrain_path model/bc_pretrain_simple/simple_drone.zip --reward_dir model/airl/reward_48_simple --policy_dir model/airl/policy_48_simple --n_envs 1 --min_expert_episodes 10 --rounds 20 --steps_per_round 8000 --demo_batch_size 256 --gen_replay_buffer_capacity 1024 --gen_train_timesteps 300 --n_disc_updates_per_round 2 --disc_lr 1e-4 --gen_lr 1e-4

python RL/ppo_train.py --policy-type custom --resume model/bc_pretrain_simple/simple_drone.zip --no-norm-obs --no-norm-reward --checkpoint-freq 500000 --tensorboard ./ppo_logs/ --timesteps 300000 --n-envs 4 --learning-rate 3e-4 --ent-coef 0.001 --gamma 0.95 --n_steps 1024 --batch_size 512 --n-epochs 5 --clip-range 0.2 --vf_coef 0.5 --max-grad-norm 0.5 --checkpoint-dir ./checkpoints/ppo_swarm_simple --output-file ppo_policy_simple
```

- Medium:
```bash
python RL/pretrain_ppo.py --seed 2 --collect_episodes 100 --max_steps_per_episode 3072 --train_step 150 --bc_lr 1e-4 --batch_size 1024 --learning_rate 3e-4 --n_steps 4096 --ent_coef 0.005 --max_grad_norm 0.3 --output ./model/bc_pretrain_medium/medium_drone.zip

python RL/airl_train.py --pretrain_path model/bc_pretrain_medium/medium_drone.zip --reward_dir model/airl/reward_48_medium --policy_dir model/airl/policy_48_medium --n_envs 1 --min_expert_episodes 24 --rounds 35 --steps_per_round 12000 --demo_batch_size 512 --gen_replay_buffer_capacity 2048 --gen_train_timesteps 500 --n_disc_updates_per_round 3 --disc_lr 5e-5 --gen_lr 5e-5

python RL/ppo_train.py --policy-type custom --resume model/bc_pretrain_medium/medium_drone.zip --tensorboard ./ppo_logs/ --checkpoint-dir ./checkpoints/ppo_swarm_medium --output-file ppo_policy_medium --timesteps 1500000 --n-envs 6 --learning-rate 3e-4 --lr-end 5e-5 --ent-coef 0.002 --gamma 0.97 --n_steps 2048 --batch_size 1024 --n-epochs 8 --clip-range 0.15 --vf_coef 0.6 --max-grad-norm 0.5 --checkpoint-freq 250000
```

- Hard:
```bash
python RL/pretrain_ppo.py --seed 2 --collect_episodes 150 --max_steps_per_episode 4096 --train_step 200 --bc_lr 1e-4 --batch_size 1024 --n_steps 4096 --learning_rate 3e-4 --n_epochs 10 --ent_coef 0.005 --max_grad_norm 0.3 --clip_range 0.2 --clip_range_vf 0.1 --output ./model/bc_pretrain_hard/hard_drone.zip

python RL/airl_train.py --pretrain_path model/bc_pretrain_hard/hard_drone.zip --reward_dir model/airl/reward_48_hard --policy_dir model/airl/policy_48_hard --n_envs 1 --min_expert_episodes 50 --rounds 50 --steps_per_round 15000 --demo_batch_size 512 --gen_replay_buffer_capacity 4096 --gen_train_timesteps 800 --n_disc_updates_per_round 4 --disc_lr 3e-5 --gen_lr 3e-5 --log_std_boost 0.0

python RL/ppo_train.py --policy-type custom --resume model/bc_pretrain_hard/hard_drone.zip --tensorboard ./ppo_logs/ --checkpoint-dir ./checkpoints/ppo_swarm_hard --output-file ppo_policy_hard --timesteps 5000000 --n-envs 8 --learning-rate 2e-4 --lr-end 3e-5 --ent-coef 0.003 --gamma 0.99 --n_steps 2048 --batch_size 1024 --n-epochs 10 --clip-range 0.1 --vf_coef 0.8 --max-grad-norm 0.3 --checkpoint-freq 500000

python RL/ppo_train.py --policy-type custom --resume model/bc_pretrain_hard/hard_drone.zip --tensorboard ./ppo_logs/ --checkpoint-dir ./checkpoints/ppo_swarm_hard_1 --output-file ppo_policy_hard_1 --timesteps 5000000 --n-envs 8 --learning-rate 2e-4 --lr-end 3e-5 --ent-coef 0.003 --gamma 0.99 --n_steps 2048 --batch_size 1024 --n-epochs 10 --clip-range 0.1 --vf_coef 0.8 --max-grad-norm 0.3 --checkpoint-freq 500000
```


- Real Hard:
```bash
python RL/pretrain_ppo.py --seed 42 --collect_episodes 250 --max_steps_per_episode 4096 --train_step 300 --bc_lr 1e-4 --batch_size 1024 --learning_rate 3e-4 --n_steps 4096 --ent_coef 0.01 --max_grad_norm 0.3 --output ./model/bc_pretrain_xhard/xhard_drone.zip

python RL/airl_train.py --pretrain_path model/bc_pretrain_xhard/xhard_drone.zip --reward_dir model/airl/reward_48_xhard --policy_dir model/airl/policy_48_xhard --n_envs 1 --min_expert_episodes 100 --rounds 60 --steps_per_round 20000 --demo_batch_size 1024 --gen_replay_buffer_capacity 8192 --gen_train_timesteps 1000 --n_disc_updates_per_round 5 --disc_lr 1e-5 --gen_lr 1e-5 --log_std_boost 0.0

python RL/ppo_train.py --policy-type custom --resume model/bc_pretrain_xhard/xhard_drone.zip --tensorboard ./ppo_logs/ --checkpoint-dir ./checkpoints/ppo_swarm_xhard --output-file ppo_policy_xhard --timesteps 15000000 --n-envs 8 --learning-rate 1e-4 --lr-end 1e-5 --ent-coef 0.005 --gamma 0.995 --n_steps 4096 --batch_size 2048 --n-epochs 5 --clip-range 0.08 --vf_coef 1.0 --max-grad-norm 0.2 --checkpoint-freq 1000000
```
