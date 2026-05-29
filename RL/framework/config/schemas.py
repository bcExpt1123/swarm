"""Dataclass-based configuration schemas for drone RL training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class RewardWeights:
    goal_progress: float = 1.0
    collision: float = -5.0
    obstacle_proximity: float = -0.5
    hover_stability: float = 0.3
    smooth_action: float = -0.05
    angular_velocity: float = -0.02
    alive: float = 0.01
    energy: float = -0.02
    use_env_reward: bool = True
    env_reward_scale: float = 1.0


@dataclass
class CurriculumStage:
    name: str
    challenge_types: list[int] = field(default_factory=lambda: [1])
    horizon_scale: float = 1.0
    min_success_rate: float = 0.25
    min_episodes: int = 50
    eval_episodes: int = 20
    moving_platform_prob: float = 0.0
    seed_offset: int = 0


@dataclass
class CurriculumConfig:
    enabled: bool = True
    stages: list[CurriculumStage] = field(default_factory=list)
    auto_advance: bool = True
    randomize_env_params: bool = True


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    lr_end: Optional[float] = None
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = 0.1
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_steps: int = 4096
    batch_size: int = 512
    n_epochs: int = 10
    normalize_advantage: bool = True


@dataclass
class PolicyConfig:
    policy_type: str = "custom"  # custom | multiinput | mlp | cnn | recurrent
    features_dim: int = 96
    pi_layers: list[int] = field(default_factory=lambda: [128, 64, 32, 16])
    vf_layers: list[int] = field(default_factory=lambda: [128, 64, 32, 16])
    share_features_extractor: bool = False
    log_std_init: float = -1.8
    lstm_hidden_size: int = 256
    n_lstm_layers: int = 1


@dataclass
class VecEnvConfig:
    n_envs: int = 8
    gui: bool = False
    norm_obs: bool = True
    norm_reward: bool = True
    clip_obs: float = 10.0
    subproc: bool = True


@dataclass
class LoggingConfig:
    """SB3 writes event files only; it does NOT start the TensorBoard web server."""
    tensorboard_dir: str = "./ppo_logs/"
    tensorboard_write_events: bool = True
    wandb_project: str = "swarm-drone-rl"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_enabled: bool = False
    log_interval: int = 1


@dataclass
class CheckpointConfig:
    dir: str = "./checkpoints/ppo_swarm/"
    freq: int = 500_000
    keep_last_n: int = 5
    submission_output: str = "ppo_policy"


@dataclass
class EvaluationConfig:
    n_episodes: int = 50
    deterministic: bool = True
    eval_freq: int = 50_000
    metrics_weights: dict[str, float] = field(
        default_factory=lambda: {
            "success_rate": 0.35,
            "collision_rate": -0.25,
            "smoothness": 0.15,
            "mean_reward": 0.15,
            "path_efficiency": 0.10,
        }
    )


@dataclass
class AIRLConfig:
    expert_path: str = "expert/expert_data.pkl"
    pretrain_path: str = "model/bc_pretrain/pretrain_model.zip"
    min_expert_episodes: int = 24
    rounds: int = 35
    steps_per_round: int = 12_000
    demo_batch_size: int = 512
    gen_replay_buffer_capacity: int = 2048
    gen_train_timesteps: int = 500
    n_disc_updates_per_round: int = 3
    disc_lr: float = 5e-5
    gen_lr: float = 5e-5
    log_std_boost: float = 0.0
    reward_dir: str = "model/airl/reward"
    policy_dir: str = "model/airl/policy"


@dataclass
class BCConfig:
    collect_episodes: int = 100
    max_steps_per_episode: int = 3072
    train_steps: int = 150
    bc_lr: float = 1e-4
    batch_size: int = 1024


@dataclass
class OptunaConfig:
    n_trials: int = 30
    n_timesteps_per_trial: int = 200_000
    n_eval_episodes: int = 15
    study_name: str = "swarm_ppo_hpo"
    storage: Optional[str] = None
    sampler_seed: int = 0


@dataclass
class TrainingConfig:
    """Root configuration loaded from YAML."""

    seed: int = 2
    total_timesteps: int = 1_000_000
    device: str = "auto"
    resume_path: Optional[str] = None
    reward: RewardWeights = field(default_factory=RewardWeights)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    vec_env: VecEnvConfig = field(default_factory=VecEnvConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    airl: AIRLConfig = field(default_factory=AIRLConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def submission_dir(self) -> Path:
        return Path(__file__).resolve().parents[3] / "swarm" / "submission_template"
