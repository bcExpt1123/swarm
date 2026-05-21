import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from imitation.util import networks as im_networks
from imitation.util.networks import BaseNorm, RunningNorm
from stable_baselines3.common.torch_layers import CombinedExtractor, BaseFeaturesExtractor
from imitation.rewards.reward_nets import BasicShapedRewardNet
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Type, cast
from imitation.rewards.reward_nets import ShapedRewardNet, BasicPotentialMLP, BasicRewardNet


def obs_split(combined):
    batch_size = combined.shape[0]
    img = combined[:, :, :, :128]
    vel = combined[:, :, :, 128:]
    vel = vel.reshape(batch_size, 1, 72, 3)
    vel = vel[:, 0, 0, :]
    return img, vel


class feature_extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, normalize_velocity_layer=None, normalize_image_layer=None,
                 features_dim=96):
        super().__init__(observation_space, features_dim=features_dim)
        self.normalize_velocity_layer = normalize_velocity_layer
        self.normalize_image_layer = normalize_image_layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True)
        )
        self.conv1[0].weight.data.normal_(0, 0.5)  # initialization

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2[0].weight.data.normal_(0, 0.5)  # initialization

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True)
        )
        self.conv3[0].weight.data.normal_(0, 0.5)  # initialization
        with torch.no_grad():
            _dummy = torch.zeros(1, 1, 128, 128)
            _x = self.conv3(self.conv2(self.conv1(_dummy)))
            _flat_dim = int(torch.flatten(_x, start_dim=1).shape[1])
        self.image_fc1 = nn.Linear(_flat_dim, 64, bias=False)
        self.image_fc2 = nn.Linear(64, 48, bias=False)
        self.image_fc1.weight.data.normal_(0, 0.5)
        self.image_fc2.weight.data.normal_(0, 0.5)

        self.vel_fc1 = nn.Linear(3, 48, bias=False)
        self.vel_fc1.weight.data.normal_(0, 0.5)
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, obs):
        if isinstance(obs, dict):
            depth = obs["depth"]
            state = obs["state"]
            device = self.conv1[0].weight.device
            if not isinstance(depth, torch.Tensor):
                depth = torch.as_tensor(depth, dtype=torch.float32, device=device)
            else:
                depth = depth.float().to(device)
            if not isinstance(state, torch.Tensor):
                state = torch.as_tensor(state, dtype=torch.float32, device=device)
            else:
                state = state.float().to(device)
            if depth.dim() == 3:
                depth = depth.unsqueeze(0)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            # MovingDroneAviary: depth (N, H, W, 1) -> (N, 1, H, W); vel from state[10:13]
            if depth.shape[-1] == 1:
                image = depth.permute(0, 3, 1, 2).contiguous()
            elif depth.shape[1] == 1:
                image = depth
            else:
                raise ValueError(f"Unexpected depth shape {tuple(depth.shape)}")
            vel = state[:, 10:13]
        else:
            image, vel = obs_split(obs)

        x = self.conv1(image)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.image_fc1(x)
        x = self.relu(x)
        x = self.image_fc2(x)

        vel = self.vel_fc1(vel)

        state_process = torch.cat((x, vel), dim=1)
        state_process = self.tanh(state_process)

        return state_process


class CustomShapedRewardNet(ShapedRewardNet):
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            *,
            normalize: Optional[Type[RunningNorm]] = None,
            reward_hid_sizes: Sequence[int] = (64, 32),
            potential_hid_sizes: Sequence[int] = (64, 32),
            use_state: bool = True,
            use_action: bool = True,
            use_next_state: bool = False,
            use_done: bool = False,
            if_share_feature_extractor: bool = False,
            discount_factor: float = 0.99,
            feature_extractor: Type[BaseFeaturesExtractor],
            feature_output_dim: int,
            **kwargs,
    ):
        # 保存传入参数为实例属性
        self.use_state = use_state
        self.use_action = use_action
        self.use_next_state = use_next_state
        self.use_done = use_done
        self.discount_factor = discount_factor
        self.if_share_feature_extractor = if_share_feature_extractor
        self.normalize = normalize

        # Initialize the base reward network
        base_reward_net = BasicRewardNet(
            observation_space=observation_space,
            action_space=action_space,
            use_state=use_state,
            use_action=use_action,
            use_next_state=use_next_state,
            use_done=use_done,
            hid_sizes=reward_hid_sizes,
            **kwargs,
        )

        # Potential is applied to state features *after* feature_transform (same dim as base net).
        potential_net = BasicPotentialMLP(
            observation_space=gym.spaces.Box(
                low=-float("inf"), high=float("inf"), shape=(reward_hid_sizes[0],)
            ),
            hid_sizes=potential_hid_sizes,
            **kwargs,
        )

        super().__init__(
            base=base_reward_net,
            potential=potential_net,
            discount_factor=discount_factor,
        )

        # Feature extractor initialization

        self.feature_extractor = feature_extractor(observation_space, **kwargs)
        if not self.if_share_feature_extractor:
            self.feature_extractor_potential = feature_extractor(observation_space, **kwargs)
        # Linear transform to match reward network input
        self.feature_transform = nn.Linear(
            feature_output_dim, reward_hid_sizes[0]  # Match input size of the first layer
        )
        _adim = int(np.prod(np.array(action_space.shape, dtype=np.int64)))
        self.action_transform = nn.Linear(
            _adim, reward_hid_sizes[0]
        )
        # Normalize layer

        self.mlp = nn.Sequential(
            nn.Linear(reward_hid_sizes[0]*2, 64),
            nn.ReLU(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.potential = nn.Sequential(
            nn.Linear(reward_hid_sizes[0], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def preprocess(self, state, action, next_state, done):
        """Convert numpy / DictObs batches to tensors (Dict spaces skip SB3 flatten)."""
        from imitation.data.types import DictObs
        import imitation.util.util as util

        device = self.device

        def unwrap(o):
            if isinstance(o, DictObs):
                return o.unwrap()
            return o

        state = unwrap(state)
        next_state = unwrap(next_state)
        if isinstance(state, dict):
            state_th = {k: util.safe_to_tensor(state[k]).to(device).float() for k in state}
            next_state_th = {
                k: util.safe_to_tensor(next_state[k]).to(device).float() for k in next_state
            }
            action_th = util.safe_to_tensor(action).to(device).float()
            done_th = util.safe_to_tensor(done).to(device).float()
            n = len(action_th)
            assert len(done_th) == n
            assert all(len(state_th[k]) == n for k in state_th)
            return state_th, action_th, next_state_th, done_th
        return super().preprocess(state, action, next_state, done)

    @staticmethod
    def _obs_batch_size(state) -> int:
        """Best-effort batch size for ndarray, dict, or ``DictObs`` observation batches."""
        if hasattr(state, "shape"):
            return int(state.shape[0])
        if isinstance(state, dict):
            if not state:
                raise ValueError("Empty dict observation batch.")
            first = next(iter(state.values()))
            return int(first.shape[0])
        try:
            from imitation.data.types import DictObs

            if isinstance(state, DictObs):
                return int(len(state))
        except ImportError:
            pass
        return int(len(state))

    def predict_th(self, state, action, next_state, done):
        """Dict-aware variant of ``imitation.rewards.reward_nets.RewardNet.predict_th``.

        The upstream implementation asserts ``rew_th.shape == state.shape[:1]``, which
        fails when ``state`` is a ``dict`` / ``DictObs`` (no ``.shape`` attribute).
        """
        with im_networks.evaluating(self):
            state_th, action_th, next_state_th, done_th = self.preprocess(
                state, action, next_state, done
            )
            with torch.no_grad():
                rew_th = self(state_th, action_th, next_state_th, done_th)

            batch_size = self._obs_batch_size(state)
            assert rew_th.shape == (batch_size,), (
                f"reward shape {tuple(rew_th.shape)} != expected ({batch_size},)"
            )
            return rew_th

    def forward(self, obs, action, next_obs, done, **kwargs):
        # share feature extractor will be realized in the future
        if not self.if_share_feature_extractor:
            obs_feature = self.feature_extractor(obs)
            obs_feature = self.feature_transform(obs_feature)
            if self.use_next_state:
                next_obs_feature = self.feature_extractor_potential(next_obs)
                next_obs_feature = self.feature_transform(next_obs_feature)
            action = self.action_transform(action)
            inputs = []
            if self.use_state:
                inputs.append(torch.flatten(obs_feature, 1))
            if self.use_action:
                inputs.append(torch.flatten(action, 1))
            combined_inputs = torch.cat(inputs, dim=-1)
            reward = self.mlp(combined_inputs)
            if self.use_next_state:
                new_shaping_output = self.potential(next_obs_feature).flatten()
                old_shaping_output = self.potential(obs_feature).flatten()
                reward = reward.squeeze(-1)
                old_shaping_output = old_shaping_output.squeeze(-1)
                new_shaping_output = new_shaping_output.squeeze(-1)
                new_shapping = (1-done.float()) * new_shaping_output
                final_reward = (reward + self.discount_factor * new_shapping - old_shaping_output)
            else:
                final_reward = reward.squeeze(-1)

        _b = int(obs_feature.shape[0])
        assert final_reward.shape[0] == _b, (final_reward.shape, _b)
        return final_reward





