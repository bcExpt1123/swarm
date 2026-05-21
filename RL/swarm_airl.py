"""AIRL variant compatible with Gymnasium Dict observation spaces (MovingDroneAviary)."""

from __future__ import annotations

import dataclasses
from typing import Iterator, Mapping, Optional

import numpy as np
import torch as th
import torch.nn.functional as F

from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.adversarial.common import compute_train_stats
from imitation.data import types


def _numpy_obs_to_torch_eval(obs, device: th.device):
    """Observation batch for SB3 ``evaluate_actions`` (ndarray or DictObs)."""
    if isinstance(obs, np.ndarray):
        return th.as_tensor(obs, device=device, dtype=th.float32)
    if isinstance(obs, types.DictObs):
        u = obs.unwrap()
        return {k: th.as_tensor(u[k], device=device, dtype=th.float32) for k in u}
    if isinstance(obs, dict):
        return {k: th.as_tensor(v, device=device, dtype=th.float32) for k, v in obs.items()}
    raise TypeError(f"Unsupported obs batch type: {type(obs)}")


def _concat_obs_batches(expert_obs, gen_obs):
    if isinstance(expert_obs, np.ndarray):
        return np.concatenate([expert_obs, gen_obs])
    return types.DictObs.concatenate([expert_obs, gen_obs])


class SwarmAIRL(AIRL):
    """AIRL with Dict-aware replay batches (see ``dict_observation_replay_buffer``)."""

    def train_disc(
        self,
        *,
        expert_samples: Optional[Mapping] = None,
        gen_samples: Optional[Mapping] = None,
    ) -> Mapping[str, float]:
        with self.logger.accumulate_means("disc"):
            write_summaries = self._init_tensorboard and self._global_step % 20 == 0

            self._disc_opt.zero_grad()

            batch_iter = self._make_disc_train_batches(
                gen_samples=gen_samples,
                expert_samples=expert_samples,
            )
            for batch in batch_iter:
                disc_logits = self.logits_expert_is_high(
                    batch["state"],
                    batch["action"],
                    batch["next_state"],
                    batch["done"],
                    batch["log_policy_act_prob"],
                )
                loss = F.binary_cross_entropy_with_logits(
                    disc_logits,
                    batch["labels_expert_is_one"].float(),
                )

                batch_rows = batch["action"].shape[0]
                assert batch_rows == 2 * self.demo_minibatch_size
                loss *= self.demo_minibatch_size / self.demo_batch_size
                loss.backward()

            self._disc_opt.step()
            self._disc_step += 1

            with th.no_grad():
                train_stats = compute_train_stats(
                    disc_logits,
                    batch["labels_expert_is_one"],
                    loss,
                )
            self.logger.record("global_step", self._global_step)
            for k, v in train_stats.items():
                self.logger.record(k, v)
            self.logger.dump(self._disc_step)
            if write_summaries:
                self._summary_writer.add_histogram("disc_logits", disc_logits.detach())

        return train_stats

    def _make_disc_train_batches(
        self,
        *,
        gen_samples: Optional[Mapping] = None,
        expert_samples: Optional[Mapping] = None,
    ) -> Iterator[Mapping[str, th.Tensor]]:
        batch_size = self.demo_batch_size

        if expert_samples is None:
            expert_samples = self._next_expert_batch()

        if gen_samples is None:
            if self._gen_replay_buffer.size() == 0:
                raise RuntimeError(
                    "No generator samples for training. Call `train_gen()` first.",
                )
            gen_samples_dataclass = self._gen_replay_buffer.sample(batch_size)
            gen_samples = types.dataclass_quick_asdict(gen_samples_dataclass)

        if not (len(gen_samples["obs"]) == len(expert_samples["obs"]) == batch_size):
            raise ValueError(
                "Need to have exactly `demo_batch_size` expert and generator samples. "
                f"(n_gen={len(gen_samples['obs'])} "
                f"n_expert={len(expert_samples['obs'])} "
                f"demo_batch_size={batch_size})",
            )

        expert_samples = dict(expert_samples)
        gen_samples = dict(gen_samples)

        for field in dataclasses.fields(types.Transitions):
            k = field.name
            if k == "infos":
                continue
            for d in (gen_samples, expert_samples):
                if isinstance(d[k], th.Tensor):
                    d[k] = d[k].detach().numpy()

        assert isinstance(gen_samples["obs"], (np.ndarray, types.DictObs))
        assert isinstance(expert_samples["obs"], (np.ndarray, types.DictObs))

        assert batch_size == len(expert_samples["acts"])
        assert batch_size == len(expert_samples["next_obs"])
        assert batch_size == len(gen_samples["acts"])
        assert batch_size == len(gen_samples["next_obs"])

        device = self.gen_algo.device

        for start in range(0, batch_size, self.demo_minibatch_size):
            end = start + self.demo_minibatch_size
            expert_batch = {k: v[start:end] for k, v in expert_samples.items()}
            gen_batch = {k: v[start:end] for k, v in gen_samples.items()}

            obs = _concat_obs_batches(expert_batch["obs"], gen_batch["obs"])
            acts = np.concatenate([expert_batch["acts"], gen_batch["acts"]])
            next_obs = _concat_obs_batches(expert_batch["next_obs"], gen_batch["next_obs"])
            dones = np.concatenate([expert_batch["dones"], gen_batch["dones"]])
            labels_expert_is_one = np.concatenate(
                [
                    np.ones(self.demo_minibatch_size, dtype=int),
                    np.zeros(self.demo_minibatch_size, dtype=int),
                ],
            )

            with th.no_grad():
                policy_obs_th = _numpy_obs_to_torch_eval(obs, device)
                acts_eval_th = th.as_tensor(acts, device=device, dtype=th.float32)
                # SB3's ``evaluate_actions`` expects actions flat-shaped
                # ``(batch, action_dim_flat)``; for multi-dim action spaces (e.g. the
                # Swarm env uses ``Box(shape=(NUM_DRONES, 5))``) ``log_prob`` would
                # otherwise leave a trailing dim and ``reshape((batch,))`` would fail.
                acts_eval_flat_th = acts_eval_th.reshape(acts_eval_th.shape[0], -1)
                log_policy_act_prob = self._get_log_policy_act_prob(
                    policy_obs_th, acts_eval_flat_th,
                )
                if log_policy_act_prob is not None:
                    assert len(log_policy_act_prob) == 2 * self.demo_minibatch_size
                    log_policy_act_prob = log_policy_act_prob.reshape(
                        (2 * self.demo_minibatch_size,),
                    )

            obs_th, acts_th, next_obs_th, dones_th = self.reward_train.preprocess(
                obs,
                acts,
                next_obs,
                dones,
            )
            batch_dict = {
                "state": obs_th,
                "action": acts_th,
                "next_state": next_obs_th,
                "done": dones_th,
                "labels_expert_is_one": self._torchify_array(labels_expert_is_one),
                "log_policy_act_prob": log_policy_act_prob,
            }

            yield batch_dict
