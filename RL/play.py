#!/usr/bin/env python3

"""
python RL/play.py --model_path=swarm/submission_template/ppo_policy.zip --seed=2
python RL/play.py --model_path=checkpoints/ppo_swarm_2/ppo_swarm_1000000_steps.zip --seed=2
python RL/play.py --model_path=train_scripts/checkpoints/pretrain/bc_pretrain.zip --seed=2
"""

import argparse
import sys
from pathlib import Path

# Repo root on path before SB3 unpickles (checkpoints reference RL.* / customnetwork).
_RL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _RL_DIR.parent
for _p in (_REPO_ROOT, _RL_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from stable_baselines3 import PPO

from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task
from swarm.constants import SIM_DT, SPEED_LIMIT
import numpy as np
from gym_pybullet_drones.utils.enums import ActionType
import pybullet as p
import torch
import cv2
torch.set_num_threads(10)

show_rgb_camera = False


def resolve_model_zip(raw_path: str) -> Path:
    """Find a .zip checkpoint; SB3.load expects path *without* .zip suffix."""
    raw = Path(raw_path)
    search_roots = [
        Path.cwd(),
        _RL_DIR,
        _REPO_ROOT,
        _RL_DIR / "train_scripts",
    ]
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        for root in search_roots:
            candidates.append(root / raw)
    # Also try common typo pre_train -> pretrain
    extra = []
    for c in candidates:
        s = str(c).replace("pre_train", "pretrain")
        if s != str(c):
            extra.append(Path(s))
    candidates.extend(extra)

    seen: set[Path] = set()
    for c in candidates:
        c = c.resolve()
        if c in seen:
            continue
        seen.add(c)
        if c.is_file() and c.suffix.lower() == ".zip":
            return c
        alt = c.with_suffix(".zip")
        if alt.is_file():
            return alt
    tried = "\n".join(f"  - {p}" for p in sorted(seen))
    raise FileNotFoundError(
        f"Model checkpoint not found for '{raw_path}'.\n"
        f"Searched:\n{tried}\n"
        "Tip: use forward slashes and check folder name (pretrain vs pre_train)."
    )


def load_ppo_model(zip_path: Path) -> PPO:
    """Load PPO for inference only (no full training rollout buffer)."""
    # Policies trained with RL.framework use custom CNN; must be importable when unpickling.
    import customnetwork  # noqa: F401

    stem = zip_path.with_suffix("")
    print(f"Loading policy: {stem}.zip", flush=True)
    # Saved checkpoints use n_steps=2048, n_envs=8 → large rollout buffer on load.
    # Play only needs predict(); minimal buffer avoids ArrayMemoryError with GUI open.
    return PPO.load(
        str(stem),
        custom_objects={"n_steps": 1, "n_envs": 1},
        load_optimizer=False,
    )


def main(seed, model_path="swarm/submission_template/ppo_policy.zip"):
    task = random_task(sim_dt=SIM_DT, seed=seed)
    model_zip = resolve_model_zip(model_path)
    model = load_ppo_model(model_zip)

    env = make_env(task, gui=True)
    # BaseRLAviary may enable GUI sliders / RPM debug when gui=True; that ignores
    # the policy and looks like "drone never leaves start". Tests use the same off-switch.
    env.USER_DEBUG = False
    env.USE_GUI_RPM = False
    obs, _ = env.reset(seed=task.map_seed)
    t_sim = 0.0
    act_lo = env.action_space.low.flatten()
    act_hi = env.action_space.high.flatten()
    
    step_count = 0

    while t_sim < task.horizon:
        try:
            raw, _ = model.predict(obs, deterministic=True)
            if raw is None:
                raw = np.zeros(5, dtype=np.float32)
        except Exception as e:
            print("Prediction error:", e)
            raw = np.zeros(5, dtype=np.float32)

        act = np.clip(np.asarray(raw, dtype=np.float32).flatten(), act_lo, act_hi)
        if getattr(env, "ACT_TYPE", None) == ActionType.VEL:
            norm = max(float(np.linalg.norm(act[:3])), 1e-6)
            act[:3] *= min(1.0, float(SPEED_LIMIT) / norm)
            act = np.clip(act, act_lo, act_hi)

        obs, _, terminated, truncated, info = env.step(act[None, :])


        # ---- FPV CAMERA ----
        step_count += 1
        if step_count % 5 == 0:
            step_count = 0
            drone_state = env._getDroneStateVector(0)
            pos = drone_state[:3]
            quat = drone_state[3:7]

            rot = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
            forward = rot @ np.array([1, 0, 0])
            up = rot @ np.array([0, 0, 1])

            view = p.computeViewMatrix(pos, pos + forward * 2.0, up)
            proj = p.computeProjectionMatrixFOV(90, 1.0, 0.1, 50)

            _, _, rgb, dep, _ = p.getCameraImage(
                160, 120,
                viewMatrix=view,
                projectionMatrix=proj,
                renderer=p.ER_TINY_RENDERER  # 🔥 much faster
            )

            if show_rgb_camera:
                img = np.reshape(rgb, (120, 160, 4))[:, :, :3].astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img = dep

            cv2.imshow("FPV", img)
            cv2.waitKey(1)
        # ----- Camera end ---

        t_sim += float(SIM_DT)
        drone_pos = env._getDroneStateVector(0)[:3]

        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,        # zoom (smaller = closer)
            cameraYaw=45,              # horizontal angle
            cameraPitch=-30,           # vertical angle
            cameraTargetPosition=drone_pos
        )
        env.render()

        if terminated or truncated:
            success = bool(info.get("success", False))
            break
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--seed', default=3, type=int, help='Seed for random number generation', metavar='')
    parser.add_argument('--model_path', default="swarm/submission_template/ppo_policy.zip", type=str, help='Path to the trained model', metavar='')
    args = parser.parse_args()
    main(args.seed, model_path=args.model_path)
