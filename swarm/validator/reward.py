# swarm/validator/reward.py
"""Reward function for flight missions.

``MovingDroneAviary`` calls ``flight_reward`` each control step with ``task=self.task``
and ``info`` from ``_computeInfo()`` (plus ``is_terminal`` when the step ends the episode).

**Per-step scalar components** (all in roughly :math:`[0,1]` except where noted):

* ``r_success``: ``1.0`` if the episode has reached the goal, else ``0.0``.
* ``r_time``: ``clip(1 - t / horizon, 0, 1)`` where ``t`` is mission elapsed time
  (time-to-goal if ``success``, else ``time_alive``).
* ``r_progress``: ``tanh((dist_{t-1} - dist_t) * (1 + d_{norm}))`` with
  ``d_{norm} = clip(||goal - pos|| / max_dist, 0, 1)`` and ``max_dist`` from
  ``||goal - start||`` when ``task`` is set (else ``1.0``).
* ``r_speed``: ``1 - clip(|speed - desired| / SPEED_LIMIT, 0, 1)`` with
  ``desired = SPEED_LIMIT * max(d_norm, 0.2)`` and ``speed`` from consecutive
  trajectory positions in ``info['state'][:3]``.
* ``r_direction``: cosine alignment between step velocity and ``goal - pos``,
  mapped to ``[0,1]`` (neutral ``0.5`` when vectors are tiny).
* ``r_overshoot``: ``0`` if the drone crossed from “before” to “past” the goal
  along the start–goal segment in one step, else ``1`` (then ``r_overshoot`` term uses ``1 - overshoot``).
* ``r_smooth``: ``1 - clip(||Δv|| / SPEED_LIMIT, 0, 1)`` from three consecutive positions.
* ``r_safety``: ``1.0`` if ``min_clearance`` is omitted; else piecewise clearance vs
  danger/safe thresholds (and ``0`` if ``collision``).
* ``r_landing``: near-goal vertical / horizontal speed score when ``d < 0.5`` and
  full kinematic ``state`` is present; else ``0``.

**Combined step reward**

.. code-block:: text

    step_reward =
        0.50 * r_success
      + 0.20 * r_time
      + 0.15 * r_progress
      + 0.10 * r_speed
      + 0.10 * r_direction
      + 0.10 * r_overshoot
      + 0.05 * r_smooth
      + 0.05 * r_safety
      + 0.15 * r_landing

**Terminal handling** (when ``info['is_terminal']`` is true):

* If ``collision`` or not ``success``: return ``0.01`` (legitimate model and ``t>0``)
  or ``0.0``.
* If successful terminal: ``terminal_bonus = 1.0`` and
  ``return step_reward + terminal_bonus``.

Helper :func:`_calculate_target_time` is kept for benchmarks/tests; it is **not**
used inside :func:`flight_reward` today.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from swarm.protocol import MapTask

from swarm.constants import (
    HOVER_SEC,
    REWARD_W_SAFETY,
    REWARD_W_SUCCESS,
    REWARD_W_TIME,
    SAFETY_DISTANCE_DANGER,
    SAFETY_DISTANCE_SAFE,
    SPEED_LIMIT,
    TYPE_6_SAFETY_DISTANCE_SAFE,
    LANDING_MAX_VZ,
    LANDING_MAX_VXY_REL,
    LANDING_MAX_TILT_RAD,
    LANDING_STABLE_SEC
)

SAFETY_DISTANCE_SAFE_BY_TYPE = {
    6: TYPE_6_SAFETY_DISTANCE_SAFE,
}

__all__ = ["flight_reward"]


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp *value* to the inclusive range [*lower*, *upper*]."""
    return max(lower, min(upper, value))


def _calculate_target_time(task: "MapTask") -> float:
    """Calculate target time based on distance and 6% buffer."""
    start_pos = np.array(task.start)
    goal_pos = np.array(task.goal)
    distance = np.linalg.norm(goal_pos - start_pos)

    min_time = (distance / SPEED_LIMIT) + HOVER_SEC
    return min_time * 1.06


def _calculate_safety_term(
    min_clearance: float, collision: bool, challenge_type: int = 0
) -> float:
    """Calculate safety term based on minimum obstacle clearance."""
    if collision:
        return 0.0
    safe = SAFETY_DISTANCE_SAFE_BY_TYPE.get(challenge_type, SAFETY_DISTANCE_SAFE)
    if min_clearance >= safe:
        return 1.0
    if min_clearance <= SAFETY_DISTANCE_DANGER:
        return 0.0
    return (min_clearance - SAFETY_DISTANCE_DANGER) / (safe - SAFETY_DISTANCE_DANGER)

def calc_r_distance_3d(xyz, p1, p2):
    """
    xyz : (x, y, z) current point
    p1  : (x1, y1, z1) start point
    p2  : (x2, y2, z2) goal point
    """

    x, y, z = xyz
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    start = np.array([x1, y1, z1], dtype=float)
    goal  = np.array([x2, y2, z2], dtype=float)
    cur   = np.array([x, y, z], dtype=float)

    d = np.linalg.norm(goal - start)
    if d == 0:
        return 0.0

    d1 = np.linalg.norm(cur - start)
    d2 = np.linalg.norm(cur - goal)

    if d < d2:
        return 0.0

    m1 = 1 - (d1 / d)
    m2 = 1 - (np.sqrt(d2)) / d

    m = 1 + m1 + m2

    goal_dist = np.linalg.norm(goal - cur)
    p = m * np.abs(1 - goal_dist / d)

    # ---- 3D point-line distance using cross product ----
    # line_vec = goal - start
    # point_vec = cur - start

    # cross = np.linalg.norm(np.cross(line_vec, point_vec))
    # c = 1 - 0.7 * ((cross / d) ** (4 / 5))

    f = p #* c - 0.1

    return max(0.0, f)


trajectory: list[np.ndarray] = []
map_task = None  # type: Optional[MapTask]
reward_history = {
    "progress": [],
    "distance": [],
    "r_distance": [],
    "r_direction": [],
    "safety": [],
    "landing": [],
    "reward": [],
    "reward_sum": []
}
should_render_graph = False

def flight_reward_complex(
    success: bool,
    t: float,
    horizon: float,
    task: Optional["MapTask"] = None,
    *,
    min_clearance: Optional[float] = None,
    collision: bool = False,
    w_success: float = REWARD_W_SUCCESS,
    w_t: float = REWARD_W_TIME,
    w_safety: float = REWARD_W_SAFETY,
    legitimate_model: bool = True,
    info
) -> float:
    global trajectory
    global reward_history
    global map_task
    global should_render_graph
    
    max_vel = 3 / 50
    state = info.get("state", None)

    if should_render_graph:
        if task is not None:
            map_task = task
    if horizon <= 0:
        raise ValueError("'horizon' must be positive")
    if info is None:
        info = {}

    drone_pos = np.array(info["state"][:3], dtype=np.float64)

    if should_render_graph:
        trajectory.append(drone_pos)
    
    
    d = info.get("distance_to_goal", 0.0)
    prev_d = info.get("prev_distance_to_goal", d)

    r_progress = np.tanh(5 * (prev_d - d))
    if should_render_graph:
        reward_history["progress"].append(r_progress)
        reward_history["distance"].append(d)
    
    r_goal_zone = 0
    if d < 1.0:
        r_goal_zone = 0.5
    else:
        r_goal_zone = 0.0
    r_away = -0.3 * max(0, d - prev_d)
    # r_distance = 5 * np.exp(-d / 2.0)
    r_distance = 0.0
    if map_task is not None:
        r_distance = calc_r_distance_3d(drone_pos, map_task.start, map_task.goal)
    else:
        r_distance = 0.0

    if should_render_graph:
        reward_history["r_distance"].append(r_distance)

    r_height = 0
    r_slow = 0
    if map_task is not None:
        z = drone_pos[2]
        goal_z = map_task.goal[2]
        height_error = z - goal_z
        vel = np.zeros(3)
        if state is not None:
            vel = state[10:13]
        if d < 1.0:
            r_height = height_error
            r_slow = -np.abs(vel[2])
        else:
            r_height = -height_error
            r_slow = 0



    # r_time = -0.02
    r_time = -0.1

    if collision:
        # r_safety = -2.0
        r_safety = -1.0
    elif min_clearance is not None:
        safe_dist = SAFETY_DISTANCE_SAFE_BY_TYPE.get(
            getattr(task, "challenge_type", 0) if task else 0,
            SAFETY_DISTANCE_SAFE,
        )
        ratio = min_clearance / safe_dist
        r_safety = 0.3 * np.clip(ratio, 0, 1) - 0.3 * (1 - ratio)**2
    else:
        r_safety = 0.0

    if should_render_graph:
        reward_history["safety"].append(r_safety)
    # --- 5. Success (dominant signal) ---
    r_success = 5.0 if success else 0.0

    # --- 6. Landing (bounded & balanced) ---
    max_speed_deduction = 0
    min_speed_deduction = 0
    if state is not None:
        vel = state[10:13]
        v_norm = np.linalg.norm(vel) + 1e-6
        if v_norm > 3 / 50:
            max_speed_deduction = -1.0
        if v_norm < 0.5 / 50:
            min_speed_deduction = -0.2

    r_direction = 0.0
    if state is not None and map_task is not None:
        vel = state[10:13]
        # r_direction = cosine similarity between vel and (goal - pos), mapped to [-1,1]
        to_goal = np.array(map_task.goal) - drone_pos
        to_goal_norm = np.linalg.norm(to_goal) + 1e-6
        vel_norm = np.linalg.norm(vel) + 1e-6
        cos_sim = np.dot(vel, to_goal) / (vel_norm * to_goal_norm)
        r_direction = cos_sim# 0.5 * (cos_sim + 1)

    if should_render_graph:
        reward_history["r_direction"].append(r_direction)

    r_landing = 0.0

    landing_stable_time = info.get("landing_stable_time", 0.0)
    if state is not None and d < 0.5:
        vel = state[10:13]
        roll, pitch = state[7], state[8]
        vz = abs(vel[2])
        drone_vxy = vel[0:2]
        platform_vxy = info.get("platform_velocity")[0:2]
        rel_vxy = np.linalg.norm(drone_vxy - platform_vxy)
        v_score = 1.0 - np.clip(vz / LANDING_MAX_VZ, 0, 1)
        xy_score = 1.0 - np.clip(rel_vxy / LANDING_MAX_VXY_REL, 0, 1)
        tilt_score = 1.0 - np.clip(
            max(abs(roll), abs(pitch)) / LANDING_MAX_TILT_RAD, 0, 1
        )

        quality = (0.4 * v_score + 0.3 * xy_score + 0.3 * tilt_score)

        stability = np.clip(
            landing_stable_time / LANDING_STABLE_SEC, 0, 1
        )

        r_landing = 1.5 * quality + 1.5 * stability

        velocity_ok = vz <= LANDING_MAX_VZ and rel_vxy <= LANDING_MAX_VXY_REL
        upright_ok = abs(roll) <= LANDING_MAX_TILT_RAD and abs(pitch) <= LANDING_MAX_TILT_RAD

        if not (velocity_ok and upright_ok):
            r_landing -= 1.0

    if should_render_graph:
        reward_history["landing"].append(r_landing)
    # --- Combine all ---
    # reward = (
    #     r_progress
    #     + r_away
    #     + r_distance
    #     + r_direction
    #     + r_time
    #     + r_safety
    #     + max_speed_deduction
    #     + min_speed_deduction
    #     + r_success
    #     + r_landing
    # )
    reward = (
        r_progress +
        r_time +
        r_safety +
        r_goal_zone +
        r_height +
        r_slow +
        r_success
    )

    if should_render_graph:
        reward_history["reward"].append(reward)
        # reward_history["reward_sum"].append(sum(reward_history["reward"]))
        # reward_history["reward_sum"].append(0)

        if t == 0:
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import axes3d

                plt.figure(1)
                plt.clf()

                ax1 = plt.subplot(2, 1, 1, projection='3d')

                if len(trajectory) > 1:
                    xs = [p[0] for p in trajectory]
                    ys = [p[1] for p in trajectory]
                    zs = [p[2] for p in trajectory]

                    ax1.plot(xs, ys, zs, "o-", linewidth=1, markersize=3)
                if map_task is not None:
                    start_xyz = np.array(map_task.start, dtype=np.float64)
                    goal_xyz = np.array(map_task.goal, dtype=np.float64)

                    ax1.scatter(start_xyz[0], start_xyz[1], start_xyz[2], c="green", s=120, marker="o", label="Start")
                    ax1.scatter(goal_xyz[0], goal_xyz[1], goal_xyz[2], c="red", s=120, marker="x", label="Goal")

                    ax1.legend()
                ax1.set_title("Drone Trajectory (XYZ)")
                ax1.set_xlabel("X")
                ax1.set_ylabel("Y")
                ax1.set_zlabel("Z")

                plt.subplot(2, 1, 2)

                plt.plot(reward_history["progress"], label="progress", linewidth=2.5, color="tab:blue")
                plt.plot(reward_history["r_distance"], label="r_distance", linewidth=2.5, color="tab:green")
                plt.plot(reward_history["r_direction"], label="r_direction", linewidth=2.5, color="tab:orange")
                plt.plot(reward_history["reward"], label="reward", linewidth=2, color="tab:red")

                plt.title("Reward Components")
                plt.legend()

                plt.tight_layout()
                plt.pause(0.1)
            except Exception:
                pass

            try:
                # write total reward to file for logging
                with open("reward_log.txt", "a") as f:
                    f.write(f"{sum(reward_history['reward'])}\n")
                pass
            except Exception:
                pass

            trajectory.clear()
            for k in reward_history:
                reward_history[k].clear()
        # print(f"{reward}")
        
    return float(reward)

def flight_reward_simple(
    success: bool,
    t: float,
    horizon: float,
    task: Optional["MapTask"] = None,
    *,
    min_clearance: Optional[float] = None,
    collision: bool = False,
    w_success: float = REWARD_W_SUCCESS,
    w_t: float = REWARD_W_TIME,
    w_safety: float = REWARD_W_SAFETY,
    legitimate_model: bool = True,
    info
):
    d = info["distance_to_goal"]
    prev_d = info.get("prev_distance_to_goal", d)
    # Continuous progress reward
    r_progress = (prev_d - d) * 2.0          # positive for moving toward goal
    # Small time penalty
    r_time = -0.01
    # Safety penalty (only when too close)
    r_safety = 0.0
    if collision:
        r_safety = -1.0
    elif min_clearance is not None and min_clearance < 0.5:
        r_safety = -0.5
    # Sparse success reward
    r_success = 10.0 if success else 0.0
    # Optional: alive bonus to encourage exploration
    r_alive = 0.01 if not success and not collision else 0.0

    state = info.get("state", None)
    max_speed_deduction = 0
    min_speed_deduction = 0
    # if state is not None:
    #     vel = state[10:13]
    #     v_norm = np.linalg.norm(vel) + 1e-6
    #     if v_norm > 3 / 50:
    #         max_speed_deduction = -1.0
    #     if v_norm < 0.5 / 50:
    #         min_speed_deduction = -0.2

    reward = r_progress + r_time + r_safety + r_success + r_alive + max_speed_deduction + min_speed_deduction
    # Clip to reasonable range
    return np.clip(reward, -1.0, 10.0)

def flight_reward_normalized(
    success: bool,
    t: float,
    horizon: float,
    task=None,
    *,
    min_clearance=None,
    collision: bool = False,
    info=None,
):
    d = info["distance_to_goal"]
    prev_d = info.get("prev_distance_to_goal", d)

    # --- Progress (normalized) ---
    progress = (prev_d - d)
    r_progress = np.clip(progress * 5.0, -1.0, 1.0)

    # --- Time penalty ---
    r_time = -0.01  # keep small

    # --- Safety ---
    if collision:
        r_safety = -1.0
    elif min_clearance is not None and min_clearance < 0.5:
        r_safety = -0.3
    else:
        r_safety = 0.0

    # --- Success (normalized, NOT huge spike) ---
    r_success = 1.0 if success else 0.0

    # --- Alive bonus ---
    r_alive = 0.01 if not success and not collision else 0.0

    reward = r_progress + r_time + r_safety + r_success + r_alive

    # Final normalization safeguard
    return float(np.clip(reward, -1.0, 1.0))

def flight_reward(
    success: bool,
    t: float,
    horizon: float,
    task: Optional["MapTask"] = None,
    *,
    min_clearance: Optional[float] = None,
    collision: bool = False,
    w_success: float = REWARD_W_SUCCESS,
    w_t: float = REWARD_W_TIME,
    w_safety: float = REWARD_W_SAFETY,
    legitimate_model: bool = True,
) -> float:
    """Compute the reward for a single flight mission.

    Parameters
    ----------
    success
        ``True`` if the mission successfully reached its objective.
    t
        Time (in seconds) taken to reach the goal.
    horizon
        Maximum time allowed to complete the mission.
    task
        MapTask object containing start and goal positions for distance calculation.
    min_clearance
        Minimum distance (meters) to any obstacle during flight. If None, safety
        term is set to 1.0 (full score).
    collision
        ``True`` if the drone collided with an obstacle. Forces safety term to 0.
    w_success, w_t, w_safety
        Weights for success, time, and safety terms. They should sum to ``1``.
    legitimate_model
        ``True`` if the model passed verification. Legitimate models that fail
        missions receive a base reward of 0.01.

    Returns
    -------
    float
        A score in the range ``[0, 1]``.
    """

    if horizon <= 0:
        raise ValueError("'horizon' must be positive")

    if collision:
        if legitimate_model and t > 0.0:
            return 0.01
        return 0.0

    success_term = 1.0 if success else 0.0

    if success_term == 0.0:
        if legitimate_model and t > 0.0:
            return 0.01
        return 0.0

    if task is not None:
        target_time = _calculate_target_time(task)

        if t <= target_time:
            time_term = 1.0
        elif horizon <= target_time:
            time_term = 0.0
        else:
            time_term = _clamp(1.0 - (t - target_time) / (horizon - target_time))
    else:
        time_term = _clamp(1.0 - t / horizon)

    challenge_type = getattr(task, "challenge_type", 0) if task is not None else 0
    if min_clearance is not None:
        safety_term = _calculate_safety_term(min_clearance, collision, challenge_type)
    else:
        safety_term = 1.0 if not collision else 0.0

    score = (w_success * success_term) + (w_t * time_term) + (w_safety * safety_term)
    return _clamp(score)
