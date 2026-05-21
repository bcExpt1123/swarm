import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task
from swarm.constants import SIM_DT

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('one_d_rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 2
DEFAULT_MA = False

class StepDataCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.next_obs = []

    def _on_step(self) -> bool:
        # Access rollout buffer data
        print("Step data collected")
        print(self.locals)  # Print available keys in locals
        """
        {
        "self": "",
        "total_timesteps": 4000,
        "callback": "",
        "log_interval": 100,
        "tb_log_name": "PPO",
        "reset_num_timesteps": "",
        "progress_bar": "",
        "iteration": 0,
        "env": "",
        "rollout_buffer": "",
        "n_rollout_steps": 2048,
        "n_steps": 0,
        "obs_tensor": {
            "depth": [
            [
                [
                [
                    1
                ],
                [
                    1
                ],
                [
                    0
                ],
                [
                    0
                ],
                [
                    0
                ]
                ]
            ]
            ],
            "state": [
            [
                -5.0000e+00,
                -5.0000e+00,
                4.1082e-01,
                0.0000e+00,
                -0.0000e+00,
                0.0000e+00,
                0.0000e+00,
                0.0000e+00,
                0.0000e+00,
                0.0000e+00,
                0.0000e+00,
                0.0000e+00,
                3.5250e-03,
                -5.0000e+00,
                -2.3701e-01,
                7.7719e-01
            ]
            ]
        },
        "actions": [
            [
            [
                -0.31737226,
                0.9370976,
                -1.1215647,
                -0.0904863,
                1.1007768
            ]
            ]
        ],
        "values": [
            [
            0.4215
            ]
        ],
        "log_probs": [
            -6.3188
        ],
        "clipped_actions": [
            [
            [
                -0.31737226,
                0.9370976,
                -1,
                -0.0904863,
                1
            ]
            ]
        ],
        "new_obs": [
            "depth",
            [
            [
                [
                [
                    1
                ],
                [
                    1
                ],
                [
                    1
                ],
                [
                    0
                ],
                [
                    0
                ]
                ]
            ]
            ],
            "state",
            [
            [
                -5.0000000e+00,
                -5.0000000e+00,
                4.0940827e-01,
                -9.3261767e-03,
                -9.2201931e-03,
                1.1429078e-02,
                0.0000000e+00,
                0.0000000e+00,
                -7.0434377e-02,
                -4.6366602e-01,
                -4.6366602e-01,
                5.6929600e-01,
                0.0000000e+00,
                0.0000000e+00,
                0.0000000e+00,
                0.0000000e+00,
                -3.1737226e-01,
                9.3709761e-01,
                -1.0000000e+00,
                -9.0486296e-02,
                1.0000000e+00,
                3.4545655e-03,
                -5.0000000e+00,
                -2.3701246e-01,
                7.7860206e-01
            ]
            ]
        ],
        "rewards": [
            225.66478
        ],
        "dones": "",
        "infos": [
            {
            "distance_to_goal": 6.630830321570377,
            "prev_distance_to_goal": 6.630830321570377,
            "score": 0.27999999999999997,
            "success": "False",
            "collision": "False",
            "t_to_goal": "None",
            "min_clearance": 1.0,
            "landing_stable_time": 0.0,
            "state": [
                -6.39942691e+01,
                -6.45213429e+00,
                4.09408257e-01,
                -4.63660194e-03,
                -4.63660194e-03,
                5.69288830e-03,
                9.99962297e-01,
                -9.32617717e-03,
                -9.22019358e-03,
                1.14290781e-02,
                0.00000000e+00,
                0.00000000e+00,
                -7.04343780e-02,
                -4.63666021e-01,
                -4.63666021e-01,
                5.69295985e-01,
                1.15326283e+04,
                1.23918283e+04,
                9.81422831e+03,
                1.23918283e+04
            ],
            "platform_velocity": [
                0,
                0,
                0
            ],
            "TimeLimit.truncated": "False"
            }
        ]
        }
        """
        # self.obs.append(self.locals["obs"])
        # self.actions.append(self.locals["actions"])
        # self.rewards.append(self.locals["rewards"])
        # self.dones.append(self.locals["dones"])
        # self.next_obs.append(self.locals["new_obs"])

        return True  # continue training

    def save(self, path):
        np.savez(
            path,
            obs=np.array(self.obs, dtype=object),
            actions=np.array(self.actions, dtype=object),
            rewards=np.array(self.rewards, dtype=object),
            dones=np.array(self.dones, dtype=object),
            next_obs=np.array(self.next_obs, dtype=object),
        )

def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True):

    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    def make_training_env(seed):
        def _init():
            task = random_task(sim_dt=SIM_DT, seed=seed)
            return make_env(task, gui=False)
        return _init

    train_env = DummyVecEnv([make_training_env(3)])
    eval_env = make_env(random_task(sim_dt=SIM_DT, seed=4), gui=False)

    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### Train the model #######################################
    # model = PPO('MultiInputPolicy',
    #             train_env,
    #             # tensorboard_log=filename+'/tb/',
    #             verbose=1)

    # "results/save-04.29.2026_11.35.20/final_model.zip",
    model = PPO.load(
        "results/save-04.29.2026_11.35.20/final_model.zip",
        env=train_env,
        verbose=1
    )

    #### Target cumulative rewards (problem-dependent) ##########
    eval_callback = EvalCallback(eval_env,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(1000),
                                 n_eval_episodes=5,
                                 deterministic=True,
                                 render=False)

    # step_callback = StepDataCallback()
    model.learn(total_timesteps=int(1e6) if local else int(1e2), # shorter training in GitHub Actions pytest
                callback=[eval_callback],
                log_interval=100)

    #### Save the model ########################################
    model.save(filename+'/final_model.zip')
    print(filename)

    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        timesteps = data['timesteps']
        results = data['results'][:, 0] 
        print("Data from evaluations.npz")
        for j in range(timesteps.shape[0]):
            print(f"{timesteps[j]},{results[j]}")
        if local:
            plt.plot(timesteps, results, marker='o', linestyle='-', markersize=4)
            plt.xlabel('Training Steps')
            plt.ylabel('Episode Reward')
            plt.grid(True, alpha=0.6)
            plt.show()
    return

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    # parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
