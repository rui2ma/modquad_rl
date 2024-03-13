"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes HoverAviary and MultiHoverAviary are used as learning envs for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py --multiagent false
    $ python learn.py --multiagent true

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with
reinforcement learning library `stable-baselines3`.

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (EvalCallback, StopTrainingOnRewardThreshold,
                                                CallbackList, StopTrainingOnNoModelImprovement)

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize



from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.ProgressionAviary import ProgressionAviary
from gym_pybullet_drones.envs.ProgressionDemoAviary_Modquad import ProgressionDemoAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
import matplotlib.pyplot as plt
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ImageType


from gym_pybullet_drones.utils.TensorboardCallback import TensorboardCallback


DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = True
DEFAULT_OUTPUT_FOLDER = 'results_modquad/temp'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
# DEFAULT_ACT = ActionType('one_d_rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_ACT = ActionType('rpm')
DEFAULT_AGENTS = 2
DEFAULT_MA = False

def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True, eval=False):
    
    drone_model = DroneModel.PLUS
    filename=os.path.join(output_folder, "")
    ############################################################
    ############################################################
    ############################################################
    ############################################################
    ############################################################

    filename=os.path.join(output_folder, "plus_with_yaw")
    if os.path.isfile(filename+'/success_model.zip'):
        path = filename+'/success_model.zip'
    elif os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)

    model = PPO.load(path)

    #### Show (and record a video of) the model's performance ##

    test_env = ProgressionDemoAviary(waypoints=np.array([[-1,-1,0.2], 
                                                         [1,-1,0.2],
                                                         [1,1,0.2],
                                                         [-1,1,0.2], 
                                                         [-1,-1,0.2],
                                                         [-1,-1,1.2], 
                                                         [1,-1,1.2],
                                                         [1,1,1.2],
                                                         [-1,1,1.2], 
                                                         [-1,-1,1.2]]),
                                                         
                                     initial_xyzs=np.array([[0,0,0.5]]),
                                     drone_model = drone_model,
                                     initial_rpys=np.array([[0, 0,0]]),
                                     test_flag=True,
                                     gui=gui,
                                     record=record_video)
                                     
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                    num_drones=DEFAULT_AGENTS if multiagent else 1,
                    output_folder=output_folder,
                    colab=colab
                    )
    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC*300)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        log_state = np.hstack([obs2[0:3],
                               np.zeros(7),
                               obs2[12:15],
                               obs2[15:],
                               act2,
                               ])
        logger.log(drone=0,
                   timestamp=i / test_env.CTRL_FREQ,
                   state=log_state,
                   control=np.zeros(12)
                   )
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        test_env.render()
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            break
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--eval',               default=False,                 type=bool,          help='Whether to run evaluation of the model', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))