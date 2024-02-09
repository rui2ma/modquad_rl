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
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger_Modquad import Logger
from gym_pybullet_drones.envs.HoverAviary_Modquad import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, Physics

import matplotlib.pyplot as plt


DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = True
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
# DEFAULT_ACT = ActionType('one_d_rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_ACT = ActionType('rpm')
DEFAULT_AGENTS = 2
DEFAULT_MA = False

def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True, eval=False):
    if not eval:
        filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
        if not os.path.exists(filename):
            os.makedirs(filename+'/')

        if not multiagent:
            train_env = make_vec_env(HoverAviary,
                                    env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                    n_envs=1,
                                    seed=0
                                    )
            eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)

        else:
            train_env = make_vec_env(MultiHoverAviary,
                                    env_kwargs=dict(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT),
                                    n_envs=1,
                                    seed=0
                                    )
            eval_env = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)

        #### Check the environment's spaces ########################
        print('[INFO] Action space:', train_env.action_space)
        print('[INFO] Observation space:', train_env.observation_space)

        #### Train the model #######################################
        model = PPO('MlpPolicy',
                    train_env,
                    policy_kwargs=dict(activation_fn=torch.nn.Tanh, net_arch=[dict(vf=[256,256], pi=[256,256])]),   #vf: actor, pi: critic
                    # tensorboard_log=filename+'/tb/',
                    verbose=1)

        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=np.inf,
                                                        verbose=1)
        eval_callback = EvalCallback(eval_env,
                                    callback_on_new_best=callback_on_best,
                                    verbose=1,
                                    best_model_save_path=filename+'/',
                                    log_path=filename+'/',
                                    eval_freq=int(2000),
                                    deterministic=True,
                                    render=False)
        model.learn(total_timesteps=3*int(1e5) if local else int(1e2), # shorter training in GitHub Actions pytest
                    callback=eval_callback,
                    log_interval=100)

        #### Save the model ########################################
        model.save(filename+'/success_model.zip')

        #### Print training progression ############################
        with np.load(filename+'/evaluations.npz') as data:
            for j in range(data['timesteps'].shape[0]):
                print(str(data['timesteps'][j])+","+str(data['results'][j][0]))
    else:
        filename=os.path.join(output_folder, "2x1_1waypoint_far")
    ############################################################
    ############################################################
    ############################################################
    ############################################################
    ############################################################

    if os.path.isfile(filename+'/success_model.zip'):
        path = filename+'/success_model.zip'
    elif os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    model = PPO.load(path)

    #### Show (and record a video of) the model's performance ##
    if not multiagent:
        test_env = HoverAviary(gui=gui,
                               obs=DEFAULT_OBS,
                               act=DEFAULT_ACT,
                               initial_xyzs=np.array([[-0.5,-0.5,0.05]]),
                               record=record_video)
        test_env_nogui = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    else:
        test_env = MultiHoverAviary(gui=gui,
                                    num_drones=DEFAULT_AGENTS,
                                    obs=DEFAULT_OBS,
                                    act=DEFAULT_ACT,
                                    record=record_video)
        test_env_nogui = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=DEFAULT_AGENTS if multiagent else 1,
                output_folder=output_folder,
                colab=colab
                )
    rewards = []
    # mean_reward, std_reward = evaluate_policy(model,
    #                                           test_env_nogui,
    #                                           n_eval_episodes=10
    #                                           )
    # print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info, rpy = test_env.step_test(action)
        rewards.append(reward)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        if action.shape[1]==1:  # one_d_rpm
            log_state = np.hstack([obs2[0:3],
                                    np.zeros(4),
                                    rpy,
                                    obs2[12:15],
                                    obs2[15:],
                                    np.array([act2]*4)
                                    ])
        else:
            log_state = np.hstack([obs2[0:3],
                                    np.zeros(4),
                                    rpy,
                                    obs2[12:15],
                                    obs2[15:],
                                    act2,
                                    ])

        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        if DEFAULT_OBS == ObservationType.KIN:
            if not multiagent:
                logger.log(drone=0,
                    timestamp=i/test_env.CTRL_FREQ,
                    state=log_state,
                    control=np.zeros(12)
                    )
            else:
                for d in range(DEFAULT_AGENTS):
                    logger.log(drone=d,
                        timestamp=i/test_env.CTRL_FREQ,
                        state=np.hstack([obs2[d][0:3],
                                            np.zeros(4),
                                            obs2[d][3:15],
                                            act2[d]
                                            ]),
                        control=np.zeros(12)
                        )
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    plt.plot(rewards)
    plt.title("rewards progression")
    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()
        logger.plot_rpms()
if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--eval',               default=False,                 type=bool,          help='Whether run evaluation of the model', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))

#
# DEFAULT_GUI = True
# DEFAULT_RECORD_VIDEO = True
# DEFAULT_OUTPUT_FOLDER = 'results'
# DEFAULT_COLAB = False
#
# DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
# DEFAULT_ACT = ActionType('one_d_rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
# # DEFAULT_ACT = ActionType('rpm')
# DEFAULT_AGENTS = 2
# DEFAULT_MA = False
#
# def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True, eval=False):
#     if not eval:
#         filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
#         if not os.path.exists(filename):
#             os.makedirs(filename+'/')
#
#         if not multiagent:
#             train_env = make_vec_env(HoverAviary,
#                                     env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
#                                     n_envs=1,
#                                     seed=0
#                                     )
#             eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
#         else:
#             train_env = make_vec_env(MultiHoverAviary,
#                                     env_kwargs=dict(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT),
#                                     n_envs=1,
#                                     seed=0
#                                     )
#             eval_env = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)
#
#         #### Check the environment's spaces ########################
#         print('[INFO] Action space:', train_env.action_space)
#         print('[INFO] Observation space:', train_env.observation_space)
#
#         #### Train the model #######################################
#         # model = PPO('MlpPolicy',
#         #             train_env,
#         #             # policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=[512, 512, dict(vf=[256, 128], pi=[256, 128])]),
#         #             # tensorboard_log=filename+'/tb/',
#         #             verbose=1,
#         #             device="cpu")
#         model = PPO('MultiInputPolicy',
#                     train_env,
#                     # policy_kwargs=dict(activation_fn=torch.nn.ReLU, net_arch=[512, 512, dict(vf=[256, 128], pi=[256, 128])]),
#                     # tensorboard_log=filename+'/tb/',
#                     verbose=1,
#                     device="cpu")
#
#         callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=np.inf,
#                                                         verbose=1)
#         eval_callback = EvalCallback(eval_env,
#                                     callback_on_new_best=callback_on_best,
#                                     verbose=1,
#                                     best_model_save_path=filename+'/',
#                                     log_path=filename+'/',
#                                     eval_freq=int(2000),
#                                     deterministic=True,
#                                     render=False)
#         # local = False
#         model.learn(total_timesteps=3*int(1e5) if local else int(1e3), # shorter training in GitHub Actions pytest
#                     callback=eval_callback,
#                     log_interval=100)
#
#         #### Save the model ########################################
#         model.save(filename+'/success_model.zip')
#         print(filename)
#
#         #### Print training progression ############################
#         with np.load(filename+'/evaluations.npz') as data:
#             for j in range(data['timesteps'].shape[0]):
#                 print(str(data['timesteps'][j])+","+str(data['results'][j][0]))
#     else:
#         filename=os.path.join(output_folder, "2x1_hover_oscillate")
#     ############################################################
#     ############################################################
#     ############################################################
#     ############################################################
#     ############################################################
#
#     if os.path.isfile(filename+'/success_model.zip'):
#         path = filename+'/success_model.zip'
#     elif os.path.isfile(filename+'/best_model.zip'):
#         path = filename+'/best_model.zip'
#     else:
#         print("[ERROR]: no model under the specified path", filename)
#     model = PPO.load(path)
#
#     #### Show (and record a video of) the model's performance ##
#     if not multiagent:
#         test_env = HoverAviary(initial_xyzs=np.array([[0,0,0.1]]),
#                                gui=gui,
#                                obs=DEFAULT_OBS,
#                                act=DEFAULT_ACT)
#         test_env_nogui = HoverAviary(initial_xyzs=np.array([[0,0,0.1]]), obs=DEFAULT_OBS, act=DEFAULT_ACT, physics=Physics.PYB_DRAG)
#     else:
#         test_env = MultiHoverAviary(gui=gui,
#                                         num_drones=DEFAULT_AGENTS,
#                                         obs=DEFAULT_OBS,
#                                         act=DEFAULT_ACT,
#                                         record=record_video)
#         test_env_nogui = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT, physics=Physics.PYB_DRAG)
#     logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
#                 num_drones=DEFAULT_AGENTS if multiagent else 1,
#                 output_folder=output_folder,
#                 colab=colab
#                 )
#
#     mean_reward, std_reward = evaluate_policy(model,
#                                               test_env_nogui,
#                                               n_eval_episodes=10
#                                               )
#     print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")
#     # mean_reward_log.append(mean_reward)
#     rewards = []
#     obs, info = test_env.reset(seed=42, options={})
#     start = time.time()
#     for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
#         action, _states = model.predict(obs,
#                                         deterministic=True
#                                         )
#         obs, reward, terminated, truncated, info = test_env.step(action)
#         rewards.append(reward)
#         obs2 = obs.squeeze()
#         act2 = action.squeeze()
#         print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
#         if DEFAULT_OBS == ObservationType.KIN:
#             if not multiagent:
#                 logger.log(drone=0,
#                     timestamp=i/test_env.CTRL_FREQ,
#                     state=np.hstack([obs2[0:3],
#                                         np.zeros(4),
#                                         obs2[3:15],
#                                         act2
#                                         ]),
#                     control=np.zeros(12)
#                     )
#             else:
#                 for d in range(DEFAULT_AGENTS):
#                     logger.log(drone=d,
#                         timestamp=i/test_env.CTRL_FREQ,
#                         state=np.hstack([obs2[d][0:3],
#                                             np.zeros(4),
#                                             obs2[d][3:15],
#                                             act2[d]
#                                             ]),
#                         control=np.zeros(12)
#                         )
#         test_env.render()
#         print(terminated)
#         sync(i, start, test_env.CTRL_TIMESTEP)
#         if terminated:
#             obs = test_env.reset(seed=42, options={})
#     test_env.close()
#     plt.plot(rewards)
#
#     if plot and DEFAULT_OBS == ObservationType.KIN:
#         logger.plot()
#
# if __name__ == '__main__':
#     #### Define and parse (optional) arguments for the script ##
#     parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
#     parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
#     parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
#     parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
#     parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
#     parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
#     parser.add_argument('--eval',               default=False,                 type=bool,          help='Whether run evaluation of the model', metavar='')
#     ARGS = parser.parse_args()
#
#     run(**vars(ARGS))
