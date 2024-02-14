import numpy as np

from gym_pybullet_drones.envs.ProgressionRLAviary import ProgressionRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class ProgressionAviary(ProgressionRLAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def __init__(self,
                 waypoints=None,
                 window_size=2,
                 episode_len_second=10,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 test_flag = False
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.EPISODE_LEN_SEC = episode_len_second
        super().__init__(waypoints=waypoints,
                         window_size=window_size,
                         episode_len_second=episode_len_second,
                         drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         test_flag = test_flag
                         )
        self.prev_pos = self.INIT_XYZS
        self.TARGET_POS = self.waypoints[0, :]

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        # state vector (29, ): pos,quat,rpy,vel,ang_v,last_clipped_action,rot
        state = self._getDroneStateVector(0)
        b = 1e-6
        c = 1e-6
        if self.VISITED_IDX >= self.waypoints.shape[0]:    #finished all waypoints
            ret = 10
        else:
            self.TARGET_POS = self.waypoints[self.VISITED_IDX, :]
            ret = np.sqrt(max(0, 1 - np.linalg.norm(self.TARGET_POS - state[0:3]))) - b * np.linalg.norm(
                state[10:13]) - c * np.linalg.norm(state[13:16]) + self.cum_reward
            if np.linalg.norm(self.TARGET_POS-state[0:3]) < 0.01:
                self.VISITED_IDX += 1
                self.cum_reward = ret
        self.max_pts_reached = self.VISITED_IDX
        if state[2]<0.01:
            ret = -10

        # state = self._getDroneStateVector(0)
        # b = 1e-6
        # c = 1e-6
        # if self.VISITED_IDX > self.waypoints.shape[0] - 1:  # finished all waypoints
        #     ret = 10
        # else:
        #     self.TARGET_POS = self.waypoints[self.VISITED_IDX, :]
        #     ret = (np.linalg.norm(self.TARGET_POS - self.prev_pos) - np.linalg.norm(self.TARGET_POS - state[0:3]) -
        #            b*np.linalg.norm(state[13:16]))
        #     self.prev_pos = state[0:3]
        #     if np.linalg.norm(self.TARGET_POS - state[0:3]) < 0.01:
        #         self.VISITED_IDX += 1
        #         # self.cum_reward = ret
        # self.max_pts_reached = self.VISITED_IDX
        # if state[2] < 0.01:
        #     ret = -10
        return ret


    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if (state[2] < .05 or self.VISITED_IDX >= self.waypoints.shape[0]):
            return True
        else:
            return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
