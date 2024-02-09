import numpy as np

from gym_pybullet_drones.envs.ProgressionRLAviary import ProgressionRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class ProgressionAviary(ProgressionRLAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def __init__(self,
                 waypoints=None,
                 window_size=1,
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
        self.EPISODE_LEN_SEC = 8
        super().__init__(waypoints=waypoints,
                         window_size=window_size,
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

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        # state vector (29, ): pos,quat,rpy,vel,ang_v,last_clipped_action,rot
        self.TARGET_POS = self.waypoints[0,:]
        state = self._getDroneStateVector(0)
        b = 1e-3
        c = 1e-6
        ret = np.linalg.norm(self.TARGET_POS-self.prev_pos) - np.linalg.norm(self.TARGET_POS-state[0:3])-c*np.linalg.norm(state[13:16])
        # ret = max(0,2-np.linalg.norm(self.TARGET_POS-state[0:3]))**2-b*np.linalg.norm(state[10:13])-c*np.linalg.norm(state[13:16])
        #normalize distance so that progression on same magnitude
        self.prev_pos = state[0:3]

        # ret = max(0, 1 - np.linalg.norm(self.TARGET_POS - state[0:3])) - b * np.linalg.norm(state[13:16])
        if np.linalg.norm(self.TARGET_POS - state[0:3]) < .01:
            ret = 10
        if state[2]<0.01:
            ret = -10
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
        if (state[2] < .05 or state[8] >= np.pi/2 or state[9] >= np.pi/2
                or np.linalg.norm(self.TARGET_POS - state[0:3]) < .0001):
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
