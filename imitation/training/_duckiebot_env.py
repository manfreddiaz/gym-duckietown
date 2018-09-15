import numpy as np
from gym import spaces

from duckietown_slimremote.pc.robot import RemoteRobot


class DuckiebotEnvIcra2019(RemoteRobot):
    def __init__(
            self,
            host,
            gain=-1.0,
            trim=0.0,
            radius=0.0318,
            k=27.0,
            limit=1.0
    ):
        RemoteRobot.__init__(self, host=host, silent=True)

        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )

        # Should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = gain

        # Directional trim adjustment
        self.trim = trim

        # Wheel radius
        self.radius = radius

        # Motor constant
        self.k = k

        # Wheel velocity limit
        self.limit = limit

    def step(self, action, with_observation=True):
        vel = 0.4
        angle = action

        # Distance between the wheels
        baseline = 0.13

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * baseline) / self.radius
        omega_l = (vel - 0.5 * angle * baseline) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])

        obs, rew, done, misc = RemoteRobot.step(self, vels)
        return obs[:, :, ::-1], rew, done, misc

    def render_obs(self):
        return RemoteRobot.observe(self)[0]
