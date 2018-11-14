import math
import numpy as np

POSITION_THRESHOLD = 0.04
REF_VELOCITY = 0.8
GAIN = 10
FOLLOWING_DISTANCE = 0.3



class UAPurePursuitPolicy:
    def __init__(self, env, ref_velocity=REF_VELOCITY, position_threshold=POSITION_THRESHOLD,
                 following_distance=FOLLOWING_DISTANCE, max_iterations=1000):
        self.env = env
        self.following_distance = following_distance
        self.max_iterations = max_iterations
        self.ref_velocity = ref_velocity
        self.position_threshold = position_threshold

        self.gain = 1.0
        self.trim = 0.0
        self.radius = 0.0318
        self.k = 27.0
        self.limit = 1.0
        self.wheel_dist = 0.102


    def inverse_kinematics(self, action):
        vel, angle = action

        # Distance between the wheels
        baseline = self.wheel_dist

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

        return vels


    def predict(self, observation, metadata):
        closest_point, closest_tangent = self.env.closest_curve_point(self.env.cur_pos, self.env.cur_angle)

        iterations = 0
        lookup_distance = self.following_distance
        curve_point = None
        while iterations < self.max_iterations:
            # Project a point ahead along the curve tangent,
            # then find the closest point to to that
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, _ = self.env.closest_curve_point(follow_point, self.env.cur_angle)

            # If we have a valid point on the curve, stop
            if curve_point is not None:
                break

            iterations += 1
            lookup_distance *= 0.5

        if iterations == self.max_iterations:  # if cannot find a curve point in max iterations
            return None, np.inf

        # Compute a normalized vector to the curve point
        point_vec = curve_point - self.env.cur_pos
        point_vec /= np.linalg.norm(point_vec)

        dot = np.dot(self.env.get_right_vec(), point_vec)
        steering = GAIN * -dot

        position_diff = np.linalg.norm(closest_point - self.env.cur_pos, ord=1)

        action = [self.ref_velocity, steering]
        action = self.inverse_kinematics(action)

        # print(position_diff, velocity_diff)

        if position_diff > self.position_threshold: # or velocity_diff > 0.5:
            return action, 0.0
        else:
            if metadata[0] == 0:
                return action, 0.0
            if metadata[1] is None:
                return action, 0.0

        return None, math.inf
