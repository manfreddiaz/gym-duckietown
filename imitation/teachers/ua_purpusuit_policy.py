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

    def predict(self, observation, metadata):
        closest_point, closest_tangent = self.env.closest_curve_point(self.env.cur_pos)

        iterations = 0
        lookup_distance = self.following_distance
        curve_point = None
        while iterations < self.max_iterations:
            # Project a point ahead along the curve tangent,
            # then find the closest point to to that
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, _ = self.env.closest_curve_point(follow_point)

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

        action = steering  # [self.ref_velocity, steering]

        # print(position_diff, velocity_diff)

        if position_diff > self.position_threshold: # or velocity_diff > 0.5:
            return action, 0.0
        else:
            if metadata[0] == 0:
                return action, 0.0
            if metadata[1] is None:
                return action, 0.0

        return None, math.inf
