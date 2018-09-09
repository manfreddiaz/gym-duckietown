import math
import numpy as np

POSITION_THRESHOLD = 0.1 # 10 cm
VELOCITY_THRESHOLD = 0.5

class UAPurePursuitPolicy:
    def __init__(self, env, following_distance, max_iterations=1000, refresh_rate=0.1):
        self.env = env
        self.following_distance = following_distance
        self.max_iterations = max_iterations


    def predict(self, observation, metadata):

        closest_point, closest_tangent = self.env.closest_curve_point(self.env.cur_pos)
        iterations = 0
        lookup_distance = self.following_distance
        while iterations < self.max_iterations:
            # Project a point ahead along the curve tangent,
            # then find the closest point to to that
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, _ = self.env.closest_curve_point(follow_point)

            # If we have a valid point on the curve, stop
            if curve_point is not None:
                # print('found at: {}'.format(iterations))
                break

            iterations += 1
            lookup_distance *= 0.5

        if iterations == self.max_iterations:  # if cannot find a curve point in max iterations
            return None, np.inf

        # Compute a normalized vector to the curve point
        point_vec = curve_point - self.env.cur_pos
        norm = np.linalg.norm(point_vec)
        point_vec /= norm

        dot = np.dot(self.env.get_right_vec(), point_vec)
        velocity = 0.5

        steering = 8 * -dot

        position_diff = np.abs(norm - self.following_distance)
        e_v = velocity - self.env.speed
        velocity_diff = np.abs(e_v)

        action = [velocity + 2 * e_v, steering]

        # print(position_diff, velocity_diff, self.d_velocity, self.time_step, self.env.step_count)

        if position_diff > 0.1 or velocity_diff > 0.5:
            self.time_step = self.env.step_count
            # print('unsafe')
            return action, 0.0
        else:
            if metadata[0] == 0:
                return action, 0.0
            if metadata[1] is None:
                return action, 0.0

        return None, math.inf