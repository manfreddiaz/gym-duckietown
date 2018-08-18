import numpy as np
from controllers import Controller


class UncertaintyAwarePurePursuitController(Controller):
    def __init__(self, env, following_distance, max_iterations=1000, refresh_rate=0.1):
        Controller.__init__(self, env, refresh_rate)
        self.following_distance = following_distance
        self.max_iterations = max_iterations

    def _do_update(self, dt):
        return self.predict()

    def predict(self):
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
                break

            lookup_distance *= 0.5

        if iterations == self.max_iterations:  # if cannot find a curve point in max iterations
            return None, np.inf

        # Compute a normalized vector to the curve point
        point_vec = curve_point - self.env.cur_pos
        point_vec /= np.linalg.norm(point_vec)

        print(point_vec)

        dot = np.dot(self.env.get_right_vec(), point_vec)
        velocity = 0.35

        steering = 2 * -dot

        return [velocity, steering], 0.0
