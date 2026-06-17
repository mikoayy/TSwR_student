import numpy as np
from trajectory_generators.trajectory_generator import TrajectoryGenerator


class Poly3(TrajectoryGenerator):
    def __init__(self, start_q, desired_q, T):
        self.T = T
        self.q_0 = np.asarray(start_q, dtype=float)
        self.q_k = np.asarray(desired_q, dtype=float)
        # 3rd degree polynomial parametrized as in eq. (28):
        #   q_r(t) = a_3 t^3 + a_2 t^2 (1-t) + a_1 t (1-t)^2 + a_0 (1-t)^3,  t in [0,1]
        # Coefficients follow from the boundary conditions (eq. (29)) assuming
        # zero start and end velocities (q_dot_0 = q_dot_k = 0):
        #   q_r(0) = a_0 = q_0,        q_r(1) = a_3 = q_k
        #   q_dot_r(0) = a_1 - 3 a_0 = 0  ->  a_1 = 3 q_0
        #   q_dot_r(1) = 3 a_3 - a_2 = 0  ->  a_2 = 3 q_k
        self.a_0 = self.q_0
        self.a_1 = 3 * self.q_0
        self.a_2 = 3 * self.q_k
        self.a_3 = self.q_k

    def generate(self, t):
        # Hold the boundary configurations outside the motion interval so the
        # generator can also be used as a point-to-point set-point.
        t = np.clip(t, 0., self.T)
        t /= self.T
        q = self.a_3 * t ** 3 + self.a_2 * t ** 2 * (1 - t) \
            + self.a_1 * t * (1 - t) ** 2 + self.a_0 * (1 - t) ** 3
        # First derivative with respect to the normalized time t (eq. (30)).
        q_dot = 3 * self.a_3 * t ** 2 + self.a_2 * (2 * t - 3 * t ** 2) \
            + self.a_1 * (1 - 4 * t + 3 * t ** 2) - 3 * self.a_0 * (1 - t) ** 2
        # Second derivative with respect to the normalized time t (eq. (31)).
        q_ddot = 6 * self.a_3 * t + self.a_2 * (2 - 6 * t) \
            + self.a_1 * (-4 + 6 * t) + 6 * self.a_0 * (1 - t)
        # Rescale the derivatives back to the real time axis (chain rule).
        return q, q_dot / self.T, q_ddot / self.T ** 2
