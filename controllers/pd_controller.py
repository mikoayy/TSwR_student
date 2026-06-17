import numpy as np
from .controller import Controller


class PDDecentralizedController(Controller):
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def calculate_control(self, q, q_dot, q_d, q_d_dot, q_d_ddot):
        # Decentralized PD control (eq. (33)-(35)): each joint is treated
        # independently, u_i = kp_i e_i + kd_i e_dot_i, where the gains may be
        # scalars (same for both joints) or per-joint arrays.
        e = np.asarray(q_d, dtype=float) - np.asarray(q, dtype=float)
        e_dot = np.asarray(q_d_dot, dtype=float) - np.asarray(q_dot, dtype=float)
        u = self.kp * e + self.kd * e_dot
        return u
