import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp, Kp=None, Kd=None):
        self.model = ManiuplatorModel(Tp)
        # PD gains of the outer loop acting on the (linearized) double
        # integrator. Diagonal -> one independent loop per joint. Defaults give
        # critically damped closed-loop poles at s = -5 (s^2 + 10 s + 25).
        self.Kp = np.diag([25., 25.]) if Kp is None else np.asarray(Kp, dtype=float)
        self.Kd = np.diag([10., 10.]) if Kd is None else np.asarray(Kd, dtype=float)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        x = np.reshape(np.asarray(x, dtype=float), -1)
        q = x[:2]
        q_dot = x[2:]
        q_r = np.reshape(np.asarray(q_r, dtype=float), -1)
        q_r_dot = np.reshape(np.asarray(q_r_dot, dtype=float), -1)
        q_r_ddot = np.reshape(np.asarray(q_r_ddot, dtype=float), -1)

        # New control of the double integrator: feed-forward acceleration plus
        # PD feedback on the tracking error (eq. (27)). Setting Kp = Kd = 0
        # recovers the pure open-loop feed-forward case (v = q_r_ddot).
        v = q_r_ddot + self.Kd @ (q_r_dot - q_dot) + self.Kp @ (q_r - q)

        # Feedback linearization: tau = M(q) v + C(q, q_dot) q_dot (eq. (25)).
        tau = self.model.M(x) @ v[:, np.newaxis] + self.model.C(x) @ q_dot[:, np.newaxis]
        return tau
