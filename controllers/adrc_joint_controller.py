import numpy as np
from observers.eso import ESO
from .controller import Controller


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd
        self.Tp = Tp

        # Extended state of a single joint: z = [q, q_dot, f], where f is the
        # total disturbance to be estimated (Section 12).
        A = np.array([[0., 1., 0.],
                      [0., 0., 1.],
                      [0., 0., 0.]])
        B = np.array([[0.],
                      [b],
                      [0.]])
        # Only the position is measured -> output matrix W = C = [1, 0, 0].
        W = np.array([[1., 0., 0.]])
        # Observer gains from pole placement of (A - LC) at (lambda + p)^3 = 0
        # (eq. (52)): l1 = 3p, l2 = 3p^2, l3 = p^3.
        L = np.array([[3 * p],
                      [3 * p ** 2],
                      [p ** 3]])
        self.eso = ESO(A, B, W, L, q0, Tp)

    def set_b(self, b):
        # Update the input gain estimate b both in the controller and in the
        # observer input matrix B = [0, b, 0]^T (used by the improved variant
        # where b_hat = (M^-1)_ii(q), eq. (53)).
        self.b = b
        self.eso.set_B(np.array([[0.], [b], [0.]]))

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        q = x[0]
        # Current estimates from the ESO: [q_hat, q_dot_hat, f_hat].
        q_hat, q_dot_hat, f_hat = self.eso.get_state()
        # Outer-loop control of the (estimated) double integrator: feed-forward
        # acceleration + PD feedback. The measured position is used directly,
        # the unmeasured velocity is replaced by its estimate.
        v = q_d_ddot + self.kd * (q_d_dot - q_dot_hat) + self.kp * (q_d - q)
        # Reject the estimated total disturbance and scale by 1/b (eq. (37)).
        u = (v - f_hat) / self.b
        # Propagate the observer with the measured position and applied control.
        self.eso.update(q, u)
        return u
