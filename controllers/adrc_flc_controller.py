import numpy as np

from observers.eso import ESO
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.Tp = Tp
        self.model = ManiuplatorModel(Tp)
        self.Kp = np.asarray(Kp, dtype=float)
        self.Kd = np.asarray(Kd, dtype=float)

        # Centralized extended state: z = [q1, q2, q1_dot, q2_dot, f1, f2]^T,
        # where [f1, f2] is the total disturbance to estimate (eq. (58)).
        # Observer gains: each of the two channels is a triple integrator, so we
        # place the poles at (lambda + p_i)^3 -> [3p, 3p^2, p^3] per channel.
        p = np.asarray(p, dtype=float)
        self.L = np.array([[3 * p[0], 0.],
                           [0., 3 * p[1]],
                           [3 * p[0] ** 2, 0.],
                           [0., 3 * p[1] ** 2],
                           [p[0] ** 3, 0.],
                           [0., p[1] ** 3]])
        # Output matrix: only the positions [q1, q2] are measured.
        W = np.concatenate([np.eye(2), np.zeros((2, 4))], axis=1)

        # A and B depend on the (estimated) model and are filled by
        # update_params; provide a consistent placeholder for the ESO ctor.
        A = np.zeros((6, 6))
        A[0:2, 2:4] = np.eye(2)
        A[2:4, 4:6] = np.eye(2)
        B = np.zeros((6, 2))
        self.eso = ESO(A, B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        # Rebuild the model-dependent blocks of the extended-state dynamics
        # (eq. (59)) at the current operating point.
        x = np.array([q[0], q[1], q_dot[0], q_dot[1]])
        M = self.model.M(x)
        C = self.model.C(x)
        invM = np.linalg.inv(M)

        A = np.zeros((6, 6))
        A[0:2, 2:4] = np.eye(2)          # q_dot
        A[2:4, 2:4] = -invM @ C          # -M^-1 C
        A[2:4, 4:6] = np.eye(2)          # disturbance enters the acceleration
        self.eso.A = A

        B = np.zeros((6, 2))
        B[2:4, 0:2] = invM               # M^-1
        self.eso.B = B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        x = np.reshape(np.asarray(x, dtype=float), -1)
        q = x[:2]
        # Estimates from the ESO.
        z = self.eso.get_state()
        q_dot_hat = z[2:4]
        f_hat = z[4:6]

        # Refresh the model matrices at the current (estimated) operating point.
        self.update_params(q, q_dot_hat)

        # Outer-loop control (eq. (61)): v = Kp e + Kd e_dot + q_d_ddot, using
        # measured positions and estimated velocities.
        e = np.asarray(q_d, dtype=float) - q
        e_dot = np.asarray(q_d_dot, dtype=float) - q_dot_hat
        v = self.Kp @ e + self.Kd @ e_dot + np.asarray(q_d_ddot, dtype=float)

        # Control law (eq. (60)): u = M_hat (v - f_hat) + C_hat q_dot_hat.
        xz = np.concatenate([q, q_dot_hat])
        M = self.model.M(xz)
        C = self.model.C(xz)
        u = M @ (v - f_hat)[:, np.newaxis] + C @ q_dot_hat[:, np.newaxis]

        # Propagate the observer with measured positions and applied control.
        self.eso.update(q, u)
        return u
