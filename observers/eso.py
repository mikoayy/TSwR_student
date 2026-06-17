from copy import copy
import numpy as np


class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.state = np.pad(np.array(state, dtype=float), (0, A.shape[0] - len(state)))
        self.Tp = Tp
        self.states = []

    def set_B(self, B):
        self.B = B

    def update(self, q, u):
        self.states.append(copy(self.state))
        # Extended state observer (eq. (41)/(59)):
        #   z_hat_dot = A z_hat + B u + L (y - W z_hat)
        # integrated with one explicit Euler step. W (a.k.a. C) extracts the
        # measurable part (positions) of the extended state.
        z = self.state
        y = np.reshape(np.asarray(q, dtype=float), -1)
        u_vec = np.reshape(np.asarray(u, dtype=float), -1)
        z_dot = self.A @ z + self.B @ u_vec + self.L @ (y - self.W @ z)
        self.state = z + self.Tp * z_dot

    def get_state(self):
        return self.state
