import numpy as np


class ManiuplatorModel:
    def __init__(self, Tp, m3=0.0, r3=0.05):
        self.Tp = Tp
        # Parameters chosen to match urdf/planar2dof.urdf (the PyBullet truth),
        # so that the model used by the controllers is consistent with the
        # simulated plant. Centers of mass lie in the middle of each link.
        #   link 1: cylinder, length l1, radius r1, mass m1
        #   link 2: cylinder, length l2, radius r2, mass m2
        self.l1 = 0.5
        self.r1 = 0.04
        self.m1 = 3.0
        self.l2 = 0.4
        self.r2 = 0.04
        self.m2 = 2.4
        self.d1 = self.l1 / 2
        self.d2 = self.l2 / 2
        # Moment of inertia of a cylinder about an axis perpendicular to it,
        # passing through its center of mass (rotation around the world z axis).
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        # Object (ball) mounted at the tip of the second link, at distance l2
        # from the second joint. m3 == 0 corresponds to "no object".
        self.m3 = m3
        self.r3 = r3
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2

    def set_object(self, m3, r3):
        """Update the parameters of the object carried at the tip (used by the
        multi-model adaptive controller to build model candidates)."""
        self.m3 = m3
        self.r3 = r3
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2

    def _alpha_beta_gamma(self):
        """Helper coefficients of the mass matrix (see eq. (14) extended with
        the tip object m3, derived in exercise 2.1)."""
        alpha = self.I_1 + self.I_2 \
            + self.m1 * self.d1 ** 2 + self.m2 * (self.l1 ** 2 + self.d2 ** 2) \
            + self.I_3 + self.m3 * (self.l1 ** 2 + self.l2 ** 2)
        beta = self.m2 * self.l1 * self.d2 + self.m3 * self.l1 * self.l2
        gamma = self.I_2 + self.m2 * self.d2 ** 2 + self.I_3 + self.m3 * self.l2 ** 2
        return alpha, beta, gamma

    def M(self, x):
        """Mass matrix M(q) of the 2DoF planar manipulator with a point mass m3
        at the tip of the second link (eq. (20))."""
        q1, q2, q1_dot, q2_dot = x
        alpha, beta, gamma = self._alpha_beta_gamma()
        c2 = np.cos(q2)
        m_11 = alpha + 2 * beta * c2
        m_12 = gamma + beta * c2
        m_21 = m_12
        m_22 = gamma
        return np.array([[m_11, m_12],
                         [m_21, m_22]])

    def C(self, x):
        """Matrix of Coriolis and centrifugal forces C(q, q_dot) (eq. (20))."""
        q1, q2, q1_dot, q2_dot = x
        _, beta, _ = self._alpha_beta_gamma()
        s2 = np.sin(q2)
        c_11 = -beta * s2 * q2_dot
        c_12 = -beta * s2 * (q1_dot + q2_dot)
        c_21 = beta * s2 * q1_dot
        c_22 = 0.
        return np.array([[c_11, c_12],
                         [c_21, c_22]])

    def x_dot(self, x, u):
        """State derivative of the manipulator written as x_dot = A x + b u,
        with x = [q1, q2, q1_dot, q2_dot] and u the joint torques. Used for the
        SciPy simulation and for one-step model prediction in the MMAC."""
        x = np.reshape(np.asarray(x, dtype=float), -1)
        u = np.reshape(np.asarray(u, dtype=float), -1)[:, np.newaxis]
        invM = np.linalg.inv(self.M(x))
        zeros = np.zeros((2, 2))
        A = np.concatenate([np.concatenate([zeros, np.eye(2)], 1),
                            np.concatenate([zeros, -invM @ self.C(x)], 1)], 0)
        b = np.concatenate([zeros, invM], 0)
        return A @ x[:, np.newaxis] + b @ u
