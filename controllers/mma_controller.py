import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class MMAController(Controller):
    def __init__(self, Tp):
        # Three candidate models of the 2DoF manipulator differing only in the
        # tip object (m3, r3). The set matches the objects the PyBullet plant
        # may switch to (see PlanarManipulator2DOFPyBullet.objects_params).
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        self.models = [
            ManiuplatorModel(Tp, m3=0.1, r3=0.05),
            ManiuplatorModel(Tp, m3=0.01, r3=0.01),
            ManiuplatorModel(Tp, m3=1.0, r3=0.3),
        ]
        self.i = 0
        self.Tp = Tp
        # Memory needed to predict the current state from the previous one.
        self.x_prev = None
        self.u_prev = np.zeros((2, 1))
        # PD gains of the feedback-linearizing outer loop (same role as in the
        # FeedbackLinearizationController).
        self.Kp = np.diag([25., 25.])
        self.Kd = np.diag([10., 10.])

    def choose_model(self, x):
        # Predict the current state with every candidate model (one explicit
        # Euler step from the previously measured state and applied control) and
        # select the model whose prediction is closest to the measured state
        # (eq. (32): argmin_i ||x - x_mi||).
        x = np.reshape(np.asarray(x, dtype=float), -1)
        if self.x_prev is None:
            return
        errors = []
        for model in self.models:
            x_pred = self.x_prev + self.Tp * model.x_dot(self.x_prev, self.u_prev).flatten()
            errors.append(np.linalg.norm(x - x_pred))
        self.i = int(np.argmin(errors))

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        x = np.reshape(np.asarray(x, dtype=float), -1)
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        # Feedback-linearizing control with PD feedback, using the matrices of
        # the currently selected best model.
        v = np.asarray(q_r_ddot, dtype=float) \
            + self.Kd @ (np.asarray(q_r_dot, dtype=float) - q_dot) \
            + self.Kp @ (np.asarray(q_r, dtype=float) - q)
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        # Store data needed by choose_model on the next step.
        self.x_prev = x.copy()
        self.u_prev = u
        return u
