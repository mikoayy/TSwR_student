import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.integrate import odeint

from controllers.adrc_controller import ADRController

from trajectory_generators.constant_torque import ConstantTorque
from trajectory_generators.sinusonidal import Sinusoidal
from trajectory_generators.poly3 import Poly3
from utils.simulation import simulate

Tp = 0.001
end = 5

# traj_gen = ConstantTorque(np.array([0., 1.0])[:, np.newaxis])
traj_gen = Sinusoidal(np.array([0., 1.]), np.array([2., 2.]), np.array([0., 0.]))
# traj_gen = Poly3(np.array([0., 0.]), np.array([pi/4, pi/6]), end)

# Rough input-gain guesses b_i ~ (M^-1)_ii for each joint (see eq. (53));
# ADRC is robust to this estimate, the ESO absorbs the rest into f.
b_est_1 = 2.5
b_est_2 = 20.
# Outer-loop PD gains (double integrator), critically damped poles at -20 rad/s
# (s^2 + 40 s + 400 = (s + 20)^2).
kp_est_1 = 400.
kp_est_2 = 400.
kd_est_1 = 40.
kd_est_2 = 40.
# Observer bandwidth (triple pole at -p_i). Kept ~10x faster than the control
# loop so the ESO reconstructs the total disturbance before the PD acts on it.
p1 = 200.
p2 = 200.

q0, qdot0, _ = traj_gen.generate(0.)
q1_0 = np.array([q0[0], qdot0[0]])
q2_0 = np.array([q0[1], qdot0[1]])
controller = ADRController(Tp, params=[[b_est_1, kp_est_1, kd_est_1, p1, q1_0],
                                       [b_est_2, kp_est_2, kd_est_2, p2, q2_0]])

Q, Q_d, u, T = simulate("PYBULLET", traj_gen, controller, Tp, end)

eso1 = np.array(controller.joint_controllers[0].eso.states)
eso2 = np.array(controller.joint_controllers[1].eso.states)

plt.subplot(221)
plt.plot(T, eso1[:, 0])
plt.plot(T, Q[:, 0], 'r')
plt.subplot(222)
plt.plot(T, eso1[:, 1])
plt.plot(T, Q[:, 2], 'r')
plt.subplot(223)
plt.plot(T, eso2[:, 0])
plt.plot(T, Q[:, 1], 'r')
plt.subplot(224)
plt.plot(T, eso2[:, 1])
plt.plot(T, Q[:, 3], 'r')
plt.show()

plt.subplot(221)
plt.plot(T, Q[:, 0], 'r')
plt.plot(T, Q_d[:, 0], 'b')
plt.subplot(222)
plt.plot(T, Q[:, 1], 'r')
plt.plot(T, Q_d[:, 1], 'b')
plt.subplot(223)
plt.plot(T, u[:, 0], 'r')
plt.plot(T, u[:, 1], 'b')
plt.show()
