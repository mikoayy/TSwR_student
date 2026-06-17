from trajectory_generators.poly3 import Poly3


class Point2Point(Poly3):
    """Point-to-point set-point generator: smoothly drive the manipulator from
    the configuration ``start_q`` to ``desired_q`` within ``T`` seconds and hold
    ``desired_q`` afterwards.

    It reuses the 3rd degree polynomial of :class:`Poly3` (zero boundary
    velocities and accelerations at the endpoints), which already clamps the
    time to ``[0, T]`` and therefore keeps the final configuration once the
    motion is finished.
    """
    pass
