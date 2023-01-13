import numpy as np


def angle_to_state(angle):
    return int(30 * ((angle + np.pi) / (2 * np.pi) % 1))  # Discretization of the angle space


def x_to_state(x):
    return int(40 * ((x + -10) / 20))  # Discretization of the x space


def vel(theta, theta_0=0, theta_dead=np.pi / 12):
    return 1 - np.exp(-(theta - theta_0) ** 2 / theta_dead)


def rew(theta, theta_0=0, theta_dead=np.pi / 12):
    return vel(theta, theta_0, theta_dead) * np.cos(theta)
